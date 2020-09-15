# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import datetime
import errno
import os
import sys
from time import time

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from common.arguments import parse_args
from common.h36m_dataset import h36m_skeleton
from common.loss import *
from common.model import *
from data.load_dataset import load_dataset, build_generator

args = parse_args()
print(args)

# set seed
np.random.seed(1234)

try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

writer = SummaryWriter(log_dir="./logs/" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S"))

# -----------------------------generate the data splitter for training ------------------------------------
mpi_data = load_dataset("mpi", "resnet", 'S1,S2,S3,S4,S5,S6', 'S7,S8', 1)

h36m_data = load_dataset("h36m_new2", "resnet", 'S1,S5,S6,S7,S8', 'S9,S11', 2)
combined_list = zip(h36m_data, mpi_data)
combined_data = []
for data_list in combined_list:
    if None in data_list:
        combined_data += [None]
        continue
    combined_entry = [x for a_list in data_list for x in a_list]
    combined_data += [combined_entry]

test_generator, train_generator, train_generator_eval = build_generator(args.data_augmentation, args.causal,
                                                                        *combined_data)

# -----------------------------load the deep learning modelf------------------------------------

filter_widths = [int(x) for x in args.architecture.split(',')]
num_joints_in = 21
in_features = 3
num_joints_in_extra = 3
in_extra_features = 7
num_joints_out = 3
model_pos_train = TemporalModelOptimized1f(num_joints_in, in_features,
                                           num_joints_in_extra, in_extra_features,
                                           num_joints_out,
                                           filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
                                           channels=args.channels)

model_pos = TemporalModel(num_joints_in, in_features,
                          num_joints_in_extra, in_extra_features,
                          num_joints_out,
                          filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                          dense=args.dense)

model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)

if torch.cuda.is_available():
    model_pos = model_pos.cuda()
    model_pos_train = model_pos_train.cuda()

# -----------------------------Check if running the model training or model evaluation------------------------------------

if args.resume or args.evaluate:
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    model_pos_train.load_state_dict(checkpoint['model_pos'])
    model_pos.load_state_dict(checkpoint['model_pos'])
    print(checkpoint.keys())

# -----------------------------Begin training------------------------------------

lr = args.learning_rate

optimizer = optim.Adam(model_pos_train.parameters(), lr=lr, amsgrad=True)

lr_decay = args.lr_decay

losses_3d_train = []
losses_rot_train = []
losses_3d_train_eval = []
losses_rot_train_eval = []
losses_3d_valid = []
losses_rot_valid = []

epoch = 0
initial_momentum = 0.1
final_momentum = 0.001

if args.resume:
    epoch = checkpoint['epoch']
    if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')

    lr = checkpoint['lr']

print('** Note: reported losses are averaged over all frames and test-time augmentation is not used here.')
print('** The final evaluation will be carried out after the last training epoch.')

iter = 0
# Pos model only
while epoch < args.epochs:
    start_time = time()
    epoch_loss_3d_train = 0
    epoch_loss_rot_train = 0
    epoch_loss_traj_train = 0
    epoch_loss_2d_train_unlabeled = 0
    N = 0
    N_semi = 0
    model_pos_train.train()

    # Regular supervised scenario
    for _, batch_3d, batch_2d, batch_3d_input in train_generator.next_epoch():

        inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
        inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
        inputs_3d_input = torch.from_numpy(batch_3d_input.astype('float32'))
        if torch.cuda.is_available():
            inputs_3d = inputs_3d.cuda()
            inputs_2d = inputs_2d.cuda()
            inputs_3d_input = inputs_3d_input.cuda()

        optimizer.zero_grad()

        # Predict 3D poses
        predicted_3d_pos_rot = model_pos_train((inputs_2d, inputs_3d_input))
        predicted_3d_pos = predicted_3d_pos_rot[..., :3]
        predicted_3d_rot = predicted_3d_pos_rot[..., 3:]
        reference_pos = inputs_3d[..., :3]
        reference_rot = inputs_3d[..., 3:]
        loss_3d_pos = mpjpe(predicted_3d_pos, reference_pos)
        loss_3d_rot = quat_criterion(predicted_3d_rot, reference_rot)
        writer.add_scalar(f"train/loss_pos", loss_3d_pos, iter)
        writer.add_scalar(f"train/loss_rot", loss_3d_rot, iter)
        iter += 1
        epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
        epoch_loss_rot_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_rot.item()
        N += inputs_3d.shape[0] * inputs_3d.shape[1]

        loss_total = loss_3d_pos + loss_3d_rot
        loss_total.backward()

        optimizer.step()

    loss_3d_train = epoch_loss_3d_train / N
    loss_rot_train = epoch_loss_rot_train / N
    losses_3d_train.append(loss_3d_train)
    losses_rot_train.append(loss_rot_train)
    writer.add_scalar(f"train/avg_loss_pos", loss_3d_train, iter)
    writer.add_scalar(f"train/avg_loss_rot", loss_rot_train, iter)

    # End-of-epoch evaluation
    with torch.no_grad():
        model_pos.load_state_dict(model_pos_train.state_dict())
        model_pos.eval()

        epoch_loss_3d_valid = 0
        epoch_loss_rot_valid = 0
        epoch_loss_pos_valid = 0
        epoch_loss_traj_valid = 0
        epoch_loss_2d_valid = 0
        N = 0

        if not args.no_eval:
            # Evaluate on test set
            for cam, batch, batch_2d, batch_3d_input in test_generator.next_epoch():
                inputs_3d = torch.from_numpy(batch.astype('float32'))
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                inputs_3d_input = torch.from_numpy(batch_3d_input.astype('float32'))
                if torch.cuda.is_available():
                    inputs_3d = inputs_3d.cuda()
                    inputs_2d = inputs_2d.cuda()
                    inputs_3d_input = inputs_3d_input.cuda()

                inputs_traj = inputs_3d[:, :, :1].clone()

                # Predict 3D poses
                predicted_3d_pos_rot = model_pos((inputs_2d, inputs_3d_input))
                predicted_3d_pos = predicted_3d_pos_rot[..., :3]
                predicted_3d_rot = predicted_3d_pos_rot[..., 3:]
                reference_pos = inputs_3d[..., :3]
                reference_rot = inputs_3d[..., 3:]
                loss_3d_pos = mpjpe(predicted_3d_pos, reference_pos)
                loss_3d_rot = quat_criterion(predicted_3d_rot, reference_rot)
                epoch_loss_3d_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                epoch_loss_rot_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_rot.item()
                N += inputs_3d.shape[0] * inputs_3d.shape[1]

            losses_3d_valid.append(epoch_loss_3d_valid / N)
            losses_rot_valid.append(epoch_loss_rot_valid / N)
            writer.add_scalar(f"val/loss_pos_val", epoch_loss_3d_valid / N, iter)
            writer.add_scalar(f"val/loss_rot_val", epoch_loss_rot_valid / N, iter)

            # Evaluate on training set, this time in evaluation mode
            epoch_loss_3d_train_eval = 0
            epoch_loss_rot_train_eval = 0
            epoch_loss_traj_train_eval = 0
            epoch_loss_2d_train_labeled_eval = 0
            N = 0
            for cam, batch, batch_2d, batch_3d_input in train_generator_eval.next_epoch():
                if batch_2d.shape[1] == 0:
                    # This can only happen when downsampling the dataset
                    continue

                inputs_3d = torch.from_numpy(batch.astype('float32'))
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                inputs_3d_input = torch.from_numpy(batch_3d_input.astype('float32'))
                if torch.cuda.is_available():
                    inputs_3d = inputs_3d.cuda()
                    inputs_2d = inputs_2d.cuda()
                    inputs_3d_input = inputs_3d_input.cuda()
                inputs_traj = inputs_3d[:, :, :1].clone()

                # Compute 3D poses
                predicted_3d_pos_rot = model_pos((inputs_2d, inputs_3d_input))
                predicted_3d_pos = predicted_3d_pos_rot[..., :3]
                predicted_3d_rot = predicted_3d_pos_rot[..., 3:]
                reference_pos = inputs_3d[..., :3]
                reference_rot = inputs_3d[..., 3:]
                loss_3d_pos = mpjpe(predicted_3d_pos, reference_pos)
                loss_3d_rot = quat_criterion(predicted_3d_rot, reference_rot)
                epoch_loss_3d_train_eval += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                epoch_loss_rot_train_eval += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_rot.item()
                N += inputs_3d.shape[0] * inputs_3d.shape[1]

            loss_3d_train_eval = epoch_loss_3d_train_eval / N
            loss_rot_train_eval = epoch_loss_rot_train_eval / N
            writer.add_scalar(f"train/val_loss_pos_train", loss_3d_train_eval, iter)
            writer.add_scalar(f"train/val_loss_rot_train", loss_rot_train_eval, iter)

            losses_3d_train_eval.append(loss_3d_train_eval)
            losses_rot_train_eval.append(loss_rot_train_eval)

            # Evaluate 2D loss on unlabeled training set (in evaluation mode)
            epoch_loss_2d_train_unlabeled_eval = 0
            N_semi = 0

    elapsed = (time() - start_time) / 60

    if args.no_eval:
        print('[%d] time %.2f lr %f 3d_train %f' % (
            epoch + 1,
            elapsed,
            lr,
            losses_3d_train[-1] * 1000))
    else:
        print('[%d] time %.2f lr %f 3d_train %f 3d_eval %f 3d_valid %f' % (
            epoch + 1,
            elapsed,
            lr,
            losses_3d_train[-1] * 1000,
            losses_3d_train_eval[-1] * 1000,
            losses_3d_valid[-1] * 1000))

    # Decay learning rate exponentially
    lr *= lr_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    epoch += 1

    # Decay BatchNorm momentum
    momentum = initial_momentum * np.exp(-epoch / args.epochs * np.log(initial_momentum / final_momentum))
    model_pos_train.set_bn_momentum(momentum)

    # Save checkpoint if necessary
    if epoch % args.checkpoint_frequency == 0:
        chk_path = os.path.join(args.checkpoint, 'epoch_{}.bin'.format(epoch))
        print('Saving checkpoint to', chk_path)

        torch.save({
            'epoch': epoch,
            'lr': lr,
            'optimizer': optimizer.state_dict(),
            'model_pos': model_pos_train.state_dict()
        }, chk_path)

    # Save training curves after every epoch, as .png images (if requested)
    if args.export_training_curves and epoch > 3:
        if 'matplotlib' not in sys.modules:
            import matplotlib

            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

        plt.figure()
        epoch_x = np.arange(3, len(losses_3d_train)) + 1
        plt.plot(epoch_x, losses_3d_train[3:], '--', color='C0')
        plt.plot(epoch_x, losses_3d_train_eval[3:], color='C0')
        plt.plot(epoch_x, losses_3d_valid[3:], color='C1')
        plt.legend(['3d train', '3d train (eval)', '3d valid (eval)'])
        plt.ylabel('MPJPE (m)')
        plt.xlabel('Epoch')
        plt.xlim((3, epoch))
        plt.savefig(os.path.join(args.checkpoint, 'loss_3d.png'))

        plt.close('all')

joints_left = h36m_skeleton.joints_left()
joints_right = h36m_skeleton.joints_right()


# -----------------------------Begin evaluation------------------------------------
# Evaluate
def evaluate(test_generator, action=None, return_predictions=False, use_trajectory_model=False):
    epoch_loss_3d_rot = 0
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    with torch.no_grad():
        model_pos.eval()
        N = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            # Positional model
            predicted_3d_pos_rot = model_pos(inputs_2d)
            # Test-time augmentation (if enabled)
            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos_rot[1, :, :, [0, 3, 4]] *= -1
                if not use_trajectory_model:
                    predicted_3d_pos_rot[1, :, joints_left + joints_right] = predicted_3d_pos[1, :,
                                                                             joints_right + joints_left]
                predicted_3d_pos_rot = torch.mean(predicted_3d_pos_rot, dim=0, keepdim=True)

            if return_predictions:
                return predicted_3d_pos_rot.squeeze(0).cpu().numpy()

            inputs_3d = torch.from_numpy(batch.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
            inputs_3d[:, :, 0, 0:3] = 0
            if test_generator.augment_enabled():
                inputs_3d = inputs_3d[:1]

            predicted_3d_pos_rot = model_pos_train(inputs_2d)
            predicted_3d_pos = predicted_3d_pos_rot[..., :3]
            predicted_3d_rot = predicted_3d_pos_rot[..., 3:]
            reference_pos = inputs_3d[..., :3]
            reference_rot = inputs_3d[..., 3:]
            error_3d_rot = quat_criterion(predicted_3d_rot, reference_rot)

            epoch_loss_3d_rot += inputs_3d.shape[0] * inputs_3d.shape[1] * error_3d_rot.item()

            error = mpjpe(predicted_3d_pos, reference_pos)
            epoch_loss_3d_pos_scale += inputs_3d.shape[0] * inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos,
                                                                                         reference_pos).item()

            epoch_loss_3d_pos += inputs_3d.shape[0] * inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0] * inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

            # Compute velocity error
            epoch_loss_3d_vel += inputs_3d.shape[0] * inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

    if action is None:
        print('----------')
    else:
        print('----' + action + '----')
    e0 = epoch_loss_3d_rot / N
    e1 = (epoch_loss_3d_pos / N) * 1000
    e2 = (epoch_loss_3d_pos_procrustes / N) * 1000
    e3 = (epoch_loss_3d_pos_scale / N) * 1000
    ev = (epoch_loss_3d_vel / N) * 1000
    print('Test time augmentation:', test_generator.augment_enabled())
    print('Rotation Error (rot):', e0, 'radian')
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
    print('Velocity Error (MPJVE):', ev, 'mm')
    print('----------')

    return e1, e2, e3, ev


writer.close()
