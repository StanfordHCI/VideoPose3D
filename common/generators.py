# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from itertools import zip_longest
import numpy as np
from torch.utils.data import DataLoader, Dataset

from common.camera import random_x_y_shift, random_z_rot, apply_transform_combined
from common.skeleton import Skeleton


def extract_3d(poses_3d, input_joints, output_joints):
    poses_3d_input = []
    poses_3d_output = []
    for i in range(len(poses_3d)):
        poses_3d_input.append(poses_3d[i][:, input_joints, :])
        poses_3d_output.append(poses_3d[i][:, output_joints, :])
    return poses_3d_output, poses_3d_input


def convert_to_vr(orig_data, t, q):
    orig_data_size = list(orig_data.shape[:-1]) + [1]
    current_t = np.tile(t, orig_data_size)
    current_q = np.tile(q, orig_data_size)
    return apply_transform_combined(orig_data.astype('float32'), current_q, current_t)


def element_index_batch(item, in_list):
    return [in_list.index(i) for i in item]


class ChunkedGeneratorDataset(Dataset):
    """
    Batched data generator, used for training.
    The sequences are split into equal-length chunks and padded as necessary.
    
    Arguments:
    batch_size -- the batch size to use for training
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    chunk_length -- number of output frames to predict for each training example (usually 1)
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    shuffle -- randomly shuffle the dataset before each epoch
    random_seed -- initial seed to use for the random generator
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """

    def __init__(self, batch_size, cameras, poses_3d, poses_2d,
                 chunk_length, pad=0, causal_shift=0,
                 shuffle=True, random_seed=1234,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False, skeleton: Skeleton = None):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cameras is None or len(cameras) == len(poses_2d)
        # Build lineage info
        pairs = []  # (seq_idx, start_frame, end_frame, flip) tuples
        if poses_3d is not None:
            poses_3d, poses_3d_input = extract_3d(poses_3d, skeleton.input_joints(), skeleton.output_joints())
        for i in range(len(poses_2d)):
            assert poses_3d is None or poses_3d[i].shape[0] == poses_3d[i].shape[0]
            n_chunks = (poses_2d[i].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - poses_2d[i].shape[0]) // 2
            bounds = np.arange(n_chunks + 1) * chunk_length - offset
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], augment_vector)
            if augment:
                pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], ~augment_vector)

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d
        self.poses_3d_input = [] if poses_3d is None else poses_3d_input

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.input_joints = skeleton.input_joints()
        self.output_joints = skeleton.output_joints()
        if joints_left is not None and joints_right is not None:
            self.joints_left = \
                element_index_batch(set(joints_left).intersection(self.output_joints), self.output_joints)
            self.joints_right = \
                element_index_batch(set(joints_right).intersection(set(self.output_joints)), self.output_joints)
            self.joints_left_input = \
                element_index_batch(set(joints_left).intersection(set(self.input_joints)), self.input_joints)
            self.joints_right_input = \
                element_index_batch(set(joints_right).intersection(set(self.input_joints)), self.input_joints)
        else:
            self.joints_left = None
            self.joints_right = None
            self.joints_left_input = None
            self.joints_right_input = None

    def num_frames(self):
        return self.num_batches * self.batch_size

    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def augment_enabled(self):
        return self.augment

    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, item) -> (np.array, np.array, np.array, np.array):
        """
        Get next training data
        Returns:
            cam_array:
            batch_3d: as output
            batch_2d: as input
            batch_3d_extra: as input
       """
        seq_i, start_3d, end_3d, flip = self.pairs[item]
        transform_t = np.zeros((1, 3))  # generate a random transform
        transform_q = random_z_rot([])
        start_2d = start_3d - self.pad - self.causal_shift
        end_2d = end_3d + self.pad - self.causal_shift

        # 2D poses
        seq_2d = self.poses_2d[seq_i]
        seq_3d_input = self.poses_3d_input[seq_i]
        low_2d = max(start_2d, 0)
        high_2d = min(end_2d, seq_2d.shape[0])
        pad_left_2d = low_2d - start_2d
        pad_right_2d = end_2d - high_2d
        if pad_left_2d != 0 or pad_right_2d != 0:
            batch_2d = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)),
                              'edge')
            batch_3d_input = np.pad(seq_3d_input[low_2d:high_2d],
                                    ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)),
                                    'edge')
        else:
            batch_2d = seq_2d[low_2d:high_2d]
            batch_3d_input = seq_3d_input[low_2d:high_2d]

        batch_3d_input = convert_to_vr(batch_3d_input, transform_t, transform_q)

        if flip:
            # Flip 2D keypoints
            batch_2d[:, :, 0] *= -1
            batch_2d[:, self.kps_left + self.kps_right] = batch_2d[:, self.kps_right + self.kps_left]
            batch_3d_input[:, :, [0, 3, 4]] *= -1
            batch_3d_input[:, self.joints_left_input + self.joints_right_input] = \
                batch_3d_input[:, self.joints_right_input + self.joints_left_input]

        # 3D poses
        if self.poses_3d is not None:
            seq_3d = self.poses_3d[seq_i]
            low_3d = max(start_3d, 0)
            high_3d = min(end_3d, seq_3d.shape[0])
            pad_left_3d = low_3d - start_3d
            pad_right_3d = end_3d - high_3d
            if pad_left_3d != 0 or pad_right_3d != 0:
                batch_3d = np.pad(seq_3d[low_3d:high_3d],
                                  ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
            else:
                batch_3d = seq_3d[low_3d:high_3d]

            batch_3d = convert_to_vr(batch_3d, transform_t, transform_q)

            if flip:
                # Flip 3D joints
                batch_3d[:, :, [0, 3, 4]] *= -1
                batch_3d[:, self.joints_left + self.joints_right] = \
                    batch_3d[:, self.joints_right + self.joints_left]

        # Cameras
        if self.cameras is not None:
            batch_cam = self.cameras[seq_i]
            if flip:
                # Flip horizontal distortion coefficients
                batch_cam[2] *= -1
                batch_cam[7] *= -1

        if self.poses_3d is None and self.cameras is None:
            return None, None, batch_2d, None
        elif self.poses_3d is not None and self.cameras is None:
            return None, batch_3d, batch_2d, batch_3d_input
        elif self.poses_3d is None:
            return batch_cam, None, batch_2d, None
        else:
            return batch_cam, batch_3d, batch_2d, batch_3d_input


class ChunkedGenerator(DataLoader):
    dataset: ChunkedGeneratorDataset

    def __init__(self, batch_size, cameras, poses_3d, poses_2d,
                 chunk_length, pad=0, causal_shift=0,
                 shuffle=True, random_seed=1234,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False, skeleton: Skeleton = None):
        self.dataset = ChunkedGeneratorDataset(batch_size, cameras, poses_3d, poses_2d,
                                               chunk_length, pad, causal_shift,
                                               shuffle, random_seed,
                                               augment, kps_left, kps_right, joints_left, joints_right,
                                               endless, skeleton)
        super(ChunkedGenerator, self).__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8,
                                               collate_fn=self.combine)

    @staticmethod
    def combine(inputs: [(np.array, np.array, np.array, np.array)]) -> (np.array, np.array, np.array, np.array):
        if len(inputs) == 0:
            return None, None, None, None
        else:
            example = inputs[0]
            result = []
            for i in range(4):
                if example[i] is None:
                    result += [None]
                result += [np.stack([input_array[i] for input_array in inputs], axis=0)]
        return result

    def next_epoch(self):
        for item in self.__iter__():
            yield item


class UnchunkedGenerator:
    """
    Non-batched data generator, used for testing.
    Sequences are returned one at a time (i.e. batch size = 1), without chunking.
    
    If data augmentation is enabled, the batches contain two sequences (i.e. batch size = 2),
    the second of which is a mirrored version of the first.
    
    Arguments:
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """

    def __init__(self, cameras, poses_3d, poses_2d, pad=0, causal_shift=0,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 skeleton: Skeleton = None):
        # TODO: parse skeleton
        assert poses_3d is None or len(poses_3d) == len(poses_2d)
        assert cameras is None or len(cameras) == len(poses_2d)
        if poses_3d is not None:
            poses_3d, poses_3d_input = extract_3d(poses_3d, skeleton.input_joints(), skeleton.output_joints())
        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.input_joints = skeleton.input_joints()
        self.output_joints = skeleton.output_joints()
        if joints_left is not None and joints_right is not None:
            self.joints_left = \
                element_index_batch(set(joints_left).intersection(self.output_joints), self.output_joints)
            self.joints_right = \
                element_index_batch(set(joints_right).intersection(set(self.output_joints)), self.output_joints)
            self.joints_left_input = \
                element_index_batch(set(joints_left).intersection(set(self.input_joints)), self.input_joints)
            self.joints_right_input = \
                element_index_batch(set(joints_right).intersection(set(self.input_joints)), self.input_joints)
        else:
            self.joints_left = None
            self.joints_right = None
            self.joints_left_input = None
            self.joints_right_input = None

        self.pad = pad
        self.causal_shift = causal_shift
        self.cameras = [] if cameras is None else cameras
        self.poses_3d = [] if poses_3d is None else poses_3d
        self.poses_2d = poses_2d
        self.poses_3d_input = [] if poses_3d is None else poses_3d_input

    def num_frames(self):
        count = 0
        for p in self.poses_2d:
            count += p.shape[0]
        return count

    def augment_enabled(self):
        return self.augment

    def set_augment(self, augment):
        self.augment = augment

    def next_epoch(self) -> (np.array, np.array, np.array, np.array):
        """
        Get next training data
        Returns:
            cam_array:
            batch_3d: as output
            batch_2d: as input
            batch_3d_extra: as input
       """
        for seq_cam, seq_3d, seq_2d, seq_3d_input in zip_longest(self.cameras, self.poses_3d, self.poses_2d,
                                                                 self.poses_3d_input):
            # TODO: apply random transformation
            batch_cam = None if seq_cam is None else np.expand_dims(seq_cam, axis=0)
            batch_3d = None if seq_3d is None else np.expand_dims(seq_3d, axis=0)
            batch_2d = np.expand_dims(np.pad(seq_2d,
                                             ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0),
                                              (0, 0)),
                                             'edge'), axis=0)
            batch_3d_input = None if seq_3d_input is None else np.expand_dims(np.pad(seq_3d_input,
                                                                                     ((self.pad + self.causal_shift,
                                                                                       self.pad - self.causal_shift),
                                                                                      (0, 0),
                                                                                      (0, 0)),
                                                                                     'edge'), axis=0)
            # generate random transform
            transform_t = np.zeros((1, 3))
            transform_q = random_z_rot([])

            # apply random transform
            batch_3d = convert_to_vr(batch_3d, transform_t, transform_q)
            batch_3d_input = convert_to_vr(batch_3d_input, transform_t, transform_q)

            if self.augment:
                # Append flipped version
                if batch_cam is not None:
                    batch_cam = np.concatenate((batch_cam, batch_cam), axis=0)
                    batch_cam[1, 2] *= -1
                    batch_cam[1, 7] *= -1

                if batch_3d is not None:
                    batch_3d = np.concatenate((batch_3d, batch_3d), axis=0)
                    batch_3d[1, :, :, [0, 3, 4]] *= -1
                    batch_3d[1, :, self.joints_left + self.joints_right] = \
                        batch_3d[1, :, self.joints_right + self.joints_left]

                    batch_3d_input = np.concatenate((batch_3d_input, batch_3d_input), axis=0)
                    batch_3d_input[1, :, :, [0, 3, 4]] *= -1
                    batch_3d_input[1, :, self.joints_left_input + self.joints_right_input] = \
                        batch_3d_input[1, :, self.joints_right_input + self.joints_left_input]

                batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
                batch_2d[1, :, :, 0] *= -1
                batch_2d[1, :, self.kps_left + self.kps_right] = batch_2d[1, :, self.kps_right + self.kps_left]

            yield batch_cam, batch_3d, batch_2d, batch_3d_input
