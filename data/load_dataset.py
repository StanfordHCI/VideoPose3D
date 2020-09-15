from common.camera import world_to_camera
import numpy as np

from common.generators import UnchunkedGenerator, ChunkedGenerator
from common.h36m_dataset import h36m_skeleton
from common.utils import deterministic_random


def load_dataset(args_dataset, args_keypoints, args_subjects_train, args_subjects_test,
                 args_downsample):
    # -----------------------------prepare the 3D dataset as ground truth ------------------------------------
    print('Loading dataset...')
    dataset_path = 'data/data_3d_' + args_dataset + '.npz'
    mpi_dataset = False
    if args_dataset == 'h36m_new2':
        from common.h36m_dataset import Human36mDataset
        dataset = Human36mDataset(dataset_path)
    elif args_dataset.startswith('humaneva'):
        from common.humaneva_dataset import HumanEvaDataset

        dataset = HumanEvaDataset(dataset_path)
    elif args_dataset.startswith('custom'):
        from common.custom_dataset import CustomDataset

        dataset = CustomDataset('data/data_2d_' + args_dataset + '_' + args_keypoints + '.npz')
    elif args_dataset.startswith('mpi'):
        from common.mpi_dataset import MpiDataset
        dataset = MpiDataset(dataset_path)

        # participants = [f'S{i}' for i in range(1, 9)]
        # for dataset_path_temp in ['data/data_3d_' + args_dataset + p + '.npz' for p in participants]
        #     dataset_temp = MpiDataset(dataset_path_temp)
        #     # noinspection PyProtectedMember
        #     dataset._data.update(dataset_temp._data)
        mpi_dataset = True
    else:
        raise KeyError('Invalid dataset')

    print('Preparing data...')

    # Model v1.5: Do not apply camera transformation
    for subject in dataset.subjects():
        lengths = dataset[subject]['lengths']
        if mpi_dataset:
            del dataset[subject]['lengths']['pos_rot']
        for action in dataset[subject].keys():
            if action == "lengths":
                continue
            anim = dataset[subject][action]

            if 'positions' in anim:
                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    pos_3d[:, 1:] -= pos_3d[:, :1]  # Remove global offset, but keep trajectory in first position
                    positions_3d.append(pos_3d)
                anim['pos_rot'] = positions_3d

            if 'pos_rot' in anim:
                positions_3d = []
                for camera_i, cam in enumerate(anim['cameras']):
                    if mpi_dataset:
                        pos_rot = anim['pos_rot'][camera_i]
                    else:
                        pos_rot = anim['pos_rot'].copy()
                    pos_rot[:, :, :2] -= pos_rot[:, :1, :2]  # Remove global offset
                    positions_3d.append(pos_rot)
                anim['pos_rot'] = positions_3d
            # anim['lengths'] = lengths

    # -----------------------------prepare the 2D dataset as input ------------------------------------
    print('Loading 2D detections...')
    if mpi_dataset:
        participants = [f'S{i}' for i in range(1, 9)]
        keypoints = {}
        for p_id in participants:
            path = 'data/data_2d_' + args_dataset + '_' + args_keypoints + p_id + '.npz'
            keypoints_temp = np.load(path, allow_pickle=True)
            keypoints_temp = keypoints_temp['positions_2d'].item()
            keypoints.update(keypoints_temp)
    else:
        keypoints = np.load('data/data_2d_' + args_dataset + '_' + args_keypoints + '.npz', allow_pickle=True)
        keypoints = keypoints['positions_2d'].item()

    for subject in dataset.subjects():
        assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
        for action in dataset[subject].keys():
            if action == "lengths":
                continue
            assert action in keypoints[
                subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
            if 'pos_rot' not in dataset[subject][action]:
                continue

            for cam_idx in range(len(keypoints[subject][action])):

                # We check for >= instead of == because some videos in H3.6M contain extra frames
                mocap_length = dataset[subject][action]['pos_rot'][cam_idx].shape[0]
                kps_length = keypoints[subject][action][cam_idx].shape[0]
                # assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                if kps_length > mocap_length:
                    # Shorten sequence
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]
                elif kps_length < mocap_length:
                    dataset[subject][action]['pos_rot'][cam_idx] = \
                        dataset[subject][action]['pos_rot'][cam_idx][:kps_length]

            assert len(keypoints[subject][action]) == len(dataset[subject][action]['pos_rot'])

    # -----------------------------normalize the 2D input ------------------------------------
    # for subject in keypoints.keys():
    #     for action in keypoints[subject]:
    #         for cam_idx, kps in enumerate(keypoints[subject][action]):
    #             # Normalize camera frame
    #             # cam = dataset.cameras()[subject][cam_idx]
    #             # kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
    #             keypoints[subject][action][cam_idx] = kps

    subjects_train = args_subjects_train.split(',')
    subjects_test = args_subjects_test.split(',')

    def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
        out_poses_3d = []
        out_poses_2d = []
        out_camera_params = []
        for subject in subjects:
            for action in keypoints[subject].keys():
                if action == 'lengths':
                    continue
                if action_filter is not None:
                    found = False
                    for a in action_filter:
                        if action.startswith(a):
                            found = True
                            break
                    if not found:
                        continue

                poses_2d = keypoints[subject][action]
                for i in range(len(poses_2d)):  # Iterate across cameras
                    out_poses_2d.append(poses_2d[i])

                if subject in dataset.cameras():
                    cams = dataset.cameras()[subject]
                    assert len(cams) == len(poses_2d), 'Camera count mismatch'
                    for cam in cams:
                        if 'intrinsic' in cam:
                            out_camera_params.append(cam['intrinsic'])

                if parse_3d_poses and 'pos_rot' in dataset[subject][action]:
                    # cam_count = len(poses_2d)
                    # if mpi_dataset:
                    #     dataset[subject][action]['pos_rot'] = dataset[subject][action]['pos_rot'][:cam_count]

                    poses_3d = dataset[subject][action]['pos_rot']
                    assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                    for i in range(len(poses_3d)):  # Iterate across cameras
                        out_poses_3d.append(poses_3d[i])
        # print(out_poses_3d[0][0:2,:,:])
        if len(out_camera_params) == 0:
            out_camera_params = None
        if len(out_poses_3d) == 0:
            out_poses_3d = None

        stride = args_downsample
        if subset < 1:
            for i in range(len(out_poses_2d)):
                n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
                start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
                out_poses_2d[i] = out_poses_2d[i][start:start + n_frames:stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][start:start + n_frames:stride]
        elif stride > 1:
            # Downsample as requested
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][::stride]

        return out_camera_params, out_poses_3d, out_poses_2d

    cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, None)
    cameras_train, poses_train, poses_train_2d = fetch(subjects_train, None)

    return cameras_valid, poses_valid, poses_valid_2d, cameras_train, poses_train, poses_train_2d


def build_generator(args_data_augmentation, args_causal,
                    cameras_valid, poses_valid, poses_valid_2d,
                    cameras_train, poses_train, poses_train_2d):
    kps_left, kps_right = [1, 3, 5, 7, 9, 11, 13, 15, 17, 18], [2, 4, 6, 8, 10, 12, 14, 16, 19, 20]
    joints_left, joints_right = list(h36m_skeleton.joints_left()), list(h36m_skeleton.joints_right())
    receptive_field = 243
    print('INFO: Receptive field: {} frames'.format(receptive_field))
    pad = (receptive_field - 1) // 2  # Padding on each side
    if args_causal:
        print('INFO: Using causal convolutions')
        causal_shift = pad
    else:
        causal_shift = 0

    # TODO: make sure symmetry is correct
    test_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
                                        pad=pad, causal_shift=causal_shift, augment=False,
                                        kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                        joints_right=joints_right, skeleton=h36m_skeleton)
    print('INFO: Testing on {} frames'.format(test_generator.num_frames()))

    # ---------------------------- data generator ----------------------------

    train_generator = ChunkedGenerator(1024, cameras_train, poses_train, poses_train_2d, 1,
                                       pad=pad, causal_shift=causal_shift, shuffle=True, augment=args_data_augmentation,
                                       kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                       joints_right=joints_right, skeleton=h36m_skeleton)
    train_generator_eval = UnchunkedGenerator(cameras_train, poses_train, poses_train_2d,
                                              pad=pad, causal_shift=causal_shift, augment=False,
                                              skeleton=h36m_skeleton)
    print('INFO: Training on {} frames'.format(train_generator_eval.num_frames()))

    return test_generator, train_generator, train_generator_eval
