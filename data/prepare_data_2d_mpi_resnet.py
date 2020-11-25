# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import ast
import os
import sys
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.append('../')
from common.mpi_dataset import MpiDataset

sys.path.append('../../resnetpose/')
from SimpleHRNet import SimpleHRNet
from misc.utils import find_person_id_associations

output_filename = 'data_3d_mpi'
output_filename_2d = 'data_2d_mpi_resnet'
subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']


def visual(frame, name, keypoints):
    img = np.zeros((960, 640, 3))
    for i in range(0, 21):
        x, y = int(keypoints[i][0]), int(keypoints[i][1])
        cv2.circle(frame, (x, y), 1, (255, 5, 0), 5)
        cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    cv2.imwrite(name, frame)


def getKptsFromImage(model, frame, disable_tracking=True, hrnet_joints_set="coco"):
    if not disable_tracking:
        prev_boxes = None
        prev_pts = None
        prev_person_ids = None
        next_person_id = 0

    t0 = time.time()
    pts = model.predict(frame)
    t1 = time.time()
    # print(f"detection time: {t1 - t0}")

    if not disable_tracking:
        boxes, pts = pts

    if not disable_tracking:
        if len(pts) > 0:
            if prev_pts is None and prev_person_ids is None:
                person_ids = np.arange(next_person_id, len(pts) + next_person_id, dtype=np.int32)
                next_person_id = len(pts) + 1
            else:
                boxes, pts, person_ids = find_person_id_associations(
                    boxes=boxes, pts=pts, prev_boxes=prev_boxes, prev_pts=prev_pts, prev_person_ids=prev_person_ids,
                    next_person_id=next_person_id, pose_alpha=0.2, similarity_threshold=0.4, smoothing_alpha=0.1,
                )
                next_person_id = max(next_person_id, np.max(person_ids) + 1)
        else:
            person_ids = np.array((), dtype=np.int32)

        prev_boxes = boxes.copy()
        prev_pts = pts.copy()
        prev_person_ids = person_ids

    else:
        person_ids = np.arange(len(pts), dtype=np.int32)
    # print(f"tracking time: {time.time() - t1}")
    '''
    for i, (pt, pid) in enumerate(zip(pts, person_ids)):
        frame = draw_points_and_skeleton(frame, pt, joints_dict()[hrnet_joints_set]['skeleton'], person_index=pid,
                                        points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                        points_palette_samples=10)
    '''
    # fps = 1. / (time.time() - t)
    # print('\rframerate: %f fps' % fps, end='')
    return pts


def Get_ResNet_Model(hrnet_m='PoseResNet', hrnet_c=50, hrnet_j=17,
                     hrnet_weights="../../resnetpose/weights/pose_resnet_50_384x288.pth",
                     hrnet_joints_set="coco", image_resolution='(384, 288)',
                     single_person=False, use_tiny_yolo=False, disable_tracking=False, max_batch_size=100):
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # print(device)
    # print(type(image_resolution))
    image_resolution = ast.literal_eval(image_resolution)

    if use_tiny_yolo:
        yolo_model_def = "../../resnetpose/models/detectors/yolo/config/yolov3-tiny.cfg"
        yolo_class_path = "../../resnetpose/models/detectors/yolo/data/coco.names"
        yolo_weights_path = "../../resnetpose/models/detectors/yolo/weights/yolov3-tiny.weights"
    else:
        yolo_model_def = "../../resnetpose/models/detectors/yolo/config/yolov3.cfg"
        yolo_class_path = "../../resnetpose/models/detectors/yolo/data/coco.names"
        yolo_weights_path = "../../resnetpose/models/detectors/yolo/weights/yolov3.weights"

    model = SimpleHRNet(
        hrnet_c,
        hrnet_j,
        hrnet_weights,
        model_name=hrnet_m,
        resolution=image_resolution,
        multiperson=not single_person,
        return_bounding_boxes=not disable_tracking,
        max_batch_size=max_batch_size,
        yolo_model_def=yolo_model_def,
        yolo_class_path=yolo_class_path,
        yolo_weights_path=yolo_weights_path,
        device=device
    )

    return model


if __name__ == '__main__':
    if os.path.basename(os.getcwd()) != 'data':
        print('This script must be launched from the "data" directory')
        exit(0)

    parser = argparse.ArgumentParser(description='Human3.6M dataset downloader/converter')

    # Convert dataset preprocessed by Martinez et al. in https://github.com/una-dinosauria/3d-pose-baseline
    parser.add_argument('--from-archive', default='', type=str, metavar='PATH', help='convert preprocessed dataset')

    # Convert dataset from original source, using files converted to .mat (the Human3.6M dataset path must be specified manually)
    # This option requires MATLAB to convert files using the provided script
    parser.add_argument('--from-source', default='', type=str, metavar='PATH', help='convert original dataset')

    # Convert dataset from original source, using original .cdf files (the Human3.6M dataset path must be specified manually)
    # This option does not require MATLAB, but the Python library cdflib must be installed
    parser.add_argument('--from-source-cdf', default='', type=str, metavar='PATH', help='convert original dataset')

    parser.add_argument('--model', type=int, default=101)
    parser.add_argument('--cam_id', type=int, default=0)
    parser.add_argument('--cam_width', type=int, default=1280)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--cam_height', type=int, default=720)
    parser.add_argument('--scale_factor', type=float, default=0.7125)
    parser.add_argument('--subject', type=int, default=0)

    args = parser.parse_args()

    model = Get_ResNet_Model(hrnet_m='PoseResNet', hrnet_c=152, hrnet_j=21,
                             hrnet_weights="../../resnetpose/weights/pose_resnet_152_384x288_new.pth",
                             hrnet_joints_set="coco", image_resolution='(384, 288)',
                             single_person=False, use_tiny_yolo=False, disable_tracking=True, max_batch_size=500)

    # output_stride = model.output_stride

    # Create 2D pose file
    print('')
    print('Computing ground-truth 2D poses...')
    dataset = MpiDataset(output_filename + '.npz')
    output_2d_poses = {}
    batch_size = args.batch_size

    for subject in dataset.subjects():
        output_2d_poses[subject] = {}
        for action in dataset[subject].keys():
            positions_2d_posenet = []
            camera_index = [f'video_{i}' for i in range(8)]
            # action = "Sitting 1"
            for cam_index in camera_index:
                f = args.from_source + '/' + subject + '/' + action + '/imageSequence/' + cam_index + '.avi'
                cap = cv2.VideoCapture(f)
                frames_num = int(cap.get(7))
                # print(frames_num)
                print(f, frames_num)
                temp_pos = []
                start_time = time.time()
                for i in tqdm(range(0, (frames_num - 1) // batch_size + 1)):
                    image_stack = []
                    for j in range(0, batch_size):
                        if i * batch_size + j >= frames_num:
                            break
                        _, frame = cap.read()
                        image_stack.append(frame[np.newaxis, :])
                    inputs = np.concatenate(image_stack, axis=0)
                    # print(inputs.shape, len(inputs))
                    # raise KeyboardInterrupt
                    start_time = time.time()
                    joint2D = getKptsFromImage(model, inputs)
                    # print(len(joint2D))

                    for j in range(0, len(joint2D)):
                        if len(joint2D[j]) <= 0: print(j, joint2D[j])
                        joint_2d = joint2D[j][0]
                        joint_2d = joint_2d[:, [1, 0, 2]]
                        temp_pos.append(joint_2d)
                        visual(image_stack[j][0], "my"+str(i * batch_size + j)+'.jpg', temp_pos[-1])
                    # raise KeyboardInterrupt
                # print(len(temp_pos))
                positions_2d_posenet.append(np.array(temp_pos))
            # raise KeyboardInterrupt

            # anim = dataset[subject][action]
            # positions_2d = []
            # for cam in anim['cameras']:
            #     pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
            #     pos_2d = wrap(project_to_2d, pos_3d, cam['intrinsic'], unsqueeze=True)
            #     pos_2d_pixel_space = image_coordinates(pos_2d, w=cam['res_w'], h=cam['res_h'])
            #     positions_2d.append(pos_2d_pixel_space.astype('float32'))
            # for i in range(0,4):
            #    visual("my"+str(i)+'.jpg', positions_2d_posenet[i][0])
            #    visual("you"+str(i)+'.jpg', positions_2d[i][0])
            # raise KeyboardInterrupt
            output_2d_poses[subject][action] = positions_2d_posenet

            print("saving user" + subject)
            np.savez_compressed(output_filename_2d + subject, positions_2d=output_2d_poses[subject])

    print('Saving...')
    metadata = {
        'num_joints': dataset.skeleton().num_joints(),
        'keypoints_symmetry': [dataset.skeleton().joints_left(), dataset.skeleton().joints_right()]
    }
    np.savez_compressed(output_filename_2d + f"{subject}", positions_2d=output_2d_poses, metadata=metadata)

    print('Done.')
