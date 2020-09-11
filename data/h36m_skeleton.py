from collections import OrderedDict
from enum import IntEnum
from typing import Optional, Union

import torch
from kornia import rotation_matrix_to_quaternion

from data.skeleton_scale import SkeletonScale


class H36mJoint(IntEnum):
    ROOT = -1
    HEAD = 0
    HEAD_TOP = 1
    FACE = 2
    SHOULDER_CENTER = 3
    HIP_CENTER = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_THUMB = 11
    RIGHT_THUMB = 12
    LEFT_INDEX = 13
    RIGHT_INDEX = 14
    LEFT_HIP = 15
    RIGHT_HIP = 16
    LEFT_KNEE = 17
    RIGHT_KNEE = 18
    LEFT_ANKLE = 19
    RIGHT_ANKLE = 20
    LEFT_TOE = 21
    RIGHT_TOE = 22


H36mJoint.count = 23

H36mJoint.left_joints = [
    H36mJoint.LEFT_SHOULDER,
    H36mJoint.LEFT_ELBOW,
    H36mJoint.LEFT_WRIST,
    H36mJoint.LEFT_THUMB,
    H36mJoint.LEFT_INDEX,
    H36mJoint.LEFT_HIP,
    H36mJoint.LEFT_KNEE,
    H36mJoint.LEFT_ANKLE,
    H36mJoint.LEFT_TOE
]

H36mJoint.right_joints = [
    H36mJoint.RIGHT_SHOULDER,
    H36mJoint.RIGHT_ELBOW,
    H36mJoint.RIGHT_WRIST,
    H36mJoint.RIGHT_THUMB,
    H36mJoint.RIGHT_INDEX,
    H36mJoint.RIGHT_HIP,
    H36mJoint.RIGHT_KNEE,
    H36mJoint.RIGHT_ANKLE,
    H36mJoint.LEFT_TOE
]

H36mJoint.parent_joint_map = OrderedDict({
    H36mJoint.HEAD: H36mJoint.ROOT,
    H36mJoint.HEAD_TOP: H36mJoint.HEAD,
    H36mJoint.FACE: H36mJoint.HEAD,
    H36mJoint.SHOULDER_CENTER: H36mJoint.HEAD,
    H36mJoint.HIP_CENTER: H36mJoint.HEAD,
    H36mJoint.LEFT_SHOULDER: H36mJoint.SHOULDER_CENTER,
    H36mJoint.RIGHT_SHOULDER: H36mJoint.SHOULDER_CENTER,
    H36mJoint.LEFT_ELBOW: H36mJoint.LEFT_SHOULDER,
    H36mJoint.RIGHT_ELBOW: H36mJoint.RIGHT_SHOULDER,
    H36mJoint.LEFT_WRIST: H36mJoint.LEFT_ELBOW,
    H36mJoint.RIGHT_WRIST: H36mJoint.RIGHT_ELBOW,
    H36mJoint.LEFT_THUMB: H36mJoint.LEFT_WRIST,
    H36mJoint.RIGHT_THUMB: H36mJoint.RIGHT_WRIST,
    H36mJoint.LEFT_INDEX: H36mJoint.LEFT_WRIST,
    H36mJoint.RIGHT_INDEX: H36mJoint.RIGHT_WRIST,
    H36mJoint.LEFT_HIP: H36mJoint.HIP_CENTER,
    H36mJoint.RIGHT_HIP: H36mJoint.HIP_CENTER,
    H36mJoint.LEFT_KNEE: H36mJoint.LEFT_HIP,
    H36mJoint.RIGHT_KNEE: H36mJoint.RIGHT_HIP,
    H36mJoint.LEFT_ANKLE: H36mJoint.LEFT_KNEE,
    H36mJoint.RIGHT_ANKLE: H36mJoint.RIGHT_KNEE,
    H36mJoint.LEFT_TOE: H36mJoint.LEFT_ANKLE,
    H36mJoint.RIGHT_TOE: H36mJoint.RIGHT_ANKLE
})

H36mJoint.scale_map = {
    H36mJoint.HEAD: SkeletonScale.NO_SCALE,
    H36mJoint.HEAD_TOP: SkeletonScale.NO_SCALE,
    H36mJoint.FACE: SkeletonScale.NO_SCALE,
    H36mJoint.SHOULDER_CENTER: SkeletonScale.HEAD_NECK,
    H36mJoint.HIP_CENTER: SkeletonScale.SHOULDER_HIP,
    H36mJoint.LEFT_SHOULDER: SkeletonScale.SHOULDER_LENGTH,
    H36mJoint.RIGHT_SHOULDER: SkeletonScale.SHOULDER_LENGTH,
    H36mJoint.LEFT_ELBOW: SkeletonScale.LEFT_SHOULDER_ELBOW,
    H36mJoint.RIGHT_ELBOW: SkeletonScale.RIGHT_SHOULDER_ELBOW,
    H36mJoint.LEFT_WRIST: SkeletonScale.LEFT_ELBOW_WRIST,
    H36mJoint.RIGHT_WRIST: SkeletonScale.RIGHT_ELBOW_WRIST,
    H36mJoint.LEFT_THUMB: SkeletonScale.NO_SCALE,
    H36mJoint.RIGHT_THUMB: SkeletonScale.NO_SCALE,
    H36mJoint.LEFT_INDEX: SkeletonScale.NO_SCALE,
    H36mJoint.RIGHT_INDEX: SkeletonScale.NO_SCALE,
    H36mJoint.LEFT_HIP: SkeletonScale.HIP_LENGTH,
    H36mJoint.RIGHT_HIP: SkeletonScale.HIP_LENGTH,
    H36mJoint.LEFT_KNEE: SkeletonScale.LEFT_HIP_KNEE,
    H36mJoint.RIGHT_KNEE: SkeletonScale.RIGHT_HIP_KNEE,
    H36mJoint.LEFT_ANKLE: SkeletonScale.LEFT_KNEE_ANKLE,
    H36mJoint.RIGHT_ANKLE: SkeletonScale.RIGHT_KNEE_ANKLE,
    H36mJoint.LEFT_TOE: SkeletonScale.NO_SCALE,
    H36mJoint.RIGHT_TOE: SkeletonScale.NO_SCALE
}


class H36mSkeleton:
    relative_pos: torch.Tensor
    lowest_pos: torch.Tensor
    temp_relative_pos: torch.Tensor
    absolute_pos: torch.Tensor
    temp_absolute_pos: torch.Tensor
    vr_joints: torch.Tensor

    def __init__(self, absolute_pos):
        self.absolute_pos = absolute_pos.detach().clone()
        self.relative_pos = absolute_pos.detach().clone()
        # noinspection PyTypeChecker
        self.lowest_pos = torch.min(self.absolute_pos[:, :, 2], dim=1, keepdim=True).values
        for joint, parent_joint in reversed(H36mJoint.parent_joint_map.items()):
            self.relative_pos[:, joint] -= self.relative_pos[:, parent_joint]
        self.vr_joints = torch.zeros((6, 7), device=absolute_pos.device)
        self.temp_absolute_pos = self.absolute_pos.detach().clone()
        self.temp_relative_pos = self.relative_pos.detach().clone()

    def scale(self, scale: torch.Tensor):
        lim = torch.arange(self.absolute_pos.shape[0])
        # Rescale relative position
        self.temp_relative_pos[lim] = self.relative_pos[lim]
        for joint_id in range(H36mJoint.count):
            scale_id = H36mJoint.scale_map[H36mJoint(joint_id)]
            if scale_id != SkeletonScale.NO_SCALE:
                self.temp_relative_pos[lim, joint_id] *= scale[scale_id]
        # Regenerate absolute position
        for joint_id in range(H36mJoint.count):
            parent_joint = H36mJoint.parent_joint_map[H36mJoint(joint_id)]
            if parent_joint == H36mJoint.ROOT:
                self.temp_absolute_pos[lim, joint_id, :] = 0
            else:
                self.temp_absolute_pos[lim, joint_id] = \
                    self.temp_absolute_pos[lim, parent_joint] + \
                    self.temp_relative_pos[lim, joint_id]
        # Rescale absolute position
        temp_lowest_pos = torch.min(self.temp_absolute_pos[lim, :, 2], dim=1, keepdim=True).values
        offset = self.lowest_pos[lim] - temp_lowest_pos
        self.temp_absolute_pos[lim, :, 2] += offset

    def get_vr_joints(self, lim: Optional[Union[range, torch.Tensor]] = None):
        if lim is None:
            lim = torch.arange(self.absolute_pos.shape[0])
        poses = self.temp_absolute_pos
        head_top = poses[lim, H36mJoint.HEAD_TOP]  # 1 x n x 3
        face = poses[lim, H36mJoint.FACE]
        head_center = poses[lim, H36mJoint.HEAD]
        left_wrist = poses[lim, H36mJoint.LEFT_WRIST]
        left_thumb = poses[lim, H36mJoint.LEFT_THUMB]
        left_hand_top = poses[lim, H36mJoint.LEFT_INDEX]
        left_hand_center = (left_wrist + left_hand_top) / 2
        right_wrist = poses[lim, H36mJoint.RIGHT_WRIST]
        right_thumb = poses[lim, H36mJoint.RIGHT_THUMB]
        right_hand_top = poses[lim, H36mJoint.RIGHT_INDEX]
        right_hand_center = (right_wrist + right_hand_top) / 2
        hip_center = poses[lim, H36mJoint.HIP_CENTER]
        shoulder_center = poses[lim, H36mJoint.SHOULDER_CENTER]
        waist_real_center = (hip_center + shoulder_center) / 2
        waist_up = shoulder_center - waist_real_center
        waist_center = (waist_real_center + hip_center) / 2
        left_hip = poses[lim, H36mJoint.LEFT_HIP]
        right_hip = poses[lim, H36mJoint.RIGHT_HIP]
        hip_right = right_hip - left_hip
        left_shoulder = poses[lim, H36mJoint.LEFT_SHOULDER]
        right_shoulder = poses[lim, H36mJoint.RIGHT_SHOULDER]
        shoulder_right = right_shoulder - left_shoulder  # this may be incorrect
        waist_right = hip_right * 2 + shoulder_right  # more emphasis on hip
        waist_front = torch.cross(waist_up, waist_right)
        left_ankle = poses[lim, H36mJoint.LEFT_ANKLE]
        left_toe = poses[lim, H36mJoint.LEFT_TOE]
        left_knee = poses[lim, H36mJoint.LEFT_KNEE]
        left_foot_center = (left_ankle + left_toe) / 2
        right_ankle = poses[lim, H36mJoint.RIGHT_ANKLE]
        right_toe = poses[lim, H36mJoint.RIGHT_TOE]
        right_knee = poses[lim, H36mJoint.RIGHT_KNEE]
        right_foot_center = (right_ankle + right_toe) / 2

        # %%
        def normalize(vec):
            return vec / torch.norm(vec, dim=1, keepdim=True)

        def look_at_quaternion_tensor(up, front):
            up = normalize(up)
            front = normalize(front)
            right = torch.cross(front, up)
            up = torch.cross(right, front)
            mat = torch.stack([right, front, up], dim=2)
            return rotation_matrix_to_quaternion(mat)

        head_quat = look_at_quaternion_tensor(head_top - head_center, face - head_center)
        left_hand_quat = look_at_quaternion_tensor(left_thumb - left_wrist, left_hand_top - left_wrist)
        right_hand_quat = look_at_quaternion_tensor(right_thumb - right_wrist, right_hand_top - right_wrist)
        waist_quat = look_at_quaternion_tensor(waist_up, waist_front)
        left_foot_quat = look_at_quaternion_tensor(left_knee - left_ankle, left_toe - left_ankle)
        right_foot_quat = look_at_quaternion_tensor(right_knee - right_ankle, right_toe - right_ankle)

        head_info = torch.cat((head_center, normalize(head_quat)), dim=1)
        left_hand_info = torch.cat((left_hand_center, normalize(left_hand_quat)), dim=1)
        right_hand_info = torch.cat((right_hand_center, normalize(right_hand_quat)), dim=1)
        waist_info = torch.cat((waist_center, normalize(waist_quat)), dim=1)
        left_foot_info = torch.cat((left_foot_center, normalize(left_foot_quat)), dim=1)
        right_foot_info = torch.cat((right_foot_center, normalize(right_foot_quat)), dim=1)

        outputs = torch.stack(
            (head_info, left_hand_info, right_hand_info, waist_info, left_foot_info, right_foot_info),
            dim=1)
        return outputs
