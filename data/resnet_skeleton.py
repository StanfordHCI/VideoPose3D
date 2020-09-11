from collections import OrderedDict
from enum import IntEnum
from typing import Optional, Union

import torch

from data.skeleton_scale import SkeletonScale


class RealResnetJoint(IntEnum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    LEFT_TOE = 17
    LEFT_HEEL = 18
    RIGHT_TOE = 19
    RIGHT_HEEL = 20

    count = 21


class ResnetJoint(IntEnum):
    ROOT = -1
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    SHOULDER_CENTER = 5
    HIP_CENTER = 6
    LEFT_SHOULDER = 7
    RIGHT_SHOULDER = 8
    LEFT_ELBOW = 9
    RIGHT_ELBOW = 10
    LEFT_WRIST = 11
    RIGHT_WRIST = 12
    LEFT_HIP = 13
    RIGHT_HIP = 14
    LEFT_KNEE = 15
    RIGHT_KNEE = 16
    LEFT_ANKLE = 17
    RIGHT_ANKLE = 18
    LEFT_HEEL = 19
    RIGHT_HEEL = 20
    LEFT_TOE = 21
    RIGHT_TOE = 22

    count = 23

    @staticmethod
    def from_real_resnet(real_resnet: torch.Tensor):
        real_resnet_size = list(real_resnet.shape)
        real_resnet_size[-2] = int(ResnetJoint.count)
        resnet = torch.zeros(size=real_resnet_size, device=real_resnet.device)
        resnet[..., ResnetJoint.resnet_ids, :] = real_resnet[..., ResnetJoint.real_resnet_ids, :]
        resnet[..., ResnetJoint.SHOULDER_CENTER, :] = \
            (real_resnet[..., RealResnetJoint.LEFT_SHOULDER, :] +
             real_resnet[..., RealResnetJoint.RIGHT_SHOULDER, :]) / 2
        resnet[..., ResnetJoint.HIP_CENTER, :] = \
            (real_resnet[..., RealResnetJoint.LEFT_HIP, :] + real_resnet[..., RealResnetJoint.RIGHT_HIP, :]) / 2
        return resnet

    @staticmethod
    def to_real_resnet(resnet: torch.Tensor):
        resnet_size = list(resnet.shape)
        resnet_size[-2] = int(RealResnetJoint.count)
        real_resnet = torch.zeros(size=resnet_size, device=resnet.device)
        real_resnet[..., ResnetJoint.real_resnet_ids, :] = resnet[..., ResnetJoint.resnet_ids, :]
        return real_resnet


class ResnetSkeleton:
    confidences: torch.Tensor
    relative_pos: torch.Tensor
    average: torch.Tensor
    temp_relative_pos: torch.Tensor
    absolute_pos: torch.Tensor
    temp_absolute_pos: torch.Tensor
    vr_joints: torch.Tensor

    def __init__(self, real_resnet):
        resnet = ResnetJoint.from_real_resnet(real_resnet)
        self.confidences = resnet[..., [2]]
        self.absolute_pos = resnet[..., :2].detach().clone()
        self.relative_pos = self.absolute_pos.detach().clone()
        # noinspection PyTypeChecker
        self.average = torch.mean(self.absolute_pos, dim=1, keepdim=True)
        for joint, parent_joint in reversed(ResnetJoint.parent_joint_map.items()):
            self.relative_pos[:, joint] -= self.relative_pos[:, parent_joint]
        self.temp_absolute_pos = self.absolute_pos.detach().clone()
        self.temp_relative_pos = self.relative_pos.detach().clone()

    def scale(self, scale: torch.Tensor, lim: Optional[Union[range, torch.Tensor]] = None):
        if lim is None:
            lim = torch.arange(self.absolute_pos.shape[0])
        # Rescale relative position
        self.temp_relative_pos[lim] = self.relative_pos[lim]
        for joint_id in range(ResnetJoint.count):
            scale_id = ResnetJoint.scale_map[ResnetJoint(joint_id)]
            if scale_id != SkeletonScale.NO_SCALE:
                self.temp_relative_pos[lim, joint_id] *= scale[scale_id]
        # Regenerate absolute position
        for joint_id in range(ResnetJoint.count):
            parent_joint = ResnetJoint.parent_joint_map[ResnetJoint(joint_id)]
            if parent_joint == ResnetJoint.ROOT:
                self.temp_absolute_pos[lim, joint_id, :] = 0
            else:
                self.temp_absolute_pos[lim, joint_id] = \
                    self.temp_absolute_pos[lim, parent_joint] + \
                    self.temp_relative_pos[lim, joint_id]
        # Rescale absolute position
        temp_average = torch.mean(self.temp_absolute_pos[lim], dim=1, keepdim=True)
        offset = self.average[lim] - temp_average
        self.temp_absolute_pos[lim] += offset

    def get_resnet_joints(self, lim: Optional[Union[range, torch.Tensor]] = None):
        if lim is None:
            lim = torch.arange(self.absolute_pos.shape[0])
        resnet = torch.cat((self.temp_absolute_pos[lim], self.confidences[lim]), dim=-1)
        return ResnetJoint.to_real_resnet(resnet)


ResnetJoint.parent_joint_map = OrderedDict({
    ResnetJoint.NOSE: ResnetJoint.ROOT,
    ResnetJoint.LEFT_EYE: ResnetJoint.NOSE,
    ResnetJoint.RIGHT_EYE: ResnetJoint.NOSE,
    ResnetJoint.LEFT_EAR: ResnetJoint.NOSE,
    ResnetJoint.RIGHT_EAR: ResnetJoint.NOSE,
    ResnetJoint.SHOULDER_CENTER: ResnetJoint.NOSE,
    ResnetJoint.HIP_CENTER: ResnetJoint.NOSE,
    ResnetJoint.LEFT_SHOULDER: ResnetJoint.SHOULDER_CENTER,
    ResnetJoint.RIGHT_SHOULDER: ResnetJoint.SHOULDER_CENTER,
    ResnetJoint.LEFT_ELBOW: ResnetJoint.LEFT_SHOULDER,
    ResnetJoint.RIGHT_ELBOW: ResnetJoint.RIGHT_SHOULDER,
    ResnetJoint.LEFT_WRIST: ResnetJoint.LEFT_ELBOW,
    ResnetJoint.RIGHT_WRIST: ResnetJoint.RIGHT_ELBOW,
    ResnetJoint.LEFT_HIP: ResnetJoint.HIP_CENTER,
    ResnetJoint.RIGHT_HIP: ResnetJoint.HIP_CENTER,
    ResnetJoint.LEFT_KNEE: ResnetJoint.LEFT_HIP,
    ResnetJoint.RIGHT_KNEE: ResnetJoint.RIGHT_HIP,
    ResnetJoint.LEFT_ANKLE: ResnetJoint.LEFT_KNEE,
    ResnetJoint.RIGHT_ANKLE: ResnetJoint.RIGHT_KNEE,
    ResnetJoint.LEFT_HEEL: ResnetJoint.LEFT_ANKLE,
    ResnetJoint.RIGHT_HEEL: ResnetJoint.RIGHT_ANKLE,
    ResnetJoint.LEFT_TOE: ResnetJoint.LEFT_HEEL,
    ResnetJoint.RIGHT_TOE: ResnetJoint.RIGHT_HEEL
})

ResnetJoint.scale_map = {
    ResnetJoint.NOSE: SkeletonScale.NO_SCALE,
    ResnetJoint.LEFT_EYE: SkeletonScale.NO_SCALE,
    ResnetJoint.RIGHT_EYE: SkeletonScale.NO_SCALE,
    ResnetJoint.LEFT_EAR: SkeletonScale.NO_SCALE,
    ResnetJoint.RIGHT_EAR: SkeletonScale.NO_SCALE,
    ResnetJoint.SHOULDER_CENTER: SkeletonScale.HEAD_NECK,
    ResnetJoint.HIP_CENTER: SkeletonScale.SHOULDER_HIP,
    ResnetJoint.LEFT_SHOULDER: SkeletonScale.SHOULDER_LENGTH,
    ResnetJoint.RIGHT_SHOULDER: SkeletonScale.SHOULDER_LENGTH,
    ResnetJoint.LEFT_ELBOW: SkeletonScale.LEFT_SHOULDER_ELBOW,
    ResnetJoint.RIGHT_ELBOW: SkeletonScale.RIGHT_SHOULDER_ELBOW,
    ResnetJoint.LEFT_WRIST: SkeletonScale.LEFT_ELBOW_WRIST,
    ResnetJoint.RIGHT_WRIST: SkeletonScale.RIGHT_ELBOW_WRIST,
    ResnetJoint.LEFT_HIP: SkeletonScale.HIP_LENGTH,
    ResnetJoint.RIGHT_HIP: SkeletonScale.HIP_LENGTH,
    ResnetJoint.LEFT_KNEE: SkeletonScale.LEFT_HIP_KNEE,
    ResnetJoint.RIGHT_KNEE: SkeletonScale.RIGHT_HIP_KNEE,
    ResnetJoint.LEFT_ANKLE: SkeletonScale.LEFT_KNEE_ANKLE,
    ResnetJoint.RIGHT_ANKLE: SkeletonScale.RIGHT_KNEE_ANKLE,
    ResnetJoint.LEFT_HEEL: SkeletonScale.NO_SCALE,
    ResnetJoint.RIGHT_HEEL: SkeletonScale.NO_SCALE,
    ResnetJoint.LEFT_TOE: SkeletonScale.NO_SCALE,
    ResnetJoint.RIGHT_TOE: SkeletonScale.NO_SCALE
}

ResnetJoint.resnet_ids = [
    int(ResnetJoint.NOSE),
    int(ResnetJoint.LEFT_EYE),
    int(ResnetJoint.RIGHT_EYE),
    int(ResnetJoint.LEFT_EAR),
    int(ResnetJoint.RIGHT_EAR),
    int(ResnetJoint.LEFT_SHOULDER),
    int(ResnetJoint.RIGHT_SHOULDER),
    int(ResnetJoint.LEFT_ELBOW),
    int(ResnetJoint.RIGHT_ELBOW),
    int(ResnetJoint.LEFT_WRIST),
    int(ResnetJoint.RIGHT_WRIST),
    int(ResnetJoint.LEFT_HIP),
    int(ResnetJoint.RIGHT_HIP),
    int(ResnetJoint.LEFT_KNEE),
    int(ResnetJoint.RIGHT_KNEE),
    int(ResnetJoint.LEFT_ANKLE),
    int(ResnetJoint.RIGHT_ANKLE),
    int(ResnetJoint.LEFT_TOE),
    int(ResnetJoint.LEFT_HEEL),
    int(ResnetJoint.RIGHT_TOE),
    int(ResnetJoint.RIGHT_HEEL),
]
ResnetJoint.real_resnet_ids = [
    int(RealResnetJoint.NOSE),
    int(RealResnetJoint.LEFT_EYE),
    int(RealResnetJoint.RIGHT_EYE),
    int(RealResnetJoint.LEFT_EAR),
    int(RealResnetJoint.RIGHT_EAR),
    int(RealResnetJoint.LEFT_SHOULDER),
    int(RealResnetJoint.RIGHT_SHOULDER),
    int(RealResnetJoint.LEFT_ELBOW),
    int(RealResnetJoint.RIGHT_ELBOW),
    int(RealResnetJoint.LEFT_WRIST),
    int(RealResnetJoint.RIGHT_WRIST),
    int(RealResnetJoint.LEFT_HIP),
    int(RealResnetJoint.RIGHT_HIP),
    int(RealResnetJoint.LEFT_KNEE),
    int(RealResnetJoint.RIGHT_KNEE),
    int(RealResnetJoint.LEFT_ANKLE),
    int(RealResnetJoint.RIGHT_ANKLE),
    int(RealResnetJoint.LEFT_TOE),
    int(RealResnetJoint.LEFT_HEEL),
    int(RealResnetJoint.RIGHT_TOE),
    int(RealResnetJoint.RIGHT_HEEL),
]
