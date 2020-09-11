from enum import IntEnum

import torch


class SkeletonScale(IntEnum):
    NO_SCALE = -1
    HEAD_NECK = 0
    SHOULDER_LENGTH = 1
    SHOULDER_HIP = 2
    HIP_LENGTH = 3
    LEFT_SHOULDER_ELBOW = 4
    RIGHT_SHOULDER_ELBOW = 5
    LEFT_ELBOW_WRIST = 6
    RIGHT_ELBOW_WRIST = 7
    LEFT_HIP_KNEE = 8
    RIGHT_HIP_KNEE = 9
    LEFT_KNEE_ANKLE = 10
    RIGHT_KNEE_ANKLE = 11

    @staticmethod
    def random_scale(min_scale=0.8, max_scale=1.2) -> torch.Tensor:
        scales = torch.rand(size=[int(SkeletonScale.count)]) * (max_scale - min_scale) + min_scale
        # make sure the scale on the left is the same as left
        scales[SkeletonScale.left_scales] = scales[SkeletonScale.right_scales]
        return scales


SkeletonScale.left_scales = [
    SkeletonScale.LEFT_SHOULDER_ELBOW,
    SkeletonScale.LEFT_ELBOW_WRIST,
    SkeletonScale.LEFT_HIP_KNEE,
    SkeletonScale.LEFT_KNEE_ANKLE
]
SkeletonScale.right_scales = [
    SkeletonScale.RIGHT_SHOULDER_ELBOW,
    SkeletonScale.RIGHT_ELBOW_WRIST,
    SkeletonScale.RIGHT_HIP_KNEE,
    SkeletonScale.RIGHT_KNEE_ANKLE
]

SkeletonScale.count = 12
