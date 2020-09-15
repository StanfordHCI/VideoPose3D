# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch

from common.utils import wrap
from common.quaternion import qrot, qinverse, q_multiply


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]


def image_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Reverse camera frame normalization
    return (X + [1, h / w]) * w / 2


def world_to_camera(X, R, t):
    Rt = wrap(qinverse, R)  # Invert rotation
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t)  # Rotate and translate


def world_to_camera_quat(q, R):
    Rt = wrap(qinverse, R)  # Invert rotation
    return wrap(q_multiply, np.tile(Rt, (*q.shape[:-1], 1)), q)  # Rotate only


def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t


def camera_to_world_rot(q, R):
    return wrap(q_multiply, q, np.tile(R, (*q.shape[:-1], 1)))


def random_z_rot(size: list) -> np.array:
    """
    Generate random rotation quaternion in arbitrary shape
    Returns:
        object: np.array [*size, 4]
    """
    random_angle = 2 * np.random.rand(*size, 1) * np.pi
    quat_w = np.sin(random_angle)
    quat_x = np.zeros(size + [1])
    quat_y = np.zeros(size + [1])
    quat_z = np.cos(random_angle)
    return np.stack((quat_w, quat_x, quat_y, quat_z), axis=-1).astype('float32')


def random_x_y_shift(size: list, x_start=-2, x_end=2, y_start=-2, y_end=2) -> np.array:
    """
    Generate random position offset in arbitrary shape
    Returns:
        object: np.array [*size, 3]
    """
    vec_x = np.random.rand(*size, 1) * (x_end - x_start) + x_start
    vec_y = np.random.rand(*size, 1) * (y_end - y_start) + y_start
    vec_z = np.zeros(size + [1])
    return np.stack((vec_x, vec_y, vec_z), axis=-1).astype('float32')


def apply_transform(x, r, t):
    return wrap(qrot, r, x) + t


def apply_transform_rot(q, r):
    return wrap(q_multiply, r, q)


def apply_transform_combined(xq, r, t) -> np.array:
    """
    Apply offset and rotation to a list of positions and rotations
    Please make sure the first few axis of `xq`, `r`, and `t` have matched shape
    Returns:
        object: array of processed position and rotations
    """
    if xq.shape[-1] == 7:
        old_x = xq[..., :3]
        old_q = xq[..., 3:]
        new_x = apply_transform(old_x, r, t)
        new_q = apply_transform_rot(old_q, r)
        return np.concatenate((new_x, new_q), axis=-1)
    elif xq.shape[-1] == 3:
        new_x = apply_transform(xq, r, t)
        return new_x
    else:
        raise KeyboardInterrupt


def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.
    
    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:]

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    r2 = torch.sum(XX[..., :2] ** 2, dim=len(XX.shape) - 1, keepdim=True)

    radial = 1 + torch.sum(k * torch.cat((r2, r2 ** 2, r2 ** 3), dim=len(r2.shape) - 1), dim=len(r2.shape) - 1,
                           keepdim=True)
    tan = torch.sum(p * XX, dim=len(XX.shape) - 1, keepdim=True)

    XXX = XX * (radial + tan) + p * r2

    return f * XXX + c


def project_to_2d_linear(X, camera_params):
    """
    Project 3D points to 2D using only linear parameters (focal length and principal point).
    
    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)

    return f * XX + c
