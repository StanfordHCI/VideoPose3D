# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

from common.mocap_dataset import MocapDataset
from common.skeleton import Skeleton

# 0   , 1        , 2         , 3    , 4        , 5
# head, left hand, right hand, waist, left foot, right foot
mpi_skeleton = Skeleton(parents=[-1, 0, 0, 0, 3, 3],
                        joints_left=[1, 4],
                        joints_right=[2, 5],
                        input_joints=[0, 1, 2],
                        output_joints=[3, 4, 5])


class MpiDataset(MocapDataset):
    def __init__(self, path):
        super().__init__(fps=25, skeleton=mpi_skeleton)

        self._cameras = list(range(8))
        self._data = {}

        data = np.load(path, allow_pickle=True)['pos_rot'].item()

        for subject, actions in data.items():
            self._data[subject] = {}
            for action_name, pos_rot in actions.items():
                self._data[subject][action_name] = {
                    'pos_rot': pos_rot,
                    'cameras': self._cameras,
                }
