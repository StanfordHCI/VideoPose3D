# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import sys
from glob import glob

import numpy as np
import scipy.io

from convert_mpi_to_vr import convert_mpi_to_pos_rot

sys.path.append('../')

output_filename = 'data_3d_h36m_mpi'
subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']

if __name__ == '__main__':
    if os.path.basename(os.getcwd()) != 'data':
        print('This script must be launched from the "data" directory')
        exit(0)

    parser = argparse.ArgumentParser(description='MPI-INF-3DHP dataset downloader/converter')

    parser.add_argument('--from-source', default='', type=str, metavar='PATH', help='convert original dataset')

    args = parser.parse_args()

    if os.path.exists(output_filename + '.npz'):
        print('The dataset already exists at', output_filename + '.npz')
        exit(0)

    elif args.from_source:
        print('Converting original MPI dataset from', args.from_source)
        output = {}

        from scipy.io import loadmat

        for subject in subjects:
            output[subject] = {}
            saved_lengths = []
            mat_file_path_list = glob(os.path.join(args.from_source, f'{subject}/*/annot.mat'))
            file_list = glob(args.from_source + '/' + subject + '/*/annot.mat')
            print(file_list)
            for f in file_list:
                action = os.path.split(os.path.split(f)[0])[-1]

                hf = scipy.io.loadmat(f)['annot3']
                output[subject][action] = []
                for i in range(8):
                    positions, lengths = convert_mpi_to_pos_rot(hf[i, 0])
                    output[subject][action] += [positions.astype('float32')]
                    saved_lengths += [lengths]
            avg_lengths = np.mean(saved_lengths, axis=0)
            output[subject]["lengths"] = avg_lengths

        print('Saving...')
        np.savez_compressed(output_filename, positions_3d=output)

        print('Done.')

        print('Saving...')
        np.savez_compressed(output_filename, pos_rot=output)

        print('Done.')

    else:
        print('Please specify the dataset source')
        exit(0)
