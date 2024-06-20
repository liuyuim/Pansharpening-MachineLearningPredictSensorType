# ------------------------------------------------------------------
# Copyright (c) 2021, Zi-Rong Jin, Tian-Jing Zhang, Cheng Jin, and 
# Liang-Jian Deng, All rights reserved.
#
# This work is licensed under GNU Affero General Public License
# v3.0 International To view a copy of this license, see the
# LICENSE file.
#
# This file is running on WorldView-3 dataset. For other dataset
# (i.e., QuickBird and GaoFen-2), please change the corresponding
# inputs.
# ------------------------------------------------------------------

import torch.utils.data as data
import torch
import h5py
import cv2
import numpy as np

import scipy.io as sio
import mat73

def get_edge(data):  # for training
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - \
                cv2.boxFilter(data[i, :, :], -1, (5, 5))
        else:
            rs[i, :, :, :] = data[i, :, :, :] - \
                cv2.boxFilter(data[i, :, :, :], -1, (5, 5))
    return rs


class Dataset_Pro(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro, self).__init__()
        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3
        # data = mat73.loadmat(file_path)  # NxCxHxW = 0x1x2x3
        # data = sio.loadmat(file_path)  # NxCxHxW = 0x1x2x3

        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / 2047
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:
        print('gt:',self.gt.shape)

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / 2047
        self.lms = torch.from_numpy(lms1)
        print('lms:',self.lms.shape)

        ms1 = data["ms"][...]  # NxCxHxW
        ms1 = np.array(ms1.transpose(0, 2, 3, 1),
                       dtype=np.float32) / 2047  # NxHxWxC
        ms1_tmp = get_edge(ms1)  # NxHxWxC
        self.ms_hp = torch.from_numpy(ms1_tmp).permute(0, 3, 1, 2)  # NxCxHxW:
        print('ms_hp:',self.ms_hp.shape)

        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1.transpose(0, 2, 3, 1),
                        dtype=np.float32) / 2047  # NxHxWx1
        pan1 = np.squeeze(pan1, axis=3)  # NxHxW
        pan_hp_tmp = get_edge(pan1)   # NxHxW
        pan_hp_tmp = np.expand_dims(pan_hp_tmp, axis=3)   # NxHxWx1
        # Nx1xHxW:
        self.pan_hp = torch.from_numpy(pan_hp_tmp).permute(0, 3, 1, 2)
        print('pan_hp:',self.pan_hp.shape)

        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / 2047  # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:
        print('pan:',self.pan.shape)

    def __getitem__(self, index):
        return self.gt[index, :, :, :].float(), \
            self.lms[index, :, :, :].float(), \
            self.ms_hp[index, :, :, :].float(), \
            self.pan_hp[index, :, :, :].float(), \
            self.pan[index, :, :, :].float()  # Nx1xHxW:

    def __len__(self):
        return self.gt.shape[0]
