# listdataset_test.py Load training images during testing
# Copyright (C) 2021  Juan Luis Gonzalez Bello (juanluisgb@kaist.ac.kr)
# This software is not for commercial use
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import torch.utils.data as data
import os
import os.path
from imageio import imread
import numpy as np
from PIL import Image


def img_loader(path_img):
    img = Image.open(path_img)
    return img


def kittidisp_loader(path_img):
    disp = imread(path_img) / 256
    return disp[:, :, np.newaxis]


class ListDataset(data.Dataset):
    def __init__(
        self,
        path_list,
        disp=False,
        split="bello_val",
        transform=None,
        target_transform=None,
    ):
        self.path_list = path_list
        self.transform = transform
        self.target_transform = target_transform
        self.disp = disp
        self.split = split

        if split in ["bello_val", "eigen_test_improved"]:
            self.input_loader = img_loader
            if self.disp:
                self.target_loader = kittidisp_loader

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        inputs, targets = self.path_list[index]

        if self.split == "eigen_test_improved":
            if self.disp:
                targets = [
                    self.target_loader(targets[0]),
                    self.target_loader(targets[1]),
                ]
        else:
            if self.disp:
                targets = [self.target_loader(targets[0])]

        file_name = os.path.basename(inputs[0])[:-4]
        inputs = [
            self.input_loader(inputs[0]),
            self.input_loader(inputs[1]),
        ]

        if self.transform is not None:
            for i in range(len(inputs)):
                inputs[i] = self.transform(inputs[i])

        if targets is None:
            return inputs, 0, file_name

        if self.target_transform is not None:
            for i in range(len(targets)):
                targets[i] = self.target_transform(targets[i])

        return inputs, targets, file_name
