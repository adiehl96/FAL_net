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

# import torch.utils.data as data
from torch.utils import data
import os
import os.path
from imageio import imread

LR_DATASETS = ["Kitti_eigen_test_improved"]


def img_loader(path_img):
    print("path_img", path_img)
    img = imread(path_img)
    return img


class ListDataset(data.Dataset):
    def __init__(
        self,
        path_list,
        transform=None,
    ):
        self.path_list = path_list
        self.transform = transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        input = self.path_list[index]

        file_name = os.path.basename(input)[:-4]
        input = img_loader(input)

        if self.transform is not None:
            input = self.transform(input)
        return [input], 0, file_name
