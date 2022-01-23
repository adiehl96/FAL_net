# data_utils.py Contains various useful data related functions
# Copyright (C) 2021  Juan Luis Gonzalez Bello (juanluisgb@kaist.ac.kr)
# Copyright (C) 2022  Arne Diehl (floaty.press-0k@icloud.com)
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

import csv, pickle, random
import os.path as path

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms.functional import hflip

from misc.listdataset import ListDataset
from misc.functional import flatten, apply


def f(x):
    return {
        "eigen_test_improved": [[0, 1], [2, 3]],
        "eigen_test_classic": [[0], []],
        "eigen_train": [[0, 1], []],
        "bello_val": [[0, 1, 2, 3], [4, 5]],
    }[x]


def load_data(split=None, **kwargs):
    input_root = kwargs.pop("root")
    dataset = kwargs.pop("dataset")
    transform = kwargs.pop("transform", lambda x: x)
    create_val = kwargs.pop("create_val", False)

    if "ASM" in dataset:
        with open(path.join(input_root, dataset), "rb") as fp:
            datasetlist = pickle.load(fp)
    elif split is not None:
        splitfilelocation = f"./splits/{dataset}/{split}.txt"
        try:
            datasetfile = open(splitfilelocation)
        except:
            raise Exception(f"Could not open file at {splitfilelocation}.")

        datasetreader = csv.reader(datasetfile, delimiter=",")
        datasetlist = []
        for row in datasetreader:
            files = apply(f(split), lambda x: path.join(input_root, row[x]))
            for item in flatten(files):
                if item != None and not path.isfile(item):
                    raise Exception(f"Could not load file in location {item}.")
            datasetlist.append(files)

    # datasetlist = datasetlist[:800]

    dataset = ListDataset(datasetlist, transform)
    if create_val:
        np.random.default_rng().shuffle(datasetlist, axis=0)
        val_size = int(len(datasetlist) * create_val)

        val_set = ListDataset(datasetlist[:val_size], transform)
        dataset = ListDataset(datasetlist[val_size:], transform)

        return dataset, val_set

    return dataset


class NormalizeInverse(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std = std.eq(0).mul(1e-7).add(std)
        std_inv = 1 / std
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)


class ApplyToMultiple:
    def __init__(
        self,
        transform,
        RandomHorizontalFlipChance=0,
        same_rand_state=True,
    ):
        self.transform = transform
        self.same_rand_state = same_rand_state
        self.RandomHorizontalFlipChance = RandomHorizontalFlipChance

    def _apply_to_features(self, transform, input, same_rand_state):
        if same_rand_state:

            # move randomness
            np.random.rand()
            random.random()
            torch.rand(1)

            # save state
            np_state = np.random.get_state()
            rd_state = random.getstate()
            tr_state = torch.random.get_rng_state()

        intermediate = input
        if self.RandomHorizontalFlipChance:
            if torch.rand(1) < self.RandomHorizontalFlipChance:
                intermediate = [hflip(x) for x in input]
                intermediate.reverse()
            torch.set_rng_state(tr_state)

        output = []
        for item in intermediate:
            output.append(transform(item))

            if same_rand_state:
                np.random.set_state(np_state)
                random.setstate(rd_state)
                torch.set_rng_state(tr_state)

        return output

    def __call__(self, input_list):
        return self._apply_to_features(self.transform, input_list, self.same_rand_state)
