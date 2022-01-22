import csv, pickle
import os.path as path
from misc.listdataset_test import ListDataset as TestListDataset
from misc.listdataset_train import ListDataset as TrainListDataset
from misc.listdataset_retrain import ListDataset as RetrainListDataset


from random import shuffle
from misc.utils import flatten
import numpy as np


def apply(item, fun):
    if isinstance(item, list):
        return [apply(x, fun) for x in item]
    else:
        return fun(item)


def f(x):
    return {
        "eigen_test_improved": [[0, 1], [2, 3]],
        "eigen_test_classic": [0],
        "eigen_train": [0, 1],
        "bello_val": [[0, 1, 2, 3], [4, 5]],
    }[x]


def load_data(split=None, **kwargs):
    input_root = kwargs.pop("root")
    dataset = kwargs.pop("dataset")
    transform = kwargs.pop("transform", None)
    target_transform = kwargs.pop("target_transform", None)
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

    # datasetlist = datasetlist[:1000]

    if (split == "eigen_test_improved" and dataset == "KITTI") or (
        split == "bello_val" and dataset == "KITTI2015"
    ):
        dataset = TestListDataset(
            datasetlist,
            split=split,
            disp=True,
            transform=transform,
            target_transform=target_transform,
        )
    if split in ["eigen_test_classic", "eigen_train"] and dataset == "KITTI":
        dataset = TrainListDataset(datasetlist, transform=transform)
    elif dataset == "ASM_stereo_small_train" or dataset == "ASM_stereo_train":
        if create_val:
            np.random.default_rng().shuffle(datasetlist, axis=0)
            val_size = int(len(datasetlist) * create_val)
            val_list = datasetlist[:val_size]
            datasetlist = datasetlist[val_size:]

            val_set = RetrainListDataset(val_list, transform=transform)

            dataset = RetrainListDataset(datasetlist, transform=transform)
            return dataset, val_set
        else:
            dataset = RetrainListDataset(datasetlist, transform=transform)
    elif dataset == "ASM_stereo_small_test" or dataset == "ASM_stereo_test":
        dataset = RetrainListDataset(datasetlist, transform=transform)

    return dataset
