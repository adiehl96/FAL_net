import os.path
from Datasets.listdataset_test import ListDataset as TestListDataset
from Datasets.listdataset_train import ListDataset as TrainListDataset
from random import shuffle
import csv
from utils.myUtils import flatten


def load_data(split, **kwargs):
    input_root = kwargs.pop("root")
    dataset = kwargs.pop("dataset")
    transform = kwargs.pop("transform", None)
    target_transform = kwargs.pop("target_transform", None)
    shuffle_test = kwargs.pop("shuffle_test", False)
    reference_transform = kwargs.pop("reference_transform", None)
    co_transform = kwargs.pop("co_transform", None)
    max_pix = kwargs.pop("max_pix", 100)
    fix = kwargs.pop("fix", False)


    if dataset == "Kitti_eigen_test_improved":
        splitfilelocation = "./Datasets/split/eigen_test_improved.txt"
    elif dataset == "KITTI":
        splitfilelocation = "./Datasets/split/eigen_train.txt"

    try:
        datasetfile = open(splitfilelocation)
    except:
        raise Exception(f"Could not open file at {splitfilelocation}.")

    datasetreader = csv.reader(datasetfile, delimiter=",")
    datasetlist = []
    for row in datasetreader:
        if dataset == "Kitti_eigen_test_improved":
            inputleft = f"{input_root}/{row[0]}"
            inputright = f"{input_root}/{row[1]}"
            groundtruthleft = f"{input_root}/{row[2]}"
            velodyneleft = f"{input_root}/{row[3]}"
            files = [[inputleft, inputright], [groundtruthleft, velodyneleft]]
        elif dataset == "KITTI":
            inputleft = f"{input_root}/{row[0]}"
            inputright = f"{input_root}/{row[1]}"
            files = [[inputleft, inputright], None]

        if all(map(lambda x: True if x == None else os.path.isfile(x), flatten(files))):
            datasetlist.append(files)
        else:
            for item in flatten(files):
                if item != None and not os.path.isfile(item):
                    raise Exception(f"Could not load file in location {item}.")

    if shuffle_test:
        shuffle(datasetlist)

    if dataset == "Kitti_eigen_test_improved":
        dataset = TestListDataset(
            input_root,
            input_root,
            datasetlist,
            data_name="Kitti_eigen_test_improved",
            disp=True,
            of=False,
            transform=transform,
            target_transform=target_transform,
        )
    elif dataset == "KITTI":
        dataset = TrainListDataset(
            input_root,
            input_root,
            datasetlist,
            data_name="Kitti2015",
            disp=False,
            of=False,
            transform=transform,
            target_transform=target_transform,
            co_transform=co_transform,
            max_pix=max_pix,
            reference_transform=reference_transform,
            fix=fix,
        )


    return dataset
