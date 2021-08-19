import os

import argparse
import time
import numpy as np
from imageio import imsave
import matplotlib.pyplot as plt
from PIL import Image

import Datasets
import models

import torch
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.functional as F

import myUtils as utils
import data_transforms
from loss_functions import realEPE
from Test_KITTI import main as test_main


def test(args, device):
    test_main(args, device)


def train1(args, device):
    pass


def train2(args, device):
    pass


def main():

    dataset_names = sorted(name for name in Datasets.__all__)
    model_names = sorted(name for name in models.__all__)

    parser = argparse.ArgumentParser(
        description="Testing pan generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--data",
        metavar="DIR",
        default="C:\\Users\\Kaist\\Desktop",
        help="path to dataset",
    )
    parser.add_argument(
        "-tn",
        "--tdataName",
        metavar="Test Data Set Name",
        default="Kitti_eigen_test_improved",
        choices=dataset_names,
    )
    parser.add_argument(
        "-relbase",
        "--rel_baselne",
        default=1,
        help="Relative baseline of testing dataset",
    )
    parser.add_argument("-mdisp", "--max_disp", default=300)  # of the training patch W
    parser.add_argument("-mindisp", "--min_disp", default=2)  # of the training patch W
    parser.add_argument("-b", "--batch_size", metavar="Batch Size", default=1)
    parser.add_argument("-eval", "--evaluate", default=True)
    parser.add_argument("-save", "--save", default=False)
    parser.add_argument("-save_pc", "--save_pc", default=False)
    parser.add_argument("-save_pan", "--save_pan", default=False)
    parser.add_argument("-save_input", "--save_input", default=False)
    parser.add_argument("-w", "--workers", metavar="Workers", default=4, type=int)
    parser.add_argument(
        "--sparse",
        default=False,
        action="store_true",
        help="Depth GT is sparse, automatically seleted when choosing a KITTIdataset",
    )
    parser.add_argument(
        "--print-freq", "-p", default=10, type=int, metavar="N", help="print frequency"
    )
    parser.add_argument(
        "-gpu",
        "--gpu_indices",
        default=[],
        type=int,
        nargs="+",
        help="GPU indices to train on. Trains on CPU if none are supplied.",
    )
    parser.add_argument(
        "-dt",
        "--dataset",
        help="Dataset and training stage directory",
        default="Kitti_stage2",
    )
    parser.add_argument(
        "-ts", "--time_stamp", help="Model timestamp", default="10-18-15_42"
    )
    parser.add_argument("-m", "--model", help="Model", default="FAL_netB")
    parser.add_argument(
        "-no_levels",
        "--no_levels",
        default=49,
        help="Number of quantization levels in MED",
    )
    parser.add_argument(
        "-dtl",
        "--details",
        help="details",
        default=",e20es,b4,lr5e-05/checkpoint.pth.tar",
    )
    parser.add_argument(
        "-fpp",
        "--f_post_process",
        default=False,
        help="Post-processing with flipped input",
    )
    parser.add_argument(
        "-mspp",
        "--ms_post_process",
        default=True,
        help="Post-processing with multi-scale input",
    )
    parser.add_argument(
        "-median",
        "--median",
        default=False,
        help="use median scaling (not needed when training from stereo",
    )
    parser.add_argument(
        "-mo",
        "--modus_operandi",
        default="test",
        help="Select the modus operandi.",
        choices=["test", "train1", "train2"],
    )

    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu_indices else "cpu")

    os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join(
        [str(item) for item in args.gpu_indices]
    )

    if args.modus_operandi == "test":
        test(args, device)
    elif args.modus_operandi == "train1":
        train1(args, device)
    elif args.modus_operandi == "train2":
        train2(args, device)
    else:
        raise Exception(f"{args.modus_operandi} is not a valid modus operandi.")


if __name__ == "__main__":

    main()
