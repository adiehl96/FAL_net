import os

import argparse
import time
import numpy as np
from imageio import imsave
import matplotlib.pyplot as plt
from PIL import Image

import models

import torch
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.functional as F

from misc import utils
from misc import data_transforms
from misc.loss_functions import realEPE
from Test_KITTI import main as test
from Train_Stage1_K import main as train1
from Train_Stage2_K import main as train2


def main():

    model_names = sorted(name for name in models.__all__)

    parser = argparse.ArgumentParser(
        description="FAL_net in pytorch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--data",
        metavar="DIR",
        default="/dataQ/arne",
        help="path to dataset",
    )

    parser.add_argument(
        "-n0",
        "--dataName0",
        metavar="Data Set Name 0",
        default="KITTI",
    )

    parser.add_argument(
        "-tn",
        "--tdataName",
        metavar="Test Data Set Name",
        default="Kitti_eigen_test_improved",
    )

    parser.add_argument(
        "-vdn",
        "--vdataName",
        metavar="Val data set Name",
        default="KITTI2015",
    )

    parser.add_argument(
        "-relbase_test",
        "--rel_baset",
        default=1,
        help="Relative baseline of testing dataset",
    )

    parser.add_argument("-mdisp", "--max_disp", default=300)  # of the training patch W
    parser.add_argument("-mindisp", "--min_disp", default=2)  # of the training patch W
    parser.add_argument("-b", "--batch_size", metavar="Batch Size", default=1, type=int)
    parser.add_argument("-eval", "--evaluate", default=True)
    parser.add_argument("-tbs", "--tbatch_size", metavar="Val Batch Size", default=1)
    parser.add_argument("-op", "--optimizer", metavar="Optimizer", default="adam")
    parser.add_argument(
        "--beta",
        metavar="BETA",
        type=float,
        help="Beta parameter for adam",
        default=0.999,
    )
    parser.add_argument("-cw", "--crop_width", metavar="Batch crop W Size", default=640)
    parser.add_argument("-save", "--save", default=False)
    parser.add_argument("--lr", metavar="learning Rate", default=0.0001)
    # todo for train2 the default learning rate is 0.00005
    parser.add_argument("-save_pc", "--save_pc", default=False)
    parser.add_argument("-save_pan", "--save_pan", default=False)
    parser.add_argument(
        "--momentum",
        default=0.5,
        type=float,
        metavar="Momentum",
        help="Momentum for Optimizer",
    )

    parser.add_argument(
        "-ch", "--crop_height", metavar="Batch crop H Size", default=192
    )
    parser.add_argument("-save_input", "--save_input", default=False)
    parser.add_argument("-w", "--workers", metavar="Workers", default=4, type=int)
    parser.add_argument(
        "--sparse",
        default=True,
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
        "-mm",
        "--m_model",
        metavar="Mono Model",
        default="FAL_netB",
        choices=model_names,
    )
    parser.add_argument(
        "-smooth", "--a_sm", default=0.2 * 2 / 512, help="Smoothness loss weight"
    )  # todo: the smoothness loss weight of train2 was 0.4*2/512 but I don't know why
    parser.add_argument("-mirror_loss", "--a_mr", default=1, help="Mirror loss weight")

    parser.add_argument("-perc", "--a_p", default=0.01, help="Perceptual loss weight")

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
    parser.add_argument(
        "--milestones",
        default=[30, 40],
        metavar="N",
        nargs="*",
        help="epochs at which learning rate is divided by 2",
    )
    # todo: the default for train2 is [5, 10],
    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=0.0,
        type=float,
        metavar="W",
        help="weight decay",
    )

    parser.add_argument(
        "--epochs",
        default=50,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    # todo default for train stage 2 is 20
    parser.add_argument(
        "--epoch_size",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch size (will match dataset size if set to 0)",
    )

    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )

    parser.add_argument(
        "--fix_model",
        dest="fix_model",
        default="KITTI_stage1/08-20-13_25/FAL_netB,e50es,b1,lr0.0001/checkpoint.pth.tar",
    )

    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        default=None,
        help="path to pre-trained model",
    )
    # todo train2
    #     parser.add_argument(
    #     "--pretrained",
    #     dest="pretrained",
    #     default="KITTI_stage1/08-20-13_25/FAL_netB,e50es,b1,lr0.0001/checkpoint.pth.tar",
    #     help="directory of run",
    # )

    parser.add_argument(
        "--bias-decay", default=0.0, type=float, metavar="B", help="bias decay"
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
