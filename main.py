import sys
import os

from misc.download import check_kitti_availability
from misc.flags import specific_argparse
from misc.utils import print_and_save_config
from misc.save_path_handler import make_save_path


def main():
    print(" ".join(sys.argv[:]))

    args, kitti_needed, script = specific_argparse()

    os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join(
        [str(item) for item in args.gpu_indices]
    )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if kitti_needed:
        check_kitti_availability(args)

    make_save_path(args, script)
    print_and_save_config(args)

    import torch

    from testing.test_k import main as testk
    from testing.test_k_eigen_classic import main as testk_eigenclassic
    from testing.test_a import main as testa
    from train.train import main as train
    from predict.predict import predict
    from train.retrain_stage1_a import main as retrain1a
    from mean.mean import main as mean

    device = torch.device("cuda" if args.gpu_indices else "cpu")

    run = {
        "predict": predict,
        "testk_eigenclassic": testk_eigenclassic,
        "testk": testk,
        "testa": testa,
        "train": train,
        "retrain1a": retrain1a,
        "mean": mean,
    }[script]

    run(args, device)


if __name__ == "__main__":

    main()
