import os, pickle

import numpy as np

import torch
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.functional as F
from misc.listdataset_run import ListDataset as RunListDataset
import matplotlib.pyplot as plt

from models.FAL_netB import FAL_netB
from misc import utils, data_transforms


def predict(parser, device="cpu"):
    print("-------Predicting on " + str(device) + "-------")

    parser.add_argument(
        "-sp", "--save_path", help="Path that outputs will be saved to", required=True
    )

    parser.add_argument(
        "-mp", "--model_path", help="Path that model will be loaded from", required=True
    )

    parser.add_argument(
        "--input",
        dest="input",
        default="./data/test.png",
        help="path to the input image to be depth predicted",
        required=True,
    )

    parser.add_argument(
        "-pickle", "--pickle_predictions", action="store_true", default=False
    )

    args = parser.parse_args()

    print("=> Saving to {}".format(args.save_path))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    utils.display_config(args, args.save_path)

    input_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1]),
        ]
    )

    output_transform = transforms.Compose(
        [
            data_transforms.NormalizeInverse(mean=[0.411, 0.432, 0.45], std=[1, 1, 1]),
            transforms.ToPILImage(),
        ]
    )

    # Torch Data Set List
    predict_dataset = RunListDataset(
        path_list=[args.input]
        if os.path.isfile(args.input)
        else [
            os.path.join(args.input, x) for x in sorted(next(os.walk(args.input))[2])
        ],
        transform=input_transform,
    )

    print("len(predict_dataset)", len(predict_dataset))
    # Torch Data Loader
    val_loader = torch.utils.data.DataLoader(
        predict_dataset,
        batch_size=1,  # kitty mixes image sizes!
        num_workers=args.workers,
        pin_memory=False,
        shuffle=False,
    )

    print(args.model_path)

    pan_model = FAL_netB(no_levels=args.no_levels, device=device)
    checkpoint = torch.load(args.model_path, map_location=device)
    pan_model.load_state_dict(checkpoint["model_state_dict"])
    if device.type == "cuda":
        pan_model = torch.nn.DataParallel(pan_model).to(device)

    model_parameters = utils.get_n_params(pan_model)
    print("=> Number of parameters '{}'".format(model_parameters))
    cudnn.benchmark = True

    l_disp_path = os.path.join(args.save_path, "l_disp")
    if not os.path.exists(l_disp_path):
        os.makedirs(l_disp_path)

    input_path = os.path.join(args.save_path, "input")
    if args.save_input and not os.path.exists(input_path):
        os.makedirs(input_path, exist_ok=True)

    pickle_path = os.path.join(args.save_path, "pickle")
    if args.pickle_predictions and not os.path.exists(pickle_path):
        os.makedirs(pickle_path, exist_ok=True)
    if args.pickle_predictions:
        predicte_disparities = []

    # Set the max disp
    right_shift = args.max_disp * args.rel_baset

    with torch.no_grad():
        for i, input in enumerate(val_loader):
            input_left = input.to(device)
            B, C, H, W = input_left.shape

            # Prepare flip grid for post-processing
            i_tetha = torch.zeros(B, 2, 3).to(device)
            i_tetha[:, 0, 0] = 1
            i_tetha[:, 1, 1] = 1
            flip_grid = F.affine_grid(i_tetha, [B, C, H, W], align_corners=False)
            flip_grid[:, :, :, 0] = -flip_grid[:, :, :, 0]

            # Convert min and max disp to bx1x1 tensors
            max_disp = (
                torch.Tensor([right_shift])
                .unsqueeze(1)
                .unsqueeze(1)
                .type(input_left.type())
            )
            min_disp = max_disp * args.min_disp / args.max_disp

            disp = pan_model(
                input_left=input_left,
                min_disp=min_disp,
                max_disp=max_disp,
                ret_disp=True,
                ret_subocc=False,
                ret_pan=False,
            )

            if args.ms_post_process:
                disp = ms_pp(input_left, pan_model, flip_grid, disp, min_disp, max_disp)

            disparity = disp.squeeze().cpu().numpy()
            disparity = 256 * np.clip(
                disparity / (np.percentile(disparity, 99) + 1e-9), 0, 1
            )
            plt.imsave(
                os.path.join(l_disp_path, "{:010d}.png".format(i)),
                np.rint(disparity).astype(np.int32),
                cmap="inferno",
                vmin=0,
                vmax=256,
            )

            if args.save_input:
                print("save the input image in path", input_path)
                input_image = output_transform(input[0])
                input_image.save(os.path.join(input_path, "{:010d}.png".format(i)))

            if args.pickle_predictions:
                predicte_disparities.append(disp.squeeze().cpu().numpy())

        if args.pickle_predictions:
            pickle.dump(
                predicte_disparities,
                open(os.path.join(pickle_path, "predictions.pickle"), "wb"),
            )


def ms_pp(input_view, pan_model, flip_grid, disp, min_disp, max_pix):
    _, _, H, W = input_view.shape

    up_fac = 2 / 3
    upscaled = F.interpolate(
        F.grid_sample(input_view, flip_grid, align_corners=False),
        scale_factor=up_fac,
        mode="bilinear",
        align_corners=True,
        recompute_scale_factor=True,
    )
    dwn_flip_disp = pan_model(
        input_left=upscaled,
        min_disp=min_disp,
        max_disp=max_pix,
        ret_disp=True,
        ret_pan=False,
        ret_subocc=False,
    )
    dwn_flip_disp = (1 / up_fac) * F.interpolate(
        dwn_flip_disp, size=(H, W), mode="nearest"
    )  # , align_corners=True)
    dwn_flip_disp = F.grid_sample(dwn_flip_disp, flip_grid, align_corners=False)

    norm = disp / (np.percentile(disp.detach().cpu().numpy(), 95) + 1e-6)
    norm[norm > 1] = 1

    return (1 - norm) * disp + norm * dwn_flip_disp


def local_normalization(img, win=3):
    B, C, _, _ = img.shape
    mean = [0.411, 0.432, 0.45]
    m_rgb = torch.ones((B, C, 1, 1)).type(img.type())
    m_rgb[:, 0, :, :] = mean[0] * m_rgb[:, 0, :, :]
    m_rgb[:, 1, :, :] = mean[1] * m_rgb[:, 1, :, :]
    m_rgb[:, 2, :, :] = mean[2] * m_rgb[:, 2, :, :]

    img = img + m_rgb
    img = img.cpu()

    # Get mean and normalize
    win_mean_T = F.avg_pool2d(
        img, kernel_size=win, stride=1, padding=(win - 1) // 2
    )  # B,C,H,W
    win_std = F.avg_pool2d(
        (img - win_mean_T) ** 2, kernel_size=win, stride=1, padding=(win - 1) // 2
    ) ** (1 / 2)
    win_norm_img = (img - win_mean_T) / (win_std + 0.0000001)

    return win_norm_img
