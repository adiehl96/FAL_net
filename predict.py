import os

import numpy as np
from PIL import Image

import models

import torch
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.functional as F
from misc.listdataset_run import ListDataset as RunListDataset

from misc import utils, data_transforms


def predict(args, device="cpu"):
    print("-------Predicting on " + str(device) + "-------")

    save_path = os.path.join("predictions", args.dataset, args.model, args.time_stamp)
    if args.f_post_process:
        save_path = save_path + "fpp"
    if args.ms_post_process:
        save_path = save_path + "mspp"
    print("=> Saving to {}".format(save_path))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    utils.display_config(args, save_path)

    input_transform = transforms.Compose(
        [
            data_transforms.ArrayToTensor(),
            transforms.Normalize(
                mean=[0, 0, 0], std=[255, 255, 255]
            ),  # (input - mean) / std
            transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1]),
        ]
    )

    # Torch Data Set List
    input_path = os.path.join(args.data_directory, args.dataset)
    test_dataset = RunListDataset(
        path_list=[args.input],
        transform=input_transform,
    )

    print("len(test_dataset)", len(test_dataset))
    # Torch Data Loader
    args.batch_size = 1  # kitty mixes image sizes!
    args.sparse = True  # disparities are sparse (from lidar)
    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=False,
    )

    # create pan model
    model_path = os.path.join(
        args.dataset + "_stage2",
        args.time_stamp,
    )
    if not os.path.exists(model_path):
        raise Exception(
            f"No pretrained model with timestamp {args.pretrained} was found."
        )
    model_path = os.path.join(
        model_path,
        next(d for d in (next(os.walk(model_path))[1]) if not d[0] == "."),
        "model_best.pth.tar",
    )

    print(model_path)
    pan_network_data = torch.load(model_path, map_location=torch.device(device))

    pan_model = pan_network_data[
        next(item for item in pan_network_data.keys() if "model" in str(item))
    ]

    print("=> using pre-trained model for pan '{}'".format(pan_model))
    pan_model = models.__dict__[pan_model](
        pan_network_data, no_levels=args.no_levels, device=device
    ).to(device)
    pan_model = torch.nn.DataParallel(pan_model).to(device)
    if device.type == "cpu":
        pan_model = pan_model.module.to(device)
    pan_model.eval()
    model_parameters = utils.get_n_params(pan_model)
    print("=> Number of parameters '{}'".format(model_parameters))
    cudnn.benchmark = True

    l_disp_path = os.path.join(save_path, "l_disp")
    if not os.path.exists(l_disp_path):
        os.makedirs(l_disp_path)

    input_path = os.path.join(save_path, "Input im")
    if not os.path.exists(input_path):
        os.makedirs(input_path)

    pan_path = os.path.join(save_path, "Pan")
    if not os.path.exists(pan_path):
        os.makedirs(pan_path)

    pc_path = os.path.join(save_path, "Point_cloud")
    if not os.path.exists(pc_path):
        os.makedirs(pc_path)

    feats_path = os.path.join(save_path, "feats")
    if not os.path.exists(feats_path):
        os.makedirs(feats_path)

    # Set the max disp
    right_shift = args.max_disp * args.rel_baset

    with torch.no_grad():
        print("with torch.no_grad():")
        for i, (input, _, _) in enumerate(val_loader):
            input_left = input[0].to(device)
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

            pred_disp = disp.squeeze(1).cpu().numpy()
            pred_depth = utils.disp_to_depth(pred_disp)
            print("pred_depth", pred_depth[0].shape)
            denormalize = np.array([0.411, 0.432, 0.45])
            denormalize = denormalize[:, np.newaxis, np.newaxis]
            im = pred_depth[0] + denormalize
            im = Image.fromarray(np.rint(255 * im.transpose(1, 2, 0)).astype(np.uint8))
            im.save(os.path.join(pan_path, "{:010d}.png".format(i)))


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
