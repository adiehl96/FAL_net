# Train_Stage2_K.py Fine tune model with MOM on KITTI only
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

import os, time

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter
import numpy as np

from models.FAL_netB import FAL_netB
from misc.data_utils import load_data, ApplyToMultiple, NormalizeInverse
from misc import utils
from misc.loss_functions import VGGLoss, realEPE, smoothness


def main(args, device="cpu"):
    print(f"-------Training Stage {args.stage} on " + str(device) + "-------")
    best_rmse = None

    validate = {
        "ASM_stereo_small": validate_asm,
        "ASM_stereo": validate_asm,
        "KITTI": validate_kitti,
    }[args.dataset]

    # Set output writters for showing up progress on tensorboardX
    train_writer = SummaryWriter(os.path.join(args.save_path, "train"))
    test_writer = SummaryWriter(os.path.join(args.save_path, "test"))
    output_writers = [
        SummaryWriter(os.path.join(args.save_path, "test", str(i))) for i in range(3)
    ]

    # Set up data augmentations
    transform = ApplyToMultiple(
        transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=(args.crop_height, args.crop_width),
                    scale=(0.10, 1.0),
                    ratio=(1, 1),
                ),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.3606, 0.3789, 0.3652], std=[0.3123, 0.3173, 0.3216]
                )
                if args.dataset == "KITTI"
                else transforms.Normalize(
                    mean=[0.3606, 0.3789, 0.3652], std=[0.3123, 0.3173, 0.3216]
                ),
            ]
        ),
        RandomHorizontalFlipChance=0.5,
    )

    val_transform = ApplyToMultiple(
        transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1]),
            ]
        )
    )

    output_transforms = ApplyToMultiple(
        transforms.Compose(
            [
                NormalizeInverse(mean=[0.411, 0.432, 0.45], std=[1, 1, 1]),
                transforms.ToPILImage(),
            ]
        )
    )

    if args.dataset == "KITTI":
        train_dataset = load_data(
            split=args.train_split,
            dataset=args.dataset,
            root=args.data_directory,
            transform=transform,
        )

        val_dataset = load_data(
            split=args.validation_split,
            dataset=args.validation_dataset,
            root=args.data_directory,
            transform=val_transform,
        )
    elif "ASM" in args.dataset:
        train_dataset, val_dataset = load_data(
            dataset=args.dataset,
            root=args.data_directory,
            split=args.train_split,
            transform=transform,
            val_transform=val_transform,
            create_val=0.1,
        )
    print("len(train_dataset)", len(train_dataset))
    print("len(val_dataset)", len(val_dataset))

    # Torch Data Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=False,
    )

    print("len(train_loader)", len(train_loader))
    print("len(val_loader)", len(val_loader))

    # create model
    model = FAL_netB(no_levels=args.no_levels, device=device).to(device)
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    if device.type == "cuda":
        print("torch.nn.DataParallel(model).to(device)")
        model = torch.nn.DataParallel(model).to(device)
    print("=> Number of parameters m-model '{}'".format(utils.get_n_params(model)))

    # create fix model (needed for stage 2)
    fix_model = None
    if args.stage == 2:
        fix_model = FAL_netB(no_levels=args.no_levels, device=device)
        checkpoint = torch.load(args.fix_model, map_location=device)
        fix_model.load_state_dict(checkpoint["model_state_dict"])
        if device.type == "cuda":
            fix_model = torch.nn.DataParallel(fix_model).to(device)

        print(
            "=> Number of parameters m-model '{}'".format(utils.get_n_params(fix_model))
        )
        fix_model.eval()

    # Optimizer Settings
    print("Setting {} Optimizer".format(args.optimizer))
    param_groups = [
        {
            "params": model.module.bias_parameters()
            if isinstance(model, torch.nn.DataParallel)
            else model.bias_parameters(),
            "weight_decay": args.bias_decay,
        },
        {
            "params": model.module.weight_parameters()
            if isinstance(model, torch.nn.DataParallel)
            else model.weight_parameters(),
            "weight_decay": args.weight_decay,
        },
    ]
    optimizer = torch.optim.Adam(
        params=param_groups, lr=args.lr, betas=(args.momentum, args.beta)
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=0.5
    )

    vgg_loss = VGGLoss(device=device)
    scaler = GradScaler()

    for epoch in range(args.epochs):
        # train for one epoch
        loss, train_loss = train(
            args,
            train_loader,
            model,
            fix_model,
            optimizer,
            epoch,
            device,
            vgg_loss,
            scaler,
        )
        train_writer.add_scalar("train_loss", train_loss, epoch)

        # evaluate on validation set, RMSE is from stereoscopic view synthesis task
        rmse = validate(
            args, val_loader, model, epoch, output_writers, output_transforms, device
        )
        test_writer.add_scalar("mean RMSE", rmse, epoch)

        # Apply LR schedule (after optimizer.step() has been called for recent pyTorch versions)
        scheduler.step()

        if best_rmse is None:
            best_rmse = rmse
        best_rmse = min(rmse, best_rmse)
        utils.save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.module.state_dict()
                if isinstance(model, torch.nn.DataParallel)
                else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            rmse == best_rmse,
            args.save_path,
        )


def train(
    args, train_loader, model, fix_model, optimizer, epoch, device, vgg_loss, scaler
):
    epoch_size = (
        len(train_loader)
        if args.epoch_size == 0
        else min(len(train_loader), args.epoch_size)
    )

    batch_time = utils.RunningAverageMeter()
    data_time = utils.AverageMeter()
    rec_losses = utils.AverageMeter()
    losses = utils.AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input_data, _, _) in enumerate(train_loader):
        # Read training data
        left_view = input_data[0].to(device)
        right_view = input_data[1].to(device)
        max_disp = (
            torch.Tensor([args.max_disp * args.relative_baseline])
            .repeat(args.batch_size)
            .unsqueeze(1)
            .unsqueeze(1)
            .type(left_view.type())
        )
        min_disp = max_disp * args.min_disp / args.max_disp
        B, C, H, W = left_view.shape

        # measure data loading time
        data_time.update(time.time() - end)

        # Reset gradients
        optimizer.zero_grad()

        # Flip Grid (differentiable)
        if args.stage == 2:
            i_tetha = torch.autograd.Variable(torch.zeros(B, 2, 3)).to(device)
            i_tetha[:, 0, 0] = 1
            i_tetha[:, 1, 1] = 1
            i_grid = F.affine_grid(i_tetha, [B, C, H, W], align_corners=True)
            flip_grid = i_grid.clone()
            flip_grid[:, :, :, 0] = -flip_grid[:, :, :, 0]

        # Get mirrored disparity from fixed falnet model
        if args.a_mr > 0:
            with torch.no_grad():
                disp = fix_model(
                    torch.cat(
                        (
                            F.grid_sample(left_view, flip_grid, align_corners=True),
                            right_view,
                        ),
                        0,
                    ),
                    torch.cat((min_disp, min_disp), 0),
                    torch.cat((max_disp, max_disp), 0),
                    ret_disp=True,
                    ret_pan=False,
                    ret_subocc=False,
                )
                mldisp = F.grid_sample(
                    disp[0:B, :, :, :], flip_grid, align_corners=True
                ).detach()
                mrdisp = disp[B::, :, :, :].detach()

        with autocast():
            ###### LEFT disp
            if args.stage == 1:
                rpan, ldisp = model(
                    input_left=left_view,
                    min_disp=min_disp,
                    max_disp=max_disp,
                    ret_disp=True,
                    ret_pan=True,
                    ret_subocc=False,
                )
            if args.stage == 2:
                pan, disp, mask0, mask1 = model(
                    input_left=torch.cat(
                        (
                            left_view,
                            F.grid_sample(right_view, flip_grid, align_corners=True),
                        ),
                        0,
                    ),
                    min_disp=torch.cat((min_disp, min_disp), 0),
                    max_disp=torch.cat((max_disp, max_disp), 0),
                    ret_disp=True,
                    ret_pan=True,
                    ret_subocc=True,
                )
                rpan = pan[0:B, :, :, :]
                lpan = pan[B::, :, :, :]
                ldisp = disp[0:B, :, :, :]
                rdisp = disp[B::, :, :, :]

                lmask = mask0[0:B, :, :, :]
                rmask = mask0[B::, :, :, :]
                rlmask = mask1[0:B, :, :, :]
                lrmask = mask1[B::, :, :, :]

                # Unflip right view stuff
                lpan = F.grid_sample(lpan, flip_grid, align_corners=True)
                rdisp = F.grid_sample(rdisp, flip_grid, align_corners=True)
                rmask = F.grid_sample(rmask, flip_grid, align_corners=True)
                lrmask = F.grid_sample(lrmask, flip_grid, align_corners=True)

            # Compute rec loss
            if args.a_p > 0:
                vgg_right = vgg_loss.vgg(right_view)
                if args.stage == 2:
                    vgg_left = vgg_loss.vgg(left_view)
            else:
                vgg_right = None
                vgg_left = None

            if args.stage == 2:
                # Obtain final occlusion masks
                O_L = lmask * lrmask
                O_L[:, :, :, 0 : int(0.20 * W)] = 1
                O_R = rmask * rlmask
                O_R[:, :, :, int(0.80 * W) : :] = 1
                if args.a_mr == 0:  # no mirror loss, then it is just more training
                    O_L = 1
                    O_R = 1

            # Over 2 as measured twice for left and right
            if args.stage == 1:
                mask = 1
                rec_loss = vgg_loss.rec_loss_fnc(
                    mask, rpan, right_view, vgg_right, args.a_p
                )
            elif args.stage == 2:
                rec_loss = (
                    vgg_loss.rec_loss_fnc(O_R, rpan, right_view, vgg_right, args.a_p)
                    + vgg_loss.rec_loss_fnc(O_L, lpan, left_view, vgg_left, args.a_p)
                ) / 2
            rec_losses.update(rec_loss.detach().cpu(), args.batch_size)

            #  Compute smooth loss
            sm_loss = 0
            if args.smooth > 0:
                if args.stage == 1:
                    # Here we ignore the 20% left dis-occluded region, as there is no suppervision for it due to parralax
                    sm_loss = smoothness(
                        left_view[:, :, :, int(0.20 * W) : :],
                        ldisp[:, :, :, int(0.20 * W) : :],
                        gamma=2,
                        device=device,
                    )
                elif args.stage == 2:
                    sm_loss = 0
                    if args.smooth > 0:
                        # Here we ignore the 20% left dis-occluded region, as there is no suppervision for it due to parralax
                        sm_loss = (
                            smoothness(
                                left_view[:, :, :, int(0.20 * W) : :],
                                ldisp[:, :, :, int(0.20 * W) : :],
                                gamma=2,
                                device=device,
                            )
                            + smoothness(
                                right_view[:, :, :, 0 : int(0.80 * W)],
                                rdisp[:, :, :, 0 : int(0.80 * W)],
                                gamma=2,
                                device=device,
                            )
                        ) / 2

            # Compute mirror loss
            mirror_loss = 0
            if args.a_mr > 0:
                # Normalize error ~ between 0-1, by diving over the max disparity value
                nmaxl = 1 / F.max_pool2d(mldisp, kernel_size=(H, W))
                nmaxr = 1 / F.max_pool2d(mrdisp, kernel_size=(H, W))
                mirror_loss = (
                    torch.mean(
                        nmaxl
                        * (1 - O_L)[:, :, :, int(0.20 * W) : :]
                        * torch.abs(ldisp - mldisp)[:, :, :, int(0.20 * W) : :]
                    )
                    + torch.mean(
                        nmaxr
                        * (1 - O_R)[:, :, :, 0 : int(0.80 * W)]
                        * torch.abs(rdisp - mrdisp)[:, :, :, 0 : int(0.80 * W)]
                    )
                ) / 2

            # compute gradient and do optimization step
            loss = rec_loss + args.smooth * sm_loss + args.a_mr * mirror_loss
            losses.update(loss.detach().cpu(), args.batch_size)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i == epoch_size - 1 or i % args.print_freq == 0 and not i == 0:
            eta = utils.eta_calculator(
                batch_time.get_avg(), epoch_size, args.epochs - epoch, i
            )
            print(
                f"Epoch: [{epoch}][{i}/{epoch_size}] ETA {eta} Batch Time {batch_time}  Loss {losses} RecLoss {rec_losses}"
            )

        # End training epoch earlier if args.epoch_size != 0
        if i >= epoch_size:
            break

    return loss, losses.avg


def validate_kitti(
    args, val_loader, model, epoch, output_writers, output_transforms, device
):
    test_time = utils.AverageMeter()
    RMSES = utils.AverageMeter()
    EPEs = utils.AverageMeter()
    kitti_erros = utils.multiAverageMeter(utils.kitti_error_names)

    # switch to evaluate mode
    model.eval()

    # Disable gradients to save memory
    with torch.no_grad():
        for i, input_data in enumerate(val_loader):
            input_left = input_data[0][0].to(device)
            input_right = input_data[0][1].to(device)
            target = input_data[1][0].to(device)
            max_disp = (
                torch.Tensor([args.max_disp * args.relative_baseline])
                .unsqueeze(1)
                .unsqueeze(1)
                .type(input_left.type())
            )

            # Prepare input data
            end = time.time()
            min_disp = max_disp * args.min_disp / args.max_disp
            p_im, disp, maskL, maskRL = model(
                input_left=input_left,
                min_disp=min_disp,
                max_disp=max_disp,
                ret_disp=True,
                ret_pan=True,
                ret_subocc=True,
            )
            test_time.update(time.time() - end)

            # Measure RMSE
            rmse = utils.get_rmse(p_im, input_right, device=device)
            RMSES.update(rmse)

            # record EPE
            flow2_EPE = realEPE(disp, target, sparse=True)
            EPEs.update(flow2_EPE.detach(), target.size(0))

            # Record kitti metrics
            target_depth, pred_depth = utils.disps_to_depths_kitti2015(
                target.detach().squeeze(1).cpu().numpy(),
                disp.detach().squeeze(1).cpu().numpy(),
            )
            kitti_erros.update(
                utils.compute_kitti_errors(target_depth[0], pred_depth[0]),
                target.size(0),
            )

            denormalize = np.array([0.411, 0.432, 0.45])
            denormalize = denormalize[:, np.newaxis, np.newaxis]

            if i < len(output_writers):  # log first output of first batches
                if epoch == 0:
                    output_writers[i].add_image(
                        "Input left", input_left[0].cpu().numpy() + denormalize, 0
                    )

                # Plot disp
                output_writers[i].add_image(
                    "Left disparity", utils.disp2rgb(disp[0].cpu().numpy(), None), epoch
                )

                # Plot left subocclsion mask
                output_writers[i].add_image(
                    "Left sub-occ", utils.disp2rgb(maskL[0].cpu().numpy(), None), epoch
                )

                # Plot right-from-left subocclsion mask
                output_writers[i].add_image(
                    "RightL sub-occ",
                    utils.disp2rgb(maskRL[0].cpu().numpy(), None),
                    epoch,
                )

                # Plot synthetic right (or panned) view output
                p_im = p_im[0].detach().cpu().numpy() + denormalize
                p_im[p_im > 1] = 1
                p_im[p_im < 0] = 0
                output_writers[i].add_image("Output Pan", p_im, epoch)

            if i % args.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t Time {2}\t RMSE {3}".format(
                        i, len(val_loader), test_time, RMSES
                    )
                )

    print("* RMSE {0}".format(RMSES.avg))
    print(" * EPE {:.3f}".format(EPEs.avg))
    print(kitti_erros)
    return RMSES.avg


def validate_asm(
    args, val_loader, model, epoch, output_writers, output_transforms, device
):
    batch_time = utils.AverageMeter()
    asm_erros = utils.multiAverageMeter(utils.image_similarity_measures)

    with torch.no_grad():
        print("with torch.no_grad():")
        for i, input_data in enumerate(val_loader):
            input_left = input_data[0][0].to(device)
            input_right = input_data[0][1].to(device)

            # Convert min and max disp to bx1x1 tensors
            max_disp = (
                torch.Tensor([args.max_disp * args.relative_baseline])
                .unsqueeze(1)
                .unsqueeze(1)
                .type(input_left.type())
            )
            min_disp = max_disp * args.min_disp / args.max_disp

            # Synthesis
            end = time.time()

            [pan_im] = model(
                input_left=input_left,
                min_disp=min_disp,
                max_disp=max_disp,
                ret_disp=False,
                ret_subocc=False,
                ret_pan=True,
            )

            # measure elapsed time
            batch_time.update(time.time() - end, 1)

            for target_im, pred_im in zip(input_right, pan_im):
                [target_im, pred_im] = output_transforms([target_im, pred_im])
                errors = utils.compute_asm_errors(target_im, pred_im)
                asm_erros.update(errors)
    print(asm_erros)
    return asm_erros.avg[3]
