import os, sys, itertools

import torch
import torchvision.transforms as transforms

from misc.data_utils import load_data, ApplyToMultiple


def get_mean_and_std(dataloader, length):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for idx, (inputs, _, _) in enumerate(dataloader):
        print(f"Processed batch {idx} of {length}.")
        for inp in inputs:
            # Mean over batch, height and width, but not over the channels
            channels_sum += torch.mean(inp, dim=[0, 2, 3])
            channels_squared_sum += torch.mean(inp ** 2, dim=[0, 2, 3])

        num_batches += len(inputs)

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def main(args, _):
    print(" ".join(sys.argv[:]))

    bunches = {
        "ASM_stereo": ["ASM_stereo_train", "ASM_stereo_test"],
        "KITTI": ["eigen_test_improved", "eigen_test_classic", "eigen_train"],
        "KITTI2015": ["bello_val"],
    }[args.dataset]

    chained_dataset = []
    for bunch in bunches:
        dataset = load_data(
            dataset=args.dataset,
            split=bunch,
            root=args.data_directory,
            transform=ApplyToMultiple(transforms.ToTensor()),
        )

        chained_dataset.append(
            torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                num_workers=4,
                pin_memory=False,
                shuffle=False,
            )
        )
    length = sum(len(ds) for ds in chained_dataset)
    chained_dataset = itertools.chain(*chained_dataset)

    print("mean & std: ", get_mean_and_std(chained_dataset, length))
