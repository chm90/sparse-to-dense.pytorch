import argparse
import os
from typing import List, Type, Iterable

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from torch import nn
from itertools import chain
from dataloaders import NYUDataset, RGBDDataset
from metrics import MaskedResult
from models import ResNet
from dataloaders import SquareShape
from utils import add_row, save_image
parser = argparse.ArgumentParser(
    description=
    "Find the images with the greates difference in loss in a dataset between two models give a checkpoint for each model"
)
parser.add_argument("first_model_checkpoint", type=str)
parser.add_argument("second_model_checkpoint", type=str)
parser.add_argument("--first-name", type=str, default="first")
parser.add_argument("--second-name", type=str, default="second")
data_names = ['nyudepthv2', "SUNRGBD"]
parser.add_argument(
    '--data',
    metavar='DATA',
    default='nyudepthv2',
    choices=data_names,
    help='dataset: ' + ' | '.join(data_names) + ' (default: nyudepthv2)')
parser.add_argument(
    "--square-width",
    type=int,
    default=50,
    help="width of square in the middle (default 50)")
parser.add_argument(
    "--out-dir",
    "-od",
    type=str,
    metavar="RUN-NAME",
    default=None,
    help="Name of output directory")
parser.add_argument(
    "--imrows",
    type=int,
    default=10,
    help="Number of rows that are printed to the output images")


def evaluate(
        model: Type[nn.Module],
        dataloader: Type[torch.utils.data.DataLoader]) -> List[MaskedResult]:
    results = []
    for i, (x, y) in enumerate(dataloader):
        # For some reason, mypy doesn't recognize that RGBDDataset is indexable
        x, y = x.cuda(), y.cuda()
        y_hat = model(x)
        result = MaskedResult(dataloader.dataset.mask_inside_square)
        result.evaluate(y_hat, y)
        results.append(result)
    return results


def main() -> None:
    args = parser.parse_args()
    args.data = os.path.join(  #type:ignore
        os.environ["DATASET_DIR"], args.data)
    #Load model
    first_model_cp = torch.load(args.first_model_checkpoint)
    second_model_cp = torch.load(args.second_model_checkpoint)

    first_model: Type[ResNet] = first_model_cp["model"]
    second_model: Type[ResNet] = second_model_cp["model"]

    # load dataset
    valdir = os.path.join(args.data, 'val')
    image_shape = (192, 256)
    val_dataset_rgb = NYUDataset(
        valdir,
        phase='val',
        modality="rgb",
        num_samples=0,
        square_width=args.square_width,
        output_shape=image_shape)
    val_dataloader_rgb = torch.utils.data.DataLoader(val_dataset_rgb)
    val_dataset_rgbd = NYUDataset(
        valdir,
        phase='val',
        modality="rgbd",
        num_samples=0,
        square_width=args.square_width,
        output_shape=image_shape)
    val_dataloader_rgbd = torch.utils.data.DataLoader(val_dataset_rgbd)

    first_model_is_rgbd = first_model.in_channels == 4
    second_model_is_rgbd = second_model.in_channels == 4

    first_results = evaluate(first_model, val_dataloader_rgbd
                             if first_model_is_rgbd else val_dataloader_rgb)
    second_results = evaluate(second_model, val_dataloader_rgbd
                              if second_model_is_rgbd else val_dataloader_rgb)
    sort_metric_first = np.array([(result.result_outside if first_model_is_rgbd
                                   else result.result).absrel
                                  for result in first_results])
    sort_metric_second = np.array([(result.result_outside
                                    if second_model_is_rgbd else
                                    result.result).absrel
                                   for result in second_results])
    ratio = sort_metric_first / sort_metric_second
    acending_ratio_idxs = np.argsort(ratio)
    decending_ratio_idxs = np.flip(acending_ratio_idxs, 0)

    print(f"=> number of images where ratio > 1: {np.sum(ratio > 1)}")
    print(f"=> total number of images: {sort_metric_first.size}")
    print(
        f"=> ration of images where ratio > 1: {np.sum(ratio > 1)/sort_metric_first.size}"
    )
    #plot error distribution
    fig, ax1 = plt.subplots()
    ax1.plot(ratio[acending_ratio_idxs], "y-",zorder=0)
    ax1.set_ylabel(
        f"$\\frac{{absrel_{{{args.first_name}}} }}{{absrel_{{{args.second_name}}} }}$"
    )
    ax2 = ax1.twinx()
    ax2.set_ylabel("absrel")
    ax2.plot(sort_metric_first[acending_ratio_idxs])
    ax2.plot(sort_metric_second[acending_ratio_idxs], "--")
    plt.title("Difference in error")
    ax1.legend([
        f"$\\frac{{absrel_{{{args.first_name}}} }}{{absrel_{{{args.second_name}}} }}$"
    ])
    ax2.legend([
        f"$absrel_{{{args.first_name}}}$",
        f"$absrel_{{{args.second_name}}}$",
    ])
    ax1.set_xlabel("image")

    if args.out_dir == None:
        plt.show()
    else:
        if not os.path.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        plt.savefig(os.path.join(args.out_dir, "difference_in_error.png"))

    #plot images from high difference in error to low

    def draw_images(idxs: Iterable[int]) -> np.ndarray:
        im_merge = None
        for i in idxs:
            plt.figure()
            x_rgbd, y = val_dataset_rgbd[i]
            x_rgb, y = val_dataset_rgb[i]
            x_rgb = torch.unsqueeze(x_rgb, 0)
            x_rgbd = torch.unsqueeze(x_rgbd, 0)
            y = torch.unsqueeze(y, 0)
            x_rgbd, x_rgb, y = x_rgbd.cuda(), x_rgb.cuda(), y.cuda()
            y_hat_first = first_model(x_rgbd  # type: ignore
                                      if first_model_is_rgbd else x_rgb)

            y_hat_second = second_model(x_rgbd  # type: ignore
                                        if second_model_is_rgbd else x_rgb)
            row = merge_into_row(x_rgbd, val_dataset_rgbd.square, y,
                                 y_hat_first, y_hat_second)
            im_merge = row if im_merge is None else add_row(im_merge, row)
        return im_merge

    first_model_best_im = draw_images(acending_ratio_idxs[:args.imrows])
    print("first_model_best_im ratios =",
          ratio[acending_ratio_idxs[:args.imrows]])
    first_model_worst_im = draw_images(decending_ratio_idxs[:args.imrows])
    print("first_model_worst_im ratios =",
          ratio[decending_ratio_idxs[:args.imrows]])
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    save_image(first_model_best_im,
               os.path.join(args.out_dir, f"{args.first_name}_best.png"))
    save_image(first_model_worst_im,
               os.path.join(args.out_dir, f"{args.first_name}_worst.png"))


cmap = plt.cm.jet


def merge_into_row(input: torch.Tensor, square: SquareShape,
                   target: torch.Tensor, first_prediction: torch.Tensor,
                   second_prediction: torch.Tensor) -> torch.Tensor:

    rgb = input[:, :3, :, :]
    rgb = 255 * np.transpose(np.squeeze(rgb.cpu().numpy()), (1, 2, 0))
    d = input[:, 3, :, :]
    depth_ims = [d, target, first_prediction, second_prediction]
    depth_ims = list(
        map(lambda d_im: np.squeeze(d_im.data.cpu().numpy()), depth_ims))
    mean = min(map(lambda d_im: np.min(d_im), depth_ims))
    scale = max(map(lambda d_im: np.max(d_im) - np.min(d_im), depth_ims))

    def process_depth(depth: torch.Tensor) -> torch.Tensor:
        depth = (depth - mean) / scale
        depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
        return depth

    xmin, xmax, ymin, ymax = square

    def paint_square(image: np.ndarray) -> np.ndarray:
        image[xmin:xmax, ymin - 1:ymin + 1, :] = 255
        image[xmin:xmax, ymax - 1:ymax + 1, :] = 255
        image[xmin - 1:xmin + 1, ymin:ymax, :] = 255
        image[xmax - 1:xmax + 1, ymin:ymax, :] = 255
        return image

    depth_ims = list(map(process_depth, depth_ims))
    ims = list(map(paint_square, chain([rgb], depth_ims)))
    img_merge = np.hstack(ims)

    return img_merge


if __name__ == "__main__":
    main()
