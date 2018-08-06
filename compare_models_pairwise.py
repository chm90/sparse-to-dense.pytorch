import argparse
import os
from typing import Iterable, List, Type

import matplotlib
import numpy as np
import torch
from torch import nn

from dataloaders import NYUDataset
from main import parser as model_parser
from metrics import MaskedResult
from models import ResNet
from utils import add_row, get_output_dir, merge_ims_into_row, save_image

matplotlib.use("Agg")
from matplotlib import pyplot as plt
parser = argparse.ArgumentParser(
    add_help=False,
    description=
    "Find the images with the greates difference in loss in a dataset between two models give a checkpoint for each model",
    parents=[model_parser])
parser.add_argument("--depth-name", type=str, default="depth")
parser.add_argument("--rgb-name", type=str, default="rgb")
parser.add_argument(
    "--imrows",
    type=int,
    default=10,
    help="Number of rows that are printed to the output images")

data_names = ['nyudepthv2', "SUNRGBD"]

rgbd_model_locations = {
    "no-skip":
    (r"results__newer/nyudepthv2.modality=rgb.arch=resnet50.skip=none.decoder=deconv3.criterion=l1."
     r"lr=0.01.bs=16.opt=sgd.depth-type=square.square-width=50/model_best.pth.tar"
     ),
}
parser.add_argument(
    "--rgb-model-kind",
    choices=rgbd_model_locations.keys(),
    type=str,
    help=
    "The kind of rgb model to use for training. The locations of these rgb models are defined in the script file."
)


def evaluate(model: Type[nn.Module],
             dataloader: torch.utils.data.DataLoader) -> List[MaskedResult]:
    results = []
    for i, (x, y) in enumerate(dataloader):
        x, y = x.cuda(), y.cuda()
        y_hat = model(x)
        result = MaskedResult(dataloader.dataset.mask_inside_square)
        result.evaluate(y_hat, y)
        results.append(result)
    return results


def main() -> None:
    args = parser.parse_args()
    #Load model
    depth_model_dir = get_output_dir(args)
    print("args.output_dir =", args.output_dir)
    print("depth_model_dir =", depth_model_dir)
    depth_model_cp_fn = os.path.join(depth_model_dir, "model_best.pth.tar")
    depth_model_cp = torch.load(depth_model_cp_fn)
    rgb_model_cp = torch.load(rgbd_model_locations[args.rgb_model_kind])
    args.data = os.path.join(  #type:ignore
        os.environ["DATASET_DIR"], args.data)

    depth_model: Type[ResNet] = depth_model_cp["model"]
    rgb_model: Type[ResNet] = rgb_model_cp["model"]

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
        output_shape=image_shape,
        depth_type=args.depth_type)
    val_dataloader_rgbd = torch.utils.data.DataLoader(val_dataset_rgbd)

    depth_model_is_rgbd = depth_model.in_channels == 4
    rgb_model_is_rgbd = rgb_model.in_channels == 4

    depth_results = evaluate(depth_model, val_dataloader_rgbd
                             if depth_model_is_rgbd else val_dataloader_rgb)
    rgb_results = evaluate(rgb_model, val_dataloader_rgbd
                           if rgb_model_is_rgbd else val_dataloader_rgb)
    sort_metric_depth = np.array([(result.result_outside if depth_model_is_rgbd
                                   else result.result).absrel
                                  for result in depth_results])
    sort_metric_rgb = np.array([(result.result_outside if rgb_model_is_rgbd
                                 else result.result).absrel
                                for result in rgb_results])
    ratio = sort_metric_depth / sort_metric_rgb
    acending_ratio_idxs = np.argsort(ratio)
    decending_ratio_idxs = np.flip(acending_ratio_idxs, 0)

    print(f"=> number of images where ratio > 1: {np.sum(ratio > 1)}")
    print(f"=> total number of images: {sort_metric_depth.size}")
    print(
        f"=> ration of images where ratio > 1: {np.sum(ratio > 1)/sort_metric_depth.size}"
    )
    #plot error distribution
    fig, ax1 = plt.subplots()
    ax1.plot(ratio[acending_ratio_idxs], "y-", zorder=0)
    ax1.set_ylabel(
        f"$\\frac{{absrel_{{{args.depth_name}}} }}{{absrel_{{{args.rgb_name}}} }}$"
    )
    ax2 = ax1.twinx()
    ax2.set_ylabel("absrel")
    ax2.plot(sort_metric_depth[acending_ratio_idxs])
    ax2.plot(sort_metric_rgb[acending_ratio_idxs], "--")
    plt.title("Difference in error")
    ax1.legend([
        f"$\\frac{{absrel_{{{args.depth_name}}} }}{{absrel_{{{args.rgb_name}}} }}$"
    ])
    ax2.legend([
        f"$absrel_{{{args.depth_name}}}$",
        f"$absrel_{{{args.rgb_name}}}$",
    ])
    ax1.set_xlabel("image")

    plt.savefig(os.path.join(depth_model_dir, "difference_in_error.png"))

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
            y_hat_depth = depth_model(x_rgbd  # type: ignore
                                      if depth_model_is_rgbd else x_rgb)

            y_hat_rgb = rgb_model(x_rgbd  # type: ignore
                                  if rgb_model_is_rgbd else x_rgb)
            images = [x_rgbd,y,y_hat_depth,y_hat_rgb]
            squares = None if "single-pixel" in args.depth_type else [val_dataset_rgbd.square] * len(images)
            row = merge_ims_into_row(images,square=squares)
            im_merge = row if im_merge is None else add_row(im_merge, row)
        return im_merge

    depth_model_best_im = draw_images(acending_ratio_idxs[:args.imrows])
    print("depth_model_best_im ratios =",
          ratio[acending_ratio_idxs[:args.imrows]])
    depth_model_worst_im = draw_images(decending_ratio_idxs[:args.imrows])
    print("depth_model_worst_im ratios =",
          ratio[decending_ratio_idxs[:args.imrows]])

    save_image(depth_model_best_im,
               os.path.join(depth_model_dir, f"{args.depth_name}_best.png"))
    save_image(depth_model_worst_im,
               os.path.join(depth_model_dir, f"{args.depth_name}_worst.png"))


cmap = plt.cm.jet

#def merge_into_row(input: torch.Tensor, square: SquareShape,
#                   target: torch.Tensor, depth_prediction: torch.Tensor,
#                   rgb_prediction: torch.Tensor) -> torch.Tensor:
#
#    rgb = input[:, :3, :, :]
#    rgb = 255 * np.transpose(np.squeeze(rgb.cpu().numpy()), (1, 2, 0))
#    d = input[:, 3, :, :]
#    depth_ims = [d, target, depth_prediction, rgb_prediction]
#    depth_ims = list(
#        map(lambda d_im: np.squeeze(d_im.data.cpu().numpy()), depth_ims))
#    mean = min(map(lambda d_im: np.min(d_im), depth_ims))
#    scale = max(map(lambda d_im: np.max(d_im) - np.min(d_im), depth_ims))
#
#    def process_depth(depth: torch.Tensor) -> torch.Tensor:
#        depth = (depth - mean) / scale
#        depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
#        return depth
#
#    xmin, xmax, ymin, ymax = square
#
#    def paint_square(image: np.ndarray) -> np.ndarray:
#        image[xmin:xmax, ymin - 1:ymin + 1, :] = 255
#        image[xmin:xmax, ymax - 1:ymax + 1, :] = 255
#        image[xmin - 1:xmin + 1, ymin:ymax, :] = 255
#        image[xmax - 1:xmax + 1, ymin:ymax, :] = 255
#        return image
#
#    depth_ims = list(map(process_depth, depth_ims))
#    ims = list(map(paint_square, chain([rgb], depth_ims)))
#    img_merge = np.hstack(ims)
#
#    return img_merge

if __name__ == "__main__":
    main()
