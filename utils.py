import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
cmap = plt.cm.jet
import torch
from itertools import chain,repeat
from typing import List, Iterable, Callable, Union, Optional
from dataloaders import SquareShape

def process_depth(depth: torch.Tensor, depth_bias: float,
                  scale: float) -> np.ndarray:
    depth = np.squeeze(depth.data.cpu().numpy())
    depth = (depth - depth_bias) / scale
    depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
    return depth


def process_rgb(rgb: torch.Tensor) -> np.ndarray:
    return 255 * np.transpose(np.squeeze(rgb.cpu().numpy()), (1, 2, 0))


def paint_square(image: np.ndarray, square : SquareShape) -> np.ndarray:
    xmin, xmax, ymin, ymax = square
    image[xmin:xmax, ymin - 1:ymin + 1, :] = 255
    image[xmin:xmax, ymax - 1:ymax + 1, :] = 255
    image[xmin - 1:xmin + 1, ymin:ymax, :] = 255
    image[xmax - 1:xmax + 1, ymin:ymax, :] = 255
    return image


def merge_into_row(input, target, depth_pred):
    rgb = input[:, :3, :, :]  #if Tensornput.shape[1] == 4 else input
    rgb = 255 * np.transpose(np.squeeze(rgb.cpu().numpy()), (1, 2, 0))
    depth = np.squeeze(target.cpu().numpy())
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
    pred = np.squeeze(depth_pred.data.cpu().numpy())
    pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
    pred = 255 * cmap(pred)[:, :, :3]  # H, W, C
    ims = [rgb, depth, pred]
    if input.shape[1] == 4:
        d = np.squeeze(input[:, 3, :, :].cpu().numpy())
        d = np.ma.masked_where(d == 0, d)
        d = (d - np.min(d[~d.mask])) / (
            np.max(d[~d.mask]) - np.min(d[~d.mask]))
        d = 255 * cmap(d)[:, :, :3]
        ims.insert(1, d)
    img_merge = np.hstack(ims)
    return img_merge


def image_type(image):
    if image.shape[1] == 4:
        return "rgbd"
    elif image.shape[1] == 3:
        return "rgb"
    else:
        return "d"


def merge_ims_into_row(images: List[torch.cuda.FloatTensor],
                       rgbd_action: str = "both",
                       square:Optional[Union[Optional[SquareShape],List[Optional[SquareShape]]]] = None) -> np.ndarray:
    for image in images:
        assert image.dim(
        ) == 4, f"invalid number of dimensions, expected 4, was {image.dim()} "

    rgbd_selector: Callable[[torch.cuda.FloatTensor], Iterable[
        torch.cuda.FloatTensor]]
    if rgbd_action == "both":
        rgbd_selector = lambda rgbd: [rgbd[:, :3, :, :], rgbd[:, 3, :, :]]
    elif rgbd_action == "depth_only":
        rgbd_selector = lambda rgbd: [rgbd[:, 3, :, :]]
    elif rgbd_action == "rgb_only":
        rgbd_selector = lambda rgbd: [rgbd[:, :3, :, :]]
    else:
        raise ValueError(f"invalid rgbd_action {rgbd_action}")
    images_np_iter = chain.from_iterable(
        rgbd_selector(image) if image_type(image) == "rgbd" else [image]
        for image in images)
    images_np = list(images_np_iter)
    depth_images = list(filter(lambda im: image_type(im) == "d", images_np))
    depth_bias = min(map(lambda d_im: float(d_im.min()), depth_images))
    depth_scale = max(
        map(lambda d_im: float(d_im.max() - d_im.min()), depth_images))

    processed_images = list(map(
        lambda im: process_depth(im, depth_bias, depth_scale) if image_type(im) == "d" else process_rgb(im),
        images_np))

    if square is not None:
        squares : Iterable[SquareShape]
        if isinstance(square,List):
            squares = square
        else:
            squares = repeat(square)
        for sq,processed_image in zip(squares,processed_images):
            if sq is not None:
                paint_square(processed_image,sq)

    im_row = np.hstack(tuple(processed_images))
    return im_row


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)


def get_output_dir(args):
    return os.path.join(
        args.output_dir,
        f'{args.data}.modality={args.modality}.arch={args.arch}'
        f'.skip={args.skip_type}.decoder={args.decoder}'
        f'.criterion={args.criterion}.lr={args.lr}.bs={args.batch_size}'
        f'.opt={args.optimizer}.depth-type={args.depth_type}'
        f'.square-width={args.square_width}')