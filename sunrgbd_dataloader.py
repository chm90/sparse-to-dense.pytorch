import os
import sys
from functools import partial
from math import sqrt
from os import listdir
from os.path import isfile, join, splitext
from typing import Callable, Tuple, Union
from random import uniform
import numpy as np
import torch
from numpy.random import randint
from PIL import Image
from skimage.io import imread
from torch import cat
from torch.utils.data import Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms import transforms as trans
from torchvision.transforms import functional as t

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

def get_im_filename(folder):
    im_paths = (
        join(folder, f) for f in listdir(folder)
        if isfile(join(folder, f)) and splitext(f)[1] in IMG_EXTENSIONS)
    return im_paths.__next__()


UINT16_MAXVAL = 65535
UINT8_MAXVAL = 255
rgb_normalize = trans.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

SquareShape = Tuple[int, int, int, int]
ImageShape = Tuple[int, int]


def no_square(image_shape: ImageShape) -> SquareShape:
    return 0, 0, 0, 0


def center_square(image_shape: ImageShape, center_width_px: int,
                  center_height_px: int) -> SquareShape:
    width, height = image_shape
    assert center_width_px <= width
    assert center_height_px <= height
    center_x, center_y = width // 2, height // 2
    xmin, xmax = (center_x - center_width_px // 2,
                  center_x + center_width_px // 2)
    ymin, ymax = (center_y - center_height_px // 2,
                  center_y + center_height_px // 2)
    return xmin, xmax, ymin, ymax


center_square_50 = partial(
    center_square, center_width_px=50, center_height_px=50)


def random_square(image_shape: ImageShape,
                  min_area: int = 20 * 20,
                  max_area: int = None) -> SquareShape:
    x_lim, y_lim = image_shape[:2]
    if max_area is None:
        max_area = x_lim * y_lim
    for i in range(10):
        x_min, x_max = sorted(randint(0, x_lim, size=2))
        y_min, y_max = sorted(randint(0, y_lim, size=2))
        width = x_max - x_min
        height = y_max - y_min
        area = width * height
        if area > min_area and area <= max_area and height > 10 and width > 10:
            break
    else:
        x_min, x_max = 0, int(sqrt(max_area))
        y_min, y_max = 0, int(sqrt(max_area))
    return x_min, x_max, y_min, y_max


square_providers_by_name = {
    "none": no_square,
    "random": random_square,
    "center-50": center_square_50,
    "center-square":  #depricated
    center_square_50,
    "no-square": no_square  #depricated
}


def apply_square(square: SquareShape,
                 image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    x_min, x_max, y_min, y_max = square
    # using try catches bellow as workaround
    # for pytorch not currently supporting zero sized tensors.
    try:
        image[:x_min, ...] = 0
    except ValueError:
        pass
    try:
        image[x_max:, ...] = 0
    except ValueError:
        pass
    try:
        image[x_min:x_max, :y_min, ...] = 0
    except ValueError:
        pass
    try:
        image[x_min:x_max:, y_max:, ...] = 0
    except ValueError:
        pass
    return image

def inverse_apply_uniform_square_batch(square : torch.Tensor,batch : Union[torch.Tensor,torch.autograd.Variable]):
    x_min, x_max, y_min, y_max = square[0,:]
    batch[...,x_min:x_max,y_min:y_max] = 0

class SUNRGBDDataset(Dataset):

    modality_names = ['rgbd']  # , 'g', 'gd'

    def __init__(self,
                 base_path: str,
                 modality: str,
                 network_input_size: Tuple[int, int] = (192, 256),
                 network_output_size: Tuple[int, int] = (94, 126),
                 square_provider: Callable[
                     [ImageShape], SquareShape] = center_square_50) -> None:
        assert modality in SUNRGBDDataset.modality_names
        self.input_size = network_input_size
        self.output_size = network_output_size
        self.square_provider = square_provider
        self._im_folder_names = ["image", "depth", "depth_bfx"]
        self._v2_im_folder_names = ["image", "depth", "depthRaw"]
        self.base_path = base_path
        self.update()
        self.x_scale_factor = network_output_size[0] / network_input_size[0]
        self.y_scale_factor = network_output_size[1] / network_input_size[1]

    def update(self):
        self.data_paths = []
        self.resolutions = set()
        for path, dirs, files in os.walk(self.base_path):
            for im_folder_names in [
                    self._im_folder_names, self._v2_im_folder_names
            ]:
                if all(d in dirs for d in im_folder_names):
                    im_fns = tuple(
                        get_im_filename(join(self.base_path, path, folder))
                        for folder in im_folder_names)
                    self.data_paths.append(im_fns)
                    im_resolutions = tuple(
                        Image.open(im_fn).size for im_fn in im_fns)
                    self.resolutions.add(im_resolutions)
                    break

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        image_path, depth_path, depth_bfx_path, = self.data_paths[idx]
        image = imread(image_path)
        depth = scale_depth_image(imread(depth_path, as_grey=True))
        depth = np.expand_dims(depth, 2)
        depth_bfx = scale_depth_image(imread(depth_bfx_path, as_grey=True))
        depth_bfx = np.expand_dims(depth_bfx, 2)
        # TODO: CHECK DEPTH SCALE and what they do in sunrgbd:
        #   Was not able to find any documentation of how depth is scaled but using
        # LEARN SCALE AND DETPH SEPARATELY
        # Somenone elses paper as baseline.

        #image = image / 255
        # The question is wether i should divide by 255 here or not. In the training of the pretrained model they dont, but in the training in sparse_depth they do.

        scaling = uniform(1, 1.5)
        degree = uniform(-5, 5)
        square_input = self.square_provider(self.input_size)
        square_output = self._scale_square(square_input)
        apply_input_square = lambda image: apply_square(square_input, image)
        rotate = partial(t.rotate, angle=degree)
        scale = trans.Scale(round(min(self.input_size) * scaling))
        #scale = trans.Resize()
        scale_target = trans.Resize(self.output_size)
        color_jiter = trans.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        )
        color_normalize = trans.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        center_crop_feature = trans.CenterCrop(self.input_size)
        center_crop_target = trans.CenterCrop(self.output_size)


        to_tensor_from_f32 = lambda f32_pil_im: torch.from_numpy(np.array(f32_pil_im,np.float32,copy=False))
        rgb_transform = trans.Compose([
            trans.ToPILImage("RGB"), scale, rotate, color_jiter,
            center_crop_feature,
            trans.ToTensor(), color_normalize
        ])

        depth_transform = trans.Compose([
            trans.ToPILImage("F"), scale, rotate, center_crop_feature,
            to_tensor_from_f32, apply_input_square
        ])

        target_depth_transfrom = trans.Compose([
            trans.ToPILImage("F"),
            scale,
            rotate,
            center_crop_feature,
            scale_target,
            to_tensor_from_f32,
        ])

        image = rgb_transform(image)
        depth = depth_transform(depth)
        depth.unsqueeze_(0)
        depth /= scaling
        depth_bfx = target_depth_transfrom(depth_bfx)
        depth_bfx.unsqueeze_(0)
        depth_bfx /= scaling
        feature = cat(
            [image, depth]) if self.square_provider is not no_square else image
        square_input = torch.from_numpy(np.array(square_input))
        square_output = torch.from_numpy(np.array(square_output))
        return feature, depth_bfx, square_input, square_output

    def _scale_square(self, square):
        x_min, x_max, y_min, y_max = square
        x_min *= self.x_scale_factor
        x_max *= self.x_scale_factor
        y_min *= self.y_scale_factor
        y_max *= self.y_scale_factor
        return round(x_min), round(x_max), round(y_min), round(y_max)


def scale_depth_image(image: np.ndarray) -> np.ndarray:
    """Implements the same scaling of depht
    images as in the file read3dPoints.m 
    in the SUNRGBDTooolbox"""

    # TODO Check that this yield resonable output values: DONE, looks like metric
    im_scale = np.bitwise_or(
        np.left_shift(image, 16 - 3), np.right_shift(image, 3)) / 1000
    im_scale[im_scale > 8] = 8
    return im_scale.astype("float32")


def plot(feature=None, target=None, prediction=None):

    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2)
    if feature is not None:
        feature = np.asarray(feature)
        image = feature[:, :, :3]
        image = image - image.min()
        image = image / image.max()
        depth_in = feature[:, :, -1]
        ax1.imshow(image)
        ax2.imshow(depth_in.squeeze(), cmap="gray")
    ax1.set_title("in image")
    ax2.set_title("in depth")

    if target is not None:
        depth_target = np.asarray(target)
        ax3.imshow(depth_target.squeeze(), cmap="gray")
    ax3.set_title("depth target")

    if prediction is not None:
        depth_prediction = np.asarray(prediction)
        ax4.imshow(depth_prediction.squeeze(), cmap="gray")
    ax4.set_title("predicted depth")
    return fig

def compute_sunrgbd__metrics(dataset):

    for i in range(len(dataset)):
        dataset[i]


def main():
    dataset = SUNRGBDDataset(sys.argv[1], square_provider=center_square_50,modality="rgbd")
    i = 0
    # while True:

    f, t, sqi, sqo = dataset[i]
    i += 1
    fig = plot(
        feature=f.numpy().transpose(1, 2, 0),
        target=t.numpy().transpose(1, 2, 0))
    plt.show(fig)

if __name__ == "__main__":
    main()
