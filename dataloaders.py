import os
import os.path
from typing import Tuple, Union, List, Any, Dict, Type
from os import listdir
from os.path import isfile, join, splitext
import h5py
import numpy as np
import torch
import torch.utils.data as data
from skimage.io import imread
from skimage.transform import downscale_local_mean
import transforms
import torchvision
from metrics import Result, AverageMeter
from sys import stdout
from numba import njit, float64, types, int64, float32,none
IMG_EXTENSIONS = [
    '.h5',
]

TNpData = Tuple[np.ndarray, np.ndarray, np.ndarray]
SquareShape = Tuple[int, int, int, int]
ImageShape = Tuple[int, int]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


default_oheight, default_owidth = 228, 304  # image size after pre-processing
color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)


def rgb2grayscale(rgb):
    return rgb[:, :, 0] * 0.2989 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114


to_tensor = transforms.ToTensor()


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


@njit(none(float32[:,:],int64,float64))
def downscale(depth_img,pixel_count,std_dev):
    x_max,y_max = depth_img.shape
    x_steps = list(range(0,x_max,pixel_count))
    if x_steps[-1] != x_max:
        x_steps.append(x_max)
    x_ranges = list(zip(x_steps,x_steps[1:]))
    y_steps = list(range(0,y_max,pixel_count))
    if y_steps[-1] != y_max:
        y_steps.append(y_max)
    y_ranges = list(zip(y_steps,y_steps[1:]))
    for xlow, xhigh in x_ranges:
        for ylow, yhigh in y_ranges:
             new_depth = depth_img[xlow:xhigh,ylow:yhigh].mean()
             new_depth = np.random.normal(new_depth, std_dev * new_depth)
             depth_img[xlow:xhigh,ylow:yhigh] = new_depth



class RGBDDataset(data.Dataset):
    modality_names = ['rgb', 'rgbd', 'd']

    def __init__(self,
                 root: str,
                 phase: str,
                 modality: str = 'rgb',
                 num_samples: int = 0,
                 square_width: int = 0,
                 output_shape: Tuple[int, int] = (default_oheight,
                                                  default_owidth),
                 low_quality_cam=True) -> None:
        self.low_quality_cam = low_quality_cam
        self.oheight, self.owidth = output_shape
        self.phase = phase
        self.paths = self._get_data_paths(root)
        self.square = center_square((self.oheight, self.owidth), square_width,
                                    square_width)
        if len(self.paths) == 0:
            raise (RuntimeError(
                "Found 0 images in subfolders of: " + root + "\n"
                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        if self.phase == 'train':
            self.transform = self.train_transform
        elif self.phase == 'val':
            self.transform = self.val_transform
        else:
            raise (RuntimeError("Invalid dataset phase: " + self.phase + "\n"
                                "Supported dataset phases are: train, val"))

        if modality in self.modality_names:
            self.modality = modality
            if modality in ['rgbd', 'd', 'gd']:
                self.num_samples = num_samples
                self.square_width = square_width
            else:
                self.num_samples = 0
                self.square_width = 0
        else:
            raise (RuntimeError("Invalid modality type: " + modality + "\n"
                                "Supported dataset types are: " +
                                ''.join(self.modality_names)))

    @property
    def output_shape(self) -> Tuple[int, int]:
        return self.oheight, self.owidth

    @property
    def mask_inside_square(self) -> Tuple[slice, slice, slice, slice]:
        x_min, x_max, y_min, y_max = self.square
        return (slice(None), slice(None), slice(x_min, x_max),
                slice(y_min, y_max))

    def train_transform(self, rgb: np.ndarray, depth_raw: np.ndarray,
                        depth_fix: np.ndarray) -> TNpData:
        s = np.random.uniform(1.0, 1.5)  # random scaling
        depth_raw = depth_raw / s
        depth_fix = depth_fix / s
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip
        # perform 1st part of data augmentation
        transform = transforms.Compose([
            transforms.Resize(
                250.0 / self.iheight
            ),  # this is for computational efficiency, since rotation is very slow
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.CenterCrop((self.oheight, self.owidth)),
            transforms.HorizontalFlip(do_flip)
        ])
        rgb = transform(rgb)

        # random color jittering
        rgb = color_jitter(rgb)

        rgb = np.asfarray(rgb, dtype='float') / 255
        depth_raw = transform(depth_raw)
        depth_fix = transform(depth_fix)

        return rgb, depth_raw, depth_fix

    def val_transform(self, rgb: np.ndarray, depth_raw: np.ndarray,
                      depth_fix: np.ndarray) -> TNpData:
        # perform 1st part of data augmentation
        transform = transforms.Compose([
            transforms.Resize(240.0 / self.iheight),
            transforms.CenterCrop((self.oheight, self.owidth)),
        ])
        rgb = transform(rgb)
        rgb = np.asfarray(rgb, dtype='float') / 255
        depth_raw = transform(depth_raw)
        depth_fix = transform(depth_fix)
        return rgb, depth_raw, depth_fix

    def create_subsampled_depth(self, depth: np.ndarray) -> np.ndarray:
        depth_subsampled = depth.copy()
        # remove depth values outside center square
        if self.low_quality_cam:
            downscale(depth_subsampled,10,0.05)
        apply_square(self.square, depth_subsampled)

        # provide random depth points

        return depth_subsampled

    def create_rgbd(self, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        sparse_depth = self.create_subsampled_depth(depth)
        # rgbd = np.dstack((rgb[:,:,0], rgb[:,:,1], rgb[:,:,2], sparse_depth))
        rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2)
        return rgbd

    def compute_depth_metrics(self, verbose=True) -> Result:
        """Computes metrics on the difference between raw and fixed depth values"""
        avg = AverageMeter()
        for i, path in enumerate(self.paths):
            _, depth_raw, depth_fix = self.load_images(path)
            depth_raw = torch.tensor(depth_raw)
            depth_fix = torch.tensor(depth_fix)
            res = Result()
            res.evaluate(depth_raw, depth_fix)
            avg.update(res, 0, 0, 1)
            if verbose:
                stdout.write(f"=> computing img {i}/{len(self)}\r")
        if verbose:
            stdout.write("\n")
        return avg.average()

    def load_images(self,
                    path: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "This method must be implemented in subclasses")

    def _get_data_paths(self, base: str) -> List[Any]:
        raise NotImplementedError(
            "This method must be implemented in subclasses")

    def __getraw__(self, index: int) -> TNpData:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (rgb, depth) the raw data.
        """
        rgb, depth_raw, depth_fix = self.load_images(self.paths[index])
        return rgb, depth_raw, depth_fix

    def __get_all_item__(self, index: int
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                    np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (input_tensor, depth_tensor, input_np, depth_np)
        """

        rgb, depth_raw, depth_fix = self.__getraw__(index)
        if self.transform is not None:
            rgb_np, depth_raw_np, depth_fix_np = self.transform(
                rgb, depth_raw, depth_fix)
        else:
            raise (RuntimeError("transform not defined"))

        # color normalization
        # rgb_tensor = normalize_rgb(rgb_tensor)
        # rgb_np = normalize_np(rgb_np)
        if self.modality == 'rgb':
            input_np = rgb_np
        elif self.modality == 'rgbd':
            input_np = self.create_rgbd(rgb_np, depth_raw_np)
        elif self.modality == 'd':
            input_np = self.create_subsampled_depth(depth_raw_np)
        input_tensor = to_tensor(input_np)
        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)
        depth_raw_tensor = to_tensor(depth_raw_np)
        depth_raw_tensor = depth_raw_tensor.unsqueeze(0)

        depth_fix_tensor = to_tensor(depth_fix_np)
        depth_fix_tensor = depth_fix_tensor.unsqueeze(0)

        return input_tensor, depth_raw_tensor, depth_fix_tensor, input_np, depth_raw_np, depth_fix_np

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (input_tensor, depth_tensor) 
        """
        input_tensor, depth_raw_tensor, depth_fix_tensor, input_np, depth_raw_np, depth_fix_np = self.__get_all_item__(
            index)

        return input_tensor, depth_fix_tensor

    def __iter__(self):
        pass

    def __len__(self) -> int:
        return len(self.paths)


class NYUDataset(RGBDDataset):
    iheight = 480
    iwidth = 640

    def _get_data_paths(self, base: str) -> List[str]:
        classes, class_to_idx = self.find_classes(base)
        paths = [path for path, idx in self.make_dataset(base, class_to_idx)]
        return paths

    def load_images(self, path: str) -> TNpData:
        h5f = h5py.File(path, "r")
        rgb = np.array(h5f['rgb'])
        rgb = np.transpose(rgb, (1, 2, 0))
        depth = np.array(h5f['depth'])
        return rgb, depth, depth.copy()

    @staticmethod
    def make_dataset(dir,
                     class_to_idx: Dict[str, int]) -> List[Tuple[str, int]]:
        images = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)

        return images

    @staticmethod
    def find_classes(dir):
        classes = [
            d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))
        ]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


class SUNRGBDDataset(RGBDDataset):
    _im_folder_names = ["image", "depth", "depth_bfx"]
    _v2_im_folder_names = ["image", "depth", "depthRaw"]
    iheight = 427
    iwidth = 561

    def load_images(self, image_paths: Tuple[str, str, str]) -> TNpData:
        rgb_fn, depth_raw_fn, depth_fix_fn = image_paths
        rgb = imread(rgb_fn)
        depth_raw = self.scale_depth_image(imread(depth_raw_fn, as_grey=True))
        depth_fix = self.scale_depth_image(imread(depth_fix_fn, as_grey=True))
        return rgb, depth_raw, depth_fix

    @staticmethod
    def _get_im_filename(folder: str) -> str:
        im_paths = (join(folder, f) for f in listdir(folder)
                    if isfile(join(folder, f)) and splitext(f)[1] in
                    torchvision.datasets.folder.IMG_EXTENSIONS)
        return next(im_paths)

    def _get_data_paths(self, base: str) -> List[Tuple[str, str, str]]:
        data_paths = []
        for path, dirs, files in os.walk(base):
            for im_folder_names in [
                    self._im_folder_names, self._v2_im_folder_names
            ]:
                if all(d in dirs for d in im_folder_names):
                    rgb_dir, depth_raw_dir, depth_fix_dir = map(
                        lambda dir: join(base, path, dir), im_folder_names)
                    rgb_fn = self._get_im_filename(rgb_dir)
                    depth_raw_fn = self._get_im_filename(depth_raw_dir)
                    depth_fix_fn = self._get_im_filename(depth_fix_dir)
                    data_paths.append((rgb_fn, depth_raw_fn, depth_fix_fn))
                    break
        data_paths = sorted(data_paths)
        split_idx = len(data_paths) // 10
        if self.phase == "val":
            data_paths = data_paths[:split_idx]
        elif self.phase == "train":
            data_paths = data_paths[split_idx:]
        else:
            raise Exception("We should never be here")

        return data_paths

    @staticmethod
    def scale_depth_image(image: np.ndarray) -> np.ndarray:
        """Implements the same scaling of depht
        images as in the file read3dPoints.m 
        in the SUNRGBDTooolbox"""

        # TODO Check that this yield resonable output values: DONE, looks like metric
        im_scale = np.bitwise_or(
            np.left_shift(image, 16 - 3), np.right_shift(image, 3)) / 1000
        im_scale[im_scale > 8] = 8
        return im_scale.astype("float32")


data_names_2_type = {
    'nyudepthv2': NYUDataset,
    'SUNRGBD': SUNRGBDDataset
}  #type: Dict[str,Type[RGBDDataset]]

data_names = list(data_names_2_type.keys())


def choose_dataset_type(dataset_name: str) -> RGBDDataset:
    return data_names_2_type[dataset_name]
