import argparse
from dataloaders import NYUDataset
import os
from utils import merge_ims_into_row, save_image, add_row
parser = argparse.ArgumentParser(
    description="create plots for introduction section")
parser.add_argument(
    "save_dir",
    metavar="SAVE-dir",
    type=str,
    help="Filename to use for saving the resulting image")


def main():
    args = parser.parse_args()
    image_shape = (192, 256)
    args.data = os.path.join(os.environ["DATASET_DIR"], "nyudepthv2")

    dataset_no_depth = NYUDataset(
        args.data,
        phase='val',
        modality="rgbd",
        num_samples=0,
        square_width=0,
        output_shape=image_shape,
        depth_type="square")

    dataset_square_hq = NYUDataset(
        args.data,
        phase='val',
        modality="rgbd",
        num_samples=0,
        square_width=50,
        output_shape=image_shape,
        depth_type="square")

    dataset_square_lq = NYUDataset(
        args.data,
        phase='val',
        modality="rgbd",
        num_samples=0,
        square_width=50,
        output_shape=image_shape,
        depth_type="low-quality-square")

    dataset_square_random_sample = NYUDataset(
        args.data,
        phase='val',
        modality="rgbd",
        num_samples=200,
        output_shape=image_shape,
        depth_type="full")

    dataset_single_pixel = NYUDataset(
        args.data,
        phase='val',
        modality="rgbd",
        num_samples=0,
        output_shape=image_shape,
        depth_type="single-pixel")
    dataset_single_pixel_lq = NYUDataset(
        args.data,
        phase='val',
        modality="rgbd",
        num_samples=0,
        output_shape=image_shape,
        depth_type="single-pixel-low-quality")

    images = []
    image_idx = 223

    #Create image with various possible depthmaps, not only from this work
    _, y = dataset_no_depth[image_idx]
    y = y.cuda().unsqueeze(0)
    images.append(y)
    for dataset in [
            dataset_square_hq, dataset_square_lq, dataset_square_random_sample
    ]:
        x, _ = dataset[image_idx]
        x = x.cuda().unsqueeze(0)
        images.append(x)
    square_shapes = [
        None, dataset_square_hq.square, dataset_square_hq.square, None
    ]
    im_merge_depth = merge_ims_into_row(
        images, rgbd_action="depth_only", square=square_shapes)

    os.makedirs(args.save_dir)
    depth_fn = os.path.join(args.save_dir, "depth.png")
    save_image(im_merge_depth, depth_fn)

    #Create image with source image
    x, y = dataset_no_depth[image_idx]
    x, y = x.cuda().unsqueeze(0), y.cuda().unsqueeze(0)
    im_merge_original = merge_ims_into_row([x, y], rgbd_action="rgb_only")
    original_fn = os.path.join(args.save_dir, "original.png")
    save_image(im_merge_original, original_fn)

    #Create image with all the depth maps investigated in this work
    #First row is hq images
    images = []
    squares = []
    x, _ = dataset_single_pixel[image_idx]
    images.append(x.cuda().unsqueeze(0))
    squares.append(None)

    for square_width in [10, 50, 100]:
        dataset_square_hq.square_width = square_width
        x, _ = dataset_square_hq[image_idx]
        images.append(x.cuda().unsqueeze(0))
        squares.append(dataset_square_hq.square)

    im_merge_depths_hq = merge_ims_into_row(images, square=squares,rgbd_action="depth_only")

    #Second row is lq images
    images = []
    squares = []

    x, _ = dataset_single_pixel_lq[image_idx]
    images.append(x.cuda().unsqueeze(0))
    squares.append(None)

    for square_width in [10, 50, 100]:
        dataset_square_lq.square_width = square_width
        x, _ = dataset_square_lq[image_idx]
        images.append(x.cuda().unsqueeze(0))
        squares.append(dataset_square_lq.square)

    im_merge_depths_lq = merge_ims_into_row(images, square=squares,rgbd_action="depth_only")
    im_merge_depths = add_row(im_merge_depths_hq, im_merge_depths_lq)
    depths_fn = os.path.join(args.save_dir, "depths.png")
    save_image(im_merge_depths, depths_fn)


if __name__ == "__main__":
    main()
