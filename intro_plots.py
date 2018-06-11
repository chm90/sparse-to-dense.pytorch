import argparse
from dataloaders import NYUDataset
import os
from utils import merge_ims_into_row, save_image
import cv2
parser = argparse.ArgumentParser(
    description="create plots for introduction section")
parser.add_argument(
    "save_dir",
    metavar="SAVE-dir",
    type=str,
    help="Filename to use for saving the resulting image")
parser.add_argument(
    "--square-width",
    type=int,
    default=50,
    help="width of square in the middle (default 50)")


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
        square_width=args.square_width,
        output_shape=image_shape,
        depth_type="square")

    dataset_square_lq = NYUDataset(
        args.data,
        phase='val',
        modality="rgbd",
        num_samples=0,
        square_width=args.square_width,
        output_shape=image_shape,
        depth_type="low-quality-square")

    dataset_square_random_sample = NYUDataset(
        args.data,
        phase='val',
        modality="rgbd",
        num_samples=200,
        output_shape=image_shape,
        depth_type="full")

    images = []
    image_idx = 223

    _, y = dataset_no_depth[image_idx]
    y = y.cuda().unsqueeze(0)
    images.append(y)
    for dataset in [
            dataset_square_hq, dataset_square_lq, dataset_square_random_sample
    ]:
        x, _ = dataset[image_idx]
        x = x.cuda()
        x = x.unsqueeze(0)
        images.append(x)
    images_with_square = [False,True, True, False]
    im_merge_depth = merge_ims_into_row(
        images,
        rgbd_action="depth_only",
        images_with_square=images_with_square,
        square=dataset_square_hq.square)

    os.makedirs(args.save_dir)
    depth_fn = os.path.join(args.save_dir, "depth.png")
    save_image(im_merge_depth, depth_fn)

    x, y = dataset_no_depth[image_idx]
    x, y = x.cuda().unsqueeze(0), y.cuda().unsqueeze(0)
    im_merge_original = merge_ims_into_row([x, y], rgbd_action="rgb_only")
    original_fn = os.path.join(args.save_dir, "original.png")
    save_image(im_merge_original, original_fn)


if __name__ == "__main__":
    main()
