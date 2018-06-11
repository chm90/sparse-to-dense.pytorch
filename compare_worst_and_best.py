import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import argparse
import csv
import numpy as np
import os
from dataloaders import NYUDataset
parser = argparse.ArgumentParser(
    description=
    "Find the images with the greates difference in loss in a dataset between two models give a csv off losses for both models"
)

parser.add_argument("first_csv", type=argparse.FileType("r"))
parser.add_argument("second_csv", type=argparse.FileType("r"))
parser.add_argument("--first-name", type=str, default="first")
parser.add_argument("--second-name", type=str, default="second")
parser.add_argument("--loss-name",type=str,default="loss")
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


def main():
    args = parser.parse_args()
    first_losses_reader = csv.reader(args.first_csv)
    second_losses_reader = csv.reader(args.second_csv)
    first_losses = np.array(next(first_losses_reader), float)
    second_losses = np.array(next(second_losses_reader), float)
    args.data = os.path.join(os.environ["DATASET_DIR"], args.data)
    diff = np.abs(first_losses - second_losses)
    maxRatio = np.maximum(first_losses / second_losses,
        second_losses / first_losses)
    ratio = first_losses / second_losses
    diff = ratio
    decending_diff_idxs = np.argsort(diff)
    largest_diff_idxs = np.flip(decending_diff_idxs,0)
    

    print(f"=> number of images where ratio > 1: {np.sum(diff > 1)}")
    print(f"=> total number of images: {first_losses.size}")
    print(f"=> ration of images where ratio > 1: {np.sum(diff > 1)/first_losses.size}")
    #plot error distribution
    plt.figure()
    plt.plot(diff[largest_diff_idxs])
    plt.plot(first_losses[largest_diff_idxs])
    plt.plot(second_losses[largest_diff_idxs],"--")
    plt.title("Difference in error")
    plt.legend(["relative diff",args.first_name,args.second_name])
    plt.xlabel("image idx(sorted on relative diff)")
    plt.ylabel(args.loss_name)
    plt.show()

    #plot images from high difference in error to low
    valdir = os.path.join(args.data, 'val')
    image_shape = (192, 256)
    val_dataset = NYUDataset(
        valdir,
        phase='val',
        modality="rgb",
        num_samples=0,
        square_width=args.square_width,
        output_shape=image_shape)
    for i in largest_diff_idxs[::-1]:
        plt.figure()
        x, y = val_dataset[i]
        x = x.numpy().transpose(1, 2, 0)
        y = y.numpy().transpose(1, 2, 0)
        plt.title(
            f"image idx {i}: {args.first_name} {args.loss_name} = {first_losses[i]}, {args.second_name} {args.loss_name} = {second_losses[i]}"
        )
        plt.imshow(x)
        plt.show()


if __name__ == "__main__":
    main()