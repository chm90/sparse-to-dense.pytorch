import argparse
import dataloaders

parser = argparse.ArgumentParser(description="create plots for introduction section")
parser.add_argument(
    "--square-width",
    type=int,
    default=50,
    help="width of square in the middle (default 50)")
def main():
    args = parser.parse_args()
    image_shape = (192, 256)
    for depth_type in dataloaders.depth_types:
        dataset = dataloaders.NYUDataset(
            data_dir,
            phase='train',
            modality=args.modality,
            num_samples=0,
            square_width=args.square_width,
            output_shape=image_shape,
            depth_type=args.depth_type)

if __name__ == "__main__":
    main()