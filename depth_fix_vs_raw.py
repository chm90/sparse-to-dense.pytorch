from dataloaders import SUNRGBDDataset, NYUDataset
from main import data_names
import argparse
import os
phases = ["val","train"]
parser = argparse.ArgumentParser()
parser.add_argument("data",choices=data_names,help=f"One of {'|'.join(data_names)}")
parser.add_argument("--phase",choices=phases,help=f"One of {'|'.join(phases)}",default="train")

def main():
    args = parser.parse_args()
    data_path = os.path.join(os.environ["DATASET_DIR"],args.data,args.phase)
    DatasetType = SUNRGBDDataset if args.data == "SUNRGBD" else NYUDataset
    print("=> reading dataset ...")
    dataset = DatasetType(data_path,"val")
    result = dataset.compute_depth_metrics()
    print(result)

if __name__ == "__main__":
    main()
