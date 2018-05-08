from argparse import ArgumentParser
import csv
import subprocess
import os
import time
from itertools import chain
import sys
from pprint import pprint
parser = ArgumentParser(
    description="run network(main.py) with arguments specified by a csv")
parser.add_argument(
    "a",
    type=str,
    help=
    "path to csv containing arguments names in the first row and various argument values in subsequent rows"
)
parser.add_argument(
    "--gpus",
    type=int,
    nargs="+",
    help="indexes of gpus to use for training (default 0)",
    default=[1])
parser.add_argument(
    "--procs-per-gpu",
    type=int,
    help="number of processes to run on each gpu(default 2)",
    default=2)

output_dir = "output"

def start_proc(gpu, arg_names, arg_values):
    #start new process
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    args_ = [sys.executable, "main.py"] + list(
        chain.from_iterable(zip(arg_names, arg_values)))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    fn_base = os.path.join(output_dir,".".join(arg_values))
    out_fn = os.path.join(fn_base + ".out.txt")
    #err_fn = os.path.join(fn_base + ".err.txt")
    out_file = open(out_fn,"w")
    err_file = out_file#open(err_fn,"w")
    proc = subprocess.Popen(
        args_, env=env, stdout=out_file, stderr=err_file)
    print("training started at pid", proc.pid)
    return proc


def main():
    args = parser.parse_args()
    gpu_procs = {gpu: [] for gpu in args.gpus}
    with open(args.a) as csv_args:
        runs = csv.reader(csv_args, skipinitialspace=True)
        arg_names = list(map(str.strip,runs.__next__()))
        arg_values = runs.__next__()
        while arg_values is not None or any(
                len(procs) > 0 for procs in gpu_procs.values()):
            for gpu, procs in gpu_procs.items():
                #Monitor processes
                for proc in procs.copy():
                    exit_code = proc.poll()
                    if exit_code is not None:
                        print("process", proc.pid, "exited with code",
                              exit_code)
                        #remove_old_process
                        procs.remove(proc)
                #Start new processes if there are more arguments and there is room in the gpu
                if len(procs) < args.procs_per_gpu and arg_values is not None:
                    proc = start_proc(gpu, arg_names, list(map(str.strip,arg_values)))
                    procs.append(proc)
                    try:
                        arg_values = runs.__next__()
                    except StopIteration:
                        arg_values = None
                    break
            else:
                time.sleep(3)
                #if no new process was added, wait some time before polling again

    print("All done, exiting")


if __name__ == "__main__":
    main()