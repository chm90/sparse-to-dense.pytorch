from argparse import ArgumentParser, FileType
import csv
import subprocess
import os
import time
from itertools import chain
import sys
import smtplib
from email.mime.text import MIMEText
import base64
from typing import Tuple, Union, Iterable
parser = ArgumentParser(
    description=
    "Run python script with arguments specified by a csv. The headder should contain the argument names."
    " If a collumn contains positional arguments, this is indicated by setting the first caracter of the first row of that column to '#'"
)
parser.add_argument(
    "script", metavar="PY-FILE", type=str, help="path to python script to run")
parser.add_argument(
    "run_spec",
    metavar="RUN-SPEC-CSV",
    type=str,
    help=
    "path to csv containing arguments names in the first row and various argument values in subsequent rows"
)
parser.add_argument(
    "--user-name",
    metavar="OUTLOOK-USER-NAME",
    type=str,
    default=None,
    help="your outlook email address for sending status messeges on")
parser.add_argument(
    "-pf",
    metavar="FILE-NAME",
    type=FileType(),
)
parser.add_argument(
    "--gpus",
    "-g",
    type=int,
    nargs="+",
    metavar="N",
    help="indexes of gpus to use for training (default 0)",
    default=[1])
parser.add_argument(
    "--procs-per-gpu",
    "-p",
    metavar="N",
    type=int,
    help="number of processes to run on each gpu(default 2)",
    default=1)

output_dir = "output"


def start_proc(script: str, gpu: int, arg_names: Iterable[str],
               arg_values: Iterable[str]) -> subprocess.Popen:
    #start new process
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    def filter_arg(arg_name_value: Tuple[str, str]
                   ) -> Union[Tuple[str, str], Tuple[str]]:
        arg_name, arg_value = arg_name_value
        is_positional = arg_name.strip()[0] == "#"
        if is_positional:
            return (arg_value, )
        is_flag = arg_value.strip() == ""
        if is_flag:
            return (arg_name, )
        return arg_name, arg_value

    args = [sys.executable, script] + list(
        chain.from_iterable(map(filter_arg, zip(arg_names, arg_values))))

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    fn_base = os.path.join(output_dir, ".".join(arg_values))
    out_fn = os.path.join(fn_base + ".out.txt")
    out_file = open(out_fn, "w")
    err_file = out_file
    proc = subprocess.Popen(args, env=env, stdout=out_file, stderr=err_file)
    print("training started at pid", proc.pid, "with args:\n", args)
    return proc


class SMTPEmailClient(object):
    def __init__(self,
                 user_name: str,
                 hash: str,
                 server_name: str = "smtp-mail.outlook.com",
                 server_port: int = 587) -> None:

        server = smtplib.SMTP(server_name, server_port)
        self.server_name = server_name
        self.server_port = server_port
        self.user_name = user_name
        self.hash = hash
        #Make sure that we can login
        server.starttls()
        server.login(self.user_name,
                     base64.b64decode(self.hash).decode("utf-8"))

    def send(self, to_email: str, subject: str, msg_text: str) -> None:
        msg = MIMEText(msg_text)
        msg["SUBJECT"] = subject
        msg["FROM"] = self.user_name
        msg["TO"] = to_email
        server = smtplib.SMTP(self.server_name, self.server_port)
        server.starttls()
        server.login(self.user_name,
                     base64.b64decode(self.hash).decode("utf-8"))
        server.send_message(msg)


def main():
    args = parser.parse_args()
    gpu_procs = {gpu: [] for gpu in args.gpus}
    sending_mails = args.user_name is not None
    if sending_mails:
        email_client = SMTPEmailClient(args.user_name, args.pf.readline())

    with open(args.run_spec) as csv_args:
        runs = csv.reader(csv_args, skipinitialspace=True)
        arg_names = list(map(str.strip, runs.__next__()))
        arg_values = runs.__next__()
        while arg_values is not None or any(
                len(procs) > 0 for procs in gpu_procs.values()):
            for gpu, procs in gpu_procs.items():
                #Monitor processes
                for proc in procs.copy():
                    exit_code = proc.poll()
                    if exit_code is not None:
                        msg = f"process {proc.pid} ,called with args\n{proc.args}\nexited with code {exit_code}"
                        subject = f"{proc.pid} exited with code {exit_code}"
                        if sending_mails:
                            try:
                                email_client.send(args.user_name, subject, msg)
                            except smtplib.SMTPHeloError as e:
                                print("Not able to send email, exception was:",
                                      e)
                        print(msg)
                        #remove_old_process
                        procs.remove(proc)
                #Start new processes if there are more arguments and there is room in the gpu
                if len(procs) < args.procs_per_gpu and arg_values is not None:
                    proc = start_proc(args.script, gpu, arg_names,
                                      list(map(str.strip, arg_values)))
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
