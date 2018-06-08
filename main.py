import argparse
import csv
import os
import shutil
import sys
import time
from sys import stdout
from typing import List, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from matplotlib import pyplot as plt

import typing
import criteria
import evaluate
import utils
from dataloaders import RGBDDataset, choose_dataset_type, depth_types
from metrics import AverageMeter, Result, MaskedResult
from models import SKIP_TYPES, Decoder, ResNet

ResultListT = List[Tuple[Result, Result, Result]]
modality_names = RGBDDataset.modality_names
model_names = ['resnet18', 'resnet50', 'my_resnet18']
loss_names = ['l1', 'l2']
data_names = ['nyudepthv2', "SUNRGBD"]
decoder_names = Decoder.names
depth_sampling_types = depth_types
optimizers = ["sgd", "adam"]
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Sparse-to-Dense Training')
# parser.add_argument('--data', metavar='DIR', help='path to dataset',
#                     default="data/NYUDataset")
parser.add_argument(
    '--arch',
    '-a',
    metavar='ARCH',
    default='resnet18',
    choices=model_names,
    help='model architecture: ' + ' | '.join(model_names) +
    ' (default: resnet18)')
parser.add_argument(
    '--data',
    metavar='DATA',
    default='nyudepthv2',
    choices=data_names,
    help='dataset: ' + ' | '.join(data_names) + ' (default: nyudepthv2)')
parser.add_argument(
    '--modality',
    '-m',
    metavar='MODALITY',
    default='rgb',
    choices=modality_names,
    help='modality: ' + ' | '.join(modality_names) + ' (default: rgb)')
parser.add_argument(
    "--depth-type",
    "-dt",
    default="square",
    choices=depth_sampling_types,
    help="The kind of depth sample to use" + " | ".join(depth_sampling_types) +
    "Only used if modality contains depth. (default: square)")
parser.add_argument(
    '-s',
    '--num-samples',
    default=0,
    type=int,
    metavar='N',
    help='number of sparse depth samples (default: 0)')
parser.add_argument(
    '--decoder',
    '-d',
    metavar='DECODER',
    default='deconv2',
    choices=decoder_names,
    help='decoder: ' + ' | '.join(decoder_names) + ' (default: deconv2)')
parser.add_argument(
    '-j',
    '--workers',
    default=10,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 10)')
parser.add_argument(
    '--epochs',
    default=100,
    type=int,
    metavar='N',
    help='maximum number of total epochs to run (default: 15)')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-c',
    '--criterion',
    metavar='LOSS',
    default='l1',
    choices=loss_names,
    help='loss function: ' + ' | '.join(loss_names) + ' (default: l1)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=16,
    type=int,
    help='mini-batch size (default: 16)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.01,
    type=float,
    metavar='LR',
    help='initial learning rate (default 0.01)')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)')
parser.add_argument(
    '--print-freq',
    '-p',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_true',
    default=True,
    help='use ImageNet pre-trained weights (default: True)')
parser.add_argument(
    "--skip-type",
    "-st",
    choices=SKIP_TYPES,
    default="none",
    help="the type of skip connection to use (default: none)")

parser.add_argument(
    "--early-stop-epochs",
    "-ese",
    type=int,
    default=10,
    help="Epochs with non improving validation error before stoping")

parser.add_argument(
    "--optimizer",
    "-opt",
    choices=optimizers,
    help=f"The optimizer to use, one of {optimizers} (default: adam)",
    default="adam")
parser.add_argument(
    "--output-dir",
    "-o",
    help="directory where to place results (default: results)",
    default="results")
parser.add_argument(
    "--adjust-lr-ep",
    help=
    "number of epochs with non decreasing validation loss before adjusting learning rate",
    default=3,
    type=int)
parser.add_argument(
    "--min-adjust-lr-ep",
    help="minimum number of epochs before adjusting learning rate",
    type=int,
    default=5)
parser.add_argument(
    "--square-width",
    type=int,
    default=50,
    help="width of square in the middle (default 50)")
parser.add_argument(
    "--transfer-from",
    help="model checkpoint to do transfer learning from",
    type=str)
parser.add_argument(
    "--train-top-only",
    action="store_true",
    default=False,
    help=
    "if set, train only the top two layers. Only applies if aditionaly using --transfer-from"
)


fieldnames = [
    'mse', 'rmse', 'rmse inside', 'rmse outside', 'absrel', 'absrel inside',
    'absrel outside', 'lg10', 'mae', 'mae inside', 'mae outside', 'delta1',
    "delta1 inside", 'delta1 outside', 'delta2', 'delta3', 'data_time',
    'gpu_time'
]
best_result = Result()
best_result.set_to_worst()


def main() -> int:
    global args, best_result, output_directory, train_csv, test_csv
    args = parser.parse_args()
    dataset = args.data
    if args.modality == 'rgb' and args.num_samples != 0:
        print("number of samples is forced to be 0 when input modality is rgb")
        args.num_samples = 0
    image_shape = (192, 256)  # if "my" in args.arch else (228, 304)

    # create results folder, if not already exists
    if args.transfer_from:
        output_directory = f"{args.transfer_from}_transfer"
    else:
        output_directory = os.path.join(
            args.output_dir,
            f'{args.data}.modality={args.modality}.arch={args.arch}'
            f'.skip={args.skip_type}.decoder={args.decoder}'
            f'.criterion={args.criterion}.lr={args.lr}.bs={args.batch_size}'
            f'.opt={args.optimizer}.depth-type={args.depth_type}')
    args.data = os.path.join(os.environ["DATASET_DIR"], args.data)
    print("output directory :", output_directory)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    elif not args.evaluate:
        raise Exception("output directory allready exists")

    train_csv = os.path.join(output_directory, 'train.csv')
    test_csv = os.path.join(output_directory, 'test.csv')
    best_txt = os.path.join(output_directory, 'best.txt')

    # define loss function (criterion) and optimizer
    if args.criterion == 'l2':
        criterion = criteria.MaskedMSELoss().cuda()
    elif args.criterion == 'l1':
        criterion = criteria.MaskedL1Loss().cuda()
    out_channels = 1
    # Data loading code
    print("=> creating data loaders ...")
    traindir = os.path.join(args.data, 'train')
    valdir = traindir if dataset == "SUNRGBD" else os.path.join(
        args.data, 'val')
    DatasetType = choose_dataset_type(dataset)
    train_dataset = DatasetType(
        traindir,
        phase='train',
        modality=args.modality,
        num_samples=args.num_samples,
        square_width=args.square_width,
        output_shape=image_shape,
        depth_type=args.depth_type)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        sampler=None)

    print("=> training examples:", len(train_dataset))

    val_dataset = DatasetType(
        valdir,
        phase='val',
        modality=args.modality,
        num_samples=args.num_samples,
        square_width=args.square_width,
        output_shape=image_shape,
        depth_type=args.depth_type)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    print("=> validation examples:", len(val_dataset))

    print("=> data loaders created.")

    # evaluation mode
    if args.evaluate:
        best_model_filename = os.path.join(output_directory,
                                           'model_best.pth.tar')
        if os.path.isfile(best_model_filename):
            print("=> loading best model '{}'".format(best_model_filename))
            checkpoint = torch.load(best_model_filename)
            args.start_epoch = checkpoint['epoch']
            best_result = checkpoint['best_result']
            model = checkpoint['model']
            print("=> loaded best model (epoch {})".format(
                checkpoint['epoch']))
        else:
            print("=> no best model found at '{}'".format(best_model_filename))
        avg_result, avg_result_inside, avg_result_outside, _, results, evaluator = validate(
            val_loader, model, checkpoint['epoch'], write_to_file=False)
        write_results(best_txt, avg_result, avg_result_inside,
                      avg_result_outside, checkpoint['epoch'])
        for loss_name, losses in [
            ("rmses", (res.result.rmse for res in results)),
            ("delta1s", (res.result.delta1 for res in results)),
            ("delta2s", (res.result.delta2 for res in results)),
            ("delta3s", (res.result.delta3 for res in results)),
            ("maes", (res.result.mae for res in results)),
            ("absrels", (res.result.absrel for res in results)),
            ("rmses_inside", (res.result_inside.rmse for res in results)),
            ("delta1s_inside", (res.result_inside.delta1 for res in results)),
            ("delta2s_inside", (res.result_inside.delta2 for res in results)),
            ("delta3s_inside", (res.result_inside.delta3 for res in results)),
            ("maes_inside", (res.result_inside.mae for res in results)),
            ("absrels_inside", (res.result_inside.absrel for res in results)),
            ("rmses_outside", (res.result_outside.rmse for res in results)),
            ("delta1s_outside", (res.result_outside.delta1 for res in results)),
            ("delta2s_outside", (res.result_outside.delta2 for res in results)),
            ("delta3s_outside", (res.result_outside.delta3 for res in results)),
            ("maes_outside", (res.result_outside.mae for res in results)),
            ("absrels_outside", (res.result_outside.absrel for res in results)),
        ]:
            with open(
                    os.path.join(output_directory,
                                 f"validation_{loss_name}.csv"),
                    "w") as csv_file:
                wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
                wr.writerow(losses)

        evaluator.save_plot(os.path.join(output_directory, "best.png"))
        return 0

    # optionally resume from a checkpoint
    elif args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            best_result = checkpoint['best_result']
            model = checkpoint['model']
            optimizer = checkpoint['optimizer']
            print("=> loaded checkpoint (epoch {})".format(
                checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return 1
    # create new model
    else:
        if args.transfer_from:
            if os.path.isfile(args.transfer_from):
                print(f"=> loading checkpoint '{args.transfer_from}'")
                checkpoint = torch.load(args.transfer_from)
                args.start_epoch = 0
                model = checkpoint['model']
                print("=> loaded checkpoint")
                train_params = list(model.conv3.parameters()) + list(
                    model.decoder.layer4.parameters()
                ) if args.train_top_only else model.parameters()
            else:
                print(f"=> no checkpoint found at '{args.transfer_from}'")
                return 1
        else:
            # define model
            print("=> creating Model ({}-{}) ...".format(
                args.arch, args.decoder))
            in_channels = len(args.modality)
            if args.arch == 'resnet50':
                n_layers = 50
            elif args.arch == 'resnet18':
                n_layers = 18
            model = ResNet(
                layers=n_layers,
                decoder=args.decoder,
                in_channels=in_channels,
                out_channels=out_channels,
                pretrained=args.pretrained,
                image_shape=image_shape,
                skip_type=args.skip_type)
            print("=> model created.")
            train_params = model.parameters()

        adjusting_learning_rate = False
        if args.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                train_params,
                args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay)
            adjusting_learning_rate = True
        elif args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                train_params, weight_decay=args.weight_decay)
        else:
            raise Exception("We should never be here")

        if adjusting_learning_rate:
            print("=> Learning rate adjustment enabled.")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=args.adjust_lr_ep, verbose=True)
        # create new csv files with only header
        with open(train_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        with open(test_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    print(model)
    print("=> model transferred to GPU.")
    epochs_since_best = 0
    train_results = []
    val_results = []
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        res_train, res_train_inside, res_train_outside = train(
            train_loader, model, criterion, optimizer, epoch)
        train_results.append((res_train, res_train_inside, res_train_outside))
        # evaluate on validation set
        res_val, res_val_inside, res_val_outside, img_merge, _, _ = validate(
            val_loader, model, epoch, True)
        val_results.append((res_val, res_val_inside, res_val_outside))
        # remember best rmse and save checkpoint
        is_best = res_val.rmse < best_result.rmse
        if is_best:
            epochs_since_best = 0
            best_result = res_val
            write_results(best_txt, res_val, res_val_inside, res_val_outside,
                          epoch)
            if img_merge is not None:
                img_filename = output_directory + '/comparison_best.png'
                utils.save_image(img_merge, img_filename)
        else:
            epochs_since_best += 1

        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'model': model,
            'best_result': best_result,
            'optimizer': optimizer,
        }, is_best, epoch)

        plot_progress(train_results, val_results, epoch)

        if epochs_since_best > args.early_stop_epochs:
            print("early stopping")
        if adjusting_learning_rate:
            scheduler.step(res_val.rmse)
    return 0


def write_results(txt: str, res: Result, res_inside: Result,
                  res_outside: Result, epoch: int):
    res.name = "result"
    res_inside.name = "result inside"
    res_outside.name = "result outside"
    with open(txt, 'w') as txtfile:
        txtfile.write(str(res) + "\n")
        txtfile.write(str(res_inside) + "\n")
        txtfile.write(str(res_outside) + "\n")
        txtfile.write(f"epoch: {epoch}")


def train(train_loader, model, criterion, optimizer,
          epoch) -> Tuple[Result, Result, Result]:
    average_meter = AverageMeter()
    inside_average_meter = AverageMeter()
    outside_average_meter = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input, target = input.cuda(), target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        torch.cuda.synchronize()
        data_time = time.time() - end
        # compute depth_pred
        end = time.time()
        depth_pred = model(input_var)
        #inverse_apply_uniform_square_batch(output_square,depth_pred)
        #inverse_apply_uniform_square_batch(output_square,target_var)
        loss = criterion(depth_pred, target_var)
        optimizer.zero_grad()
        loss.backward()  # compute gradient and do SGD step
        optimizer.step()
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        output1 = torch.index_select(depth_pred.data, 1,
                                     torch.cuda.LongTensor([0]))
        #assume all squares are of same size
        result = MaskedResult(train_loader.dataset.mask_inside_square)
        result.evaluate(output1, target)
        average_meter.update(result.result, gpu_time, data_time, input.size(0))
        inside_average_meter.update(result.result_inside, gpu_time, data_time,
                                    input.size(0))
        outside_average_meter.update(result.result_outside, gpu_time, data_time,
                                     input.size(0))
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            #print('=> output: {}'.format(output_directory))
            def print_result(result, result_name, averege_meter):
                average = averege_meter.average()
                stdout.write(
                    f"{result_name}: "
                    f'Train Epoch: {epoch} [{i + 1}/{len(train_loader)}]\t'
                    f't_Data={data_time:.3f}({average.data_time:.3f}) '
                    f't_GPU={gpu_time:.3f}({average.gpu_time:.3f}) '
                    f'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                    f'MAE={result.mae:.2f}({average.mae:.2f}) '
                    f'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                    f'REL={result.absrel:.3f}({average.absrel:.3f}) '
                    f'Lg10={result.lg10:.3f}({average.lg10:.3f}) \n'
                    '\n')

            print_result(result.result, "result", average_meter)
            print_result(result.result_inside, "result_inside", inside_average_meter)
            print_result(result.result_outside, "result_outside",
                         outside_average_meter)

    avg = average_meter.average()
    avg_inside = inside_average_meter.average()
    avg_outside = outside_average_meter.average()
    with open(train_csv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
            'mse': avg.mse,
            'rmse': avg.rmse,
            'rmse inside': avg_inside.rmse,
            'rmse outside': avg_outside.rmse,
            'absrel': avg.absrel,
            'absrel inside': avg_inside.absrel,
            'absrel outside': avg_outside.absrel,
            'lg10': avg.lg10,
            'mae': avg.mae,
            'mae inside': avg_inside.mae,
            'mae outside': avg_outside.mae,
            'delta1': avg.delta1,
            'delta1 inside': avg_inside.delta1,
            'delta1 outside': avg_outside.delta1,
            'delta2': avg.delta2,
            'delta3': avg.delta3,
            'gpu_time': avg.gpu_time,
            'data_time': avg.data_time
        })
    return avg, avg_inside, avg_outside


def validate(val_loader,
             model: torch.nn.Module,
             epoch: int,
             write_to_file: bool = True
             ) -> typing.Tuple[Result, Result, Result, np.array, typing.List[
                 MaskedResult], evaluate.Evaluator]:
    average_meter = AverageMeter()
    inside_average_meter = AverageMeter()
    outside_average_meter = AverageMeter()

    # switch to evaluate mode
    model.eval()
    evaluator = evaluate.Evaluator(val_loader.dataset.output_shape,
                                   args.square_width)
    end = time.time()
    results = []
    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        depth_pred = model(input_var)
        torch.cuda.synchronize()
        gpu_time = time.time() - end
        # measure accuracy and record loss
        output1 = torch.index_select(depth_pred.data, 1,
                                     torch.cuda.LongTensor([0]))
        evaluator.add_results(output1, target)
        #assume all squares are of same size
        result = MaskedResult(val_loader.dataset.mask_inside_square)
        result.evaluate(output1, target)
        results.append(result)
        average_meter.update(result.result, gpu_time, data_time, input.size(0))
        inside_average_meter.update(result.result_inside, gpu_time, data_time,
                                    input.size(0))
        outside_average_meter.update(result.result_outside, gpu_time, data_time,
                                     input.size(0))
        end = time.time()
        # save 8 images for visualization
        skip = 50
        if args.modality == 'd':
            img_merge = None
        else:
            if i == 0:
                img_merge = utils.merge_into_row(input, target, depth_pred)
            elif (i < 8 * skip) and (i % skip == 0):
                row = utils.merge_into_row(input, target, depth_pred)
                img_merge = utils.add_row(img_merge, row)
            elif i == 8 * skip:
                filename = output_directory + '/comparison_' + str(
                    epoch) + '.png'
                utils.save_image(img_merge, filename)
        average = average_meter.average()
        if (i + 1) % args.print_freq == 0:
            #print('=> output: {}'.format(output_directory))
            def print_result(result, result_name):
                stdout.write(
                    f'Validation Epoch: {epoch} [{i + 1}/{len(val_loader)}]\t'
                    f"{result_name}: "
                    #f't_Data={data_time:.3f}({average.data_time:.3f}) '
                    #f't_GPU={gpu_time:.3f}({average.gpu_time:.3f}) '
                    f'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                    f'MAE={result.mae:.2f}({average.mae:.2f}) '
                    f'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                    #f'REL={result.absrel:.3f}({average.absrel:.3f}) '
                    #f'Lg10={result.lg10:.3f}({average.lg10:.3f}) \n'
                    '\n')
            print_result(result.result,"result")

    avg = average_meter.average()
    avg_inside = inside_average_meter.average()
    avg_outside = outside_average_meter.average()
    avg.name = "average"
    avg_inside.name = "average inside"
    avg_outside.name = "average outside"

    gpu_time = average.gpu_time
    print(
        f'\n*\n' + str(avg) + "\n" + str(avg_inside) + "\n" + str(avg_outside))

    if write_to_file:
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'mse': avg.mse,
                'rmse': avg.rmse,
                'rmse inside': avg_inside.rmse,
                'rmse outside': avg_outside.rmse,
                'absrel': avg.absrel,
                'absrel inside': avg_inside.absrel,
                'absrel outside': avg_outside.absrel,
                'lg10': avg.lg10,
                'mae': avg.mae,
                'mae inside': avg_inside.mae,
                'mae outside': avg_outside.mae,
                'delta1': avg.delta1,
                'delta1 inside': avg_inside.delta1,
                'delta1 outside': avg_outside.delta1,
                'delta2': avg.delta2,
                'delta3': avg.delta3,
                'gpu_time': avg.gpu_time,
                'data_time': avg.data_time
            })
    evaluator.save_plot(
        os.path.join(output_directory, f"evaluation_epoch{epoch}.png"))

    return avg, avg_inside, avg_outside, img_merge, results, evaluator


def save_checkpoint(state, is_best, epoch):
    checkpoint_filename = os.path.join(output_directory,
                                       'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(
            output_directory, 'checkpoint-' + str(epoch - 1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)


def plot_progress(train_results: ResultListT, val_results: ResultListT,
                  epoch: int) -> None:
    global output_directory
    plt.figure()
    rmse_train = np.array(
        [train_result.rmse for train_result, _, _ in train_results])
    rmse_val = np.array([val_result.rmse for val_result, _, _ in val_results])
    plt.plot(rmse_train)
    plt.plot(rmse_val)
    plt.legend(["rmse train", "rmse val"])
    plt.title("Training errors")
    plt.xlabel("epoch")
    plt.ylabel("error")
    fn = os.path.join(output_directory, f"progress_plot_{epoch}.png")
    plt.savefig(fn)


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
