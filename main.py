import argparse
import os
import shutil
import time
import sys
import csv
from sys import stdout
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import evaluate
from nyu_dataloader import NYUDataset
from sunrgbd_dataloader import SUNRGBDDataset, inverse_apply_uniform_square_batch
from models import Decoder, ResNet, SKIP_TYPES
from metrics import AverageMeter, Result
import criteria
import utils
from my_model import RGBDRGB

model_names = ['resnet18', 'resnet50', 'my_resnet18']
loss_names = ['l1', 'l2']
data_names = ['nyudepthv2', "SUNRGBD"]
decoder_names = Decoder.names
modality_names = NYUDataset.modality_names
depth_sampling_types = ["sparse", "square"]
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
    default=8,
    type=int,
    help='mini-batch size (default: 8)')
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

fieldnames = [
    'mse', 'rmse', 'rmse_inside', 'rmse_outside', 'absrel', 'lg10', 'mae',
    'delta1', 'delta2', 'delta3', 'data_time', 'gpu_time'
]
best_result = Result()
best_result.set_to_worst()


def main():
    global args, best_result, output_directory, train_csv, test_csv
    args = parser.parse_args()
    dataset = args.data
    if args.modality == 'rgb' and args.num_samples != 0:
        print("number of samples is forced to be 0 when input modality is rgb")
        args.num_samples = 0
    image_shape = (192, 256)  # if "my" in args.arch else (228, 304)

    # create results folder, if not already exists
    output_directory = os.path.join(
        args.output_dir,
        f'{args.data}.modality={args.modality}.arch={args.arch}'
        f'.skip={args.skip_type}.decoder={args.decoder}'
        f'.criterion={args.criterion}.lr={args.lr}.bs={args.batch_size}'
        f'.opt={args.optimizer}')
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
    valdir = os.path.join(args.data, 'val')

    if dataset == "SUNRGBD":
        # only squares implemented currently
        train_dataset = SUNRGBDDataset(
            traindir,
            type="train",
            modality=args.modality,
            output_shape=image_shape)
    else:
        train_dataset = NYUDataset(
            traindir,
            type='train',
            modality=args.modality,
            num_samples=args.num_samples,
            square_width=50,
            output_shape=image_shape)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        sampler=None)
    print("len(train_dataset) =", len(train_dataset))
    if dataset == "SUNRGBD":
        # set batch size to be 1 for validation
        val_dataset = SUNRGBDDataset(
            valdir, type="val", modality=args.modality)
    else:
        val_dataset = NYUDataset(
            valdir,
            type='val',
            modality=args.modality,
            num_samples=args.num_samples,
            square_width=50,
            output_shape=image_shape)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

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
        validate(val_loader, model, checkpoint['epoch'], write_to_file=False)
        return

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
            return

    # create new model
    else:
        # define model
        print("=> creating Model ({}-{}) ...".format(args.arch, args.decoder))
        in_channels = len(args.modality)
        if args.arch == 'resnet50':
            model = ResNet(
                layers=50,
                decoder=args.decoder,
                in_channels=in_channels,
                out_channels=out_channels,
                pretrained=args.pretrained,
                image_shape=image_shape,
                skip_type=args.skip_type)
        elif args.arch == 'resnet18':
            model = ResNet(
                layers=18,
                decoder=args.decoder,
                in_channels=in_channels,
                out_channels=out_channels,
                pretrained=args.pretrained,
                image_shape=image_shape,
                skip_type=args.skip_type)
        elif args.arch == 'my_resnet18':
            model = RGBDRGB(image_shape, batch_size=args.batch_size)
            assert args.modality == "rgbd"
            assert args.skip_connection, "RGBDRGB only supports using skipp connections"

        print("=> model created.")

        adjusting_learning_rate = False
        if args.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay)
        elif args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), weight_decay=args.weight_decay)
            adjusting_learning_rate = True
        else:
            raise Exception("We should never be here")

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
    for epoch in range(args.start_epoch, args.epochs):
        if adjusting_learning_rate:
            adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        result,result_inside, result_outside, img_merge = validate(val_loader, model, epoch)

        # remember best rmse and save checkpoint
        is_best = result.rmse < best_result.rmse
        if is_best:
            epochs_since_best = 0
            best_result = result
            with open(best_txt, 'w') as txtfile:
                txtfile.write(
                    "epoch={}\nmse={:.3f}\nrmse={:.3f}\nrmse_inside={:.3f}\nrmse_outside={:.3f}\nabsrel={:.3f}\nlg10={:.3f}\nmae={:.3f}\ndelta1={:.3f}\nt_gpu={:.4f}\n".
                    format(epoch, result.mse, result.rmse,result_inside.rmse,result_outside.rmse,result.absrel,
                           result.lg10, result.mae, result.delta1,
                           result.gpu_time))
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

        if epochs_since_best > args.early_stop_epochs:
            print("early stopping")
            break


def train(train_loader, model, criterion, optimizer, epoch):
    average_meter = AverageMeter()
    inside_average_meter = AverageMeter()
    outside_average_meter = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, input_square,
            output_square) in enumerate(train_loader):
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
        result = Result()
        output1 = torch.index_select(depth_pred.data, 1,
                                     torch.cuda.LongTensor([0]))
        #assume all squares are of same size
        x_min, x_max, y_min, y_max = output_square[0]
        mask_inside = (slice(None), slice(None), slice(x_min, x_max),
                       slice(y_min, y_max))
        mask_outside = torch.ones_like(output1).byte()
        try:
            mask_outside[mask_inside] = False
        except ValueError:
            pass
        result_inside = Result(mask=mask_inside)
        result_outside = Result(mask=mask_outside)
        result.evaluate(output1, target)
        result_inside.evaluate(output1, target)
        result_outside.evaluate(output1, target)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        inside_average_meter.update(result_inside, gpu_time, data_time,
                                    input.size(0))
        outside_average_meter.update(result_outside, gpu_time, data_time,
                                     input.size(0))
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            #print('=> output: {}'.format(output_directory))
            def print_result(result, result_name, averege_meter):
                average = averege_meter.average()
                stdout.write(
                    f"{result_name}: "
                    f'Train Epoch: {epoch} [{i + 1}/{len(train_loader)}]\t'
                    #f't_Data={data_time:.3f}({average.data_time:.3f}) '
                    #f't_GPU={gpu_time:.3f}({average.gpu_time:.3f}) '
                    f'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                    f'MAE={result.mae:.2f}({average.mae:.2f}) '
                    f'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                    #f'REL={result.absrel:.3f}({average.absrel:.3f}) '
                    #f'Lg10={result.lg10:.3f}({average.lg10:.3f}) \n'
                    '\n')

            print_result(result, "result", average_meter)
            print_result(result_inside, "result_inside", inside_average_meter)
            print_result(result_outside, "result_outside",
                         outside_average_meter)
        break #debug
    avg = average_meter.average()
    avg_inside = inside_average_meter.average()
    avg_outside = outside_average_meter.average()
    with open(train_csv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
            'mse': avg.mse,
            'rmse': avg.rmse,
            'rmse_inside': avg_inside.rmse,
            'rmse_outside': avg_outside.rmse,
            'absrel': avg.absrel,
            'lg10': avg.lg10,
            'mae': avg.mae,
            'delta1': avg.delta1,
            'delta2': avg.delta2,
            'delta3': avg.delta3,
            'gpu_time': avg.gpu_time,
            'data_time': avg.data_time
        })


def validate(val_loader, model, epoch, write_to_file=True):
    average_meter = AverageMeter()
    inside_average_meter = AverageMeter()
    outside_average_meter = AverageMeter()

    # switch to evaluate mode
    model.eval()
    evaluator = evaluate.Evaluator(val_loader.dataset.output_shape)
    end = time.time()
    for i, (input, target, square_input,
            square_output) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        depth_pred = model(input_var)
        evaluator.add_results(depth_pred, target, square_output)
        torch.cuda.synchronize()
        gpu_time = time.time() - end
        # measure accuracy and record loss
        result = Result()
        output1 = torch.index_select(depth_pred.data, 1,
                                     torch.cuda.LongTensor([0]))
        #assume all squares are of same size
        x_min, x_max, y_min, y_max = square_output[0]
        mask_inside = (slice(None), slice(None), slice(x_min, x_max),
                       slice(y_min, y_max))
        mask_outside = torch.ones_like(output1).byte()
        try:
            mask_outside[mask_inside] = False
        except ValueError:
            pass
        result_inside = Result(mask=mask_inside)
        result_outside = Result(mask=mask_outside)
        result.evaluate(output1, target)
        result_inside.evaluate(output1, target)
        result_outside.evaluate(output1, target)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        inside_average_meter.update(result_inside, gpu_time, data_time,
                                    input.size(0))
        outside_average_meter.update(result_outside, gpu_time, data_time,
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
            def print_result(result, result_name, averege_meter):
                average = averege_meter.average()
                stdout.write(
                    f"{result_name}: "
                    f'Train Epoch: {epoch} [{i + 1}/{len(val_loader)}]\t'
                    #f't_Data={data_time:.3f}({average.data_time:.3f}) '
                    #f't_GPU={gpu_time:.3f}({average.gpu_time:.3f}) '
                    f'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                    f'MAE={result.mae:.2f}({average.mae:.2f}) '
                    f'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                    #f'REL={result.absrel:.3f}({average.absrel:.3f}) '
                    #f'Lg10={result.lg10:.3f}({average.lg10:.3f}) \n'
                    '\n')

            print_result(result, "result", average_meter)
            print_result(result_inside, "result_inside", inside_average_meter)
            print_result(result_outside, "result_outside",
                         outside_average_meter)

    average = average_meter.average()
    average_inside = inside_average_meter.average()
    average_outside = outside_average_meter.average()
    gpu_time = average.gpu_time
    print(f'\n*\n'
          f'RMSE={average.rmse:.3f}\n'
          f'RMSE_INSIDE={average_inside.rmse:.3f}\n'
          f'RMSE_OUTSIDE={average_outside.rmse:.3f}\n'
          f'MAE={average.mae:.3f}\n'
          f'Delta1={average.delta1:.3f}\n'
          f'REL={average.absrel:.3f}\n'
          f'Lg10={average.lg10:.3f}\n'
          f't_GPU={gpu_time:.3f}\n')

    if write_to_file:
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'mse': average.mse,
                'rmse': average.rmse,
                'rmse_inside': average_inside.rmse,
                'rmse_outside': average_outside.rmse,
                'absrel': average.absrel,
                'lg10': average.lg10,
                'mae': average.mae,
                'delta1': average.delta1,
                'delta2': average.delta2,
                'delta3': average.delta3,
                'data_time': average.data_time,
                'gpu_time': average.gpu_time
            })
        evaluator.save_plot(
            os.path.join(output_directory, f"evaluation_epoch{epoch}.png"))
    else:
        evaluator.save_plot("evaluation.png")
        print("saved plot to evaluation")
    return average, average_inside, average_outside, img_merge


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    lr = args.lr * (0.1**(epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
