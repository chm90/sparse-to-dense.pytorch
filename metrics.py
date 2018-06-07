import torch
import math
import numpy as np


def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / math.log(10)


class MaskedResult(object):
    def __init__(self, mask=Ellipsis):
        self.mask = mask
        self.result = Result()
        self.result_inside = Result()
        self.result_outside = Result()

    @property
    def results(self):
        return [self.result,self.result_inside,self.result_outside]

    def set_to_worst(self):
        for result in self.results:
            result.set_to_worst()

    def evaluate(self, output, target):
        mask_outside = torch.ones_like(output).byte()
        try:
            mask_outside[self.mask] = 0  #False
        except RuntimeError:
            pass
        try:
            output_inside = output[self.mask]
            target_inside = target[self.mask]
        except RuntimeError:
            output_inside = output
            target_inside = target
        try:
            output_outside = output[mask_outside]
            target_outside = target[mask_outside]
        except RuntimeError:
            output_outside = output
            target_outside = target

        self.result.evaluate(output,target)
        self.result_inside.evaluate(output_inside,target_inside)
        self.result_outside.evaluate(output_outside,target_outside)
        
class Result(object):
    def __init__(self):
        self.irmse, self.imae = 0, 0
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.absrel, self.lg10 = 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0
        self.name = ""

    def set_to_worst(self):
        self.irmse, self.imae = np.inf, np.inf
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf
        self.absrel, self.lg10 = np.inf, np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

    def update(self, irmse, imae, mse, rmse, mae, absrel, lg10, delta1, delta2,
               delta3, gpu_time, data_time):
        self.irmse, self.imae = irmse, imae
        self.mse, self.rmse, self.mae = mse, rmse, mae
        self.absrel, self.lg10 = absrel, lg10
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3
        self.data_time, self.gpu_time = data_time, gpu_time

    def evaluate(self, output, target):
        valid_mask = target > 0
        output = output[valid_mask]
        target = target[valid_mask]

        abs_diff = (output - target).abs()

        self.mse = (torch.pow(abs_diff, 2)).mean()
        self.rmse = math.sqrt(self.mse)
        self.mae = abs_diff.mean()
        self.lg10 = (log10(output) - log10(target)).abs().mean()
        self.absrel = (abs_diff / target).mean()

        maxRatio = torch.max(output / target, target / output)
        self.delta1 = (maxRatio < 1.25).float().mean()
        self.delta2 = (maxRatio < 1.25**2).float().mean()
        self.delta3 = (maxRatio < 1.25**3).float().mean()
        self.data_time = 0
        self.gpu_time = 0

        inv_output = 1 / output
        inv_target = 1 / target
        abs_inv_diff = (inv_output - inv_target).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = abs_inv_diff.mean()

        #fix for bug: [PyTorch] Error when printing tensors containing large values #6339
        self.mse = float(self.mse)
        self.mae = float(self.mae)
        self.lg10 = float(self.lg10)
        self.absrel = float(self.absrel)
        self.delta1 = float(self.delta1)
        self.delta2 = float(self.delta2)
        self.delta3 = float(self.delta3)
        self.imae = float(self.imae)

    def __str__(self):
        return (f"{self.name}:\n"
                f"\tmse = {self.mse}\n"
                f"\tmae = {self.mae}\n"
                f"\tlg10 = {self.lg10}\n"
                f"\tabsrel = {self.absrel}\n"
                f"\trmse = {self.rmse}\n"
                f"\tdelta1 = {self.delta1}\n"
                f"\tdelta2 = {self.delta2}\n"
                f"\tdelta3 = {self.delta3}\n")


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0

        self.sum_irmse, self.sum_imae = 0, 0
        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_absrel, self.sum_lg10 = 0, 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0
        self.sum_data_time, self.sum_gpu_time = 0, 0

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n
        self.sum_irmse += n * result.irmse
        self.sum_imae += n * result.imae
        self.sum_mse += n * result.mse
        self.sum_rmse += n * result.rmse
        self.sum_mae += n * result.mae
        self.sum_absrel += n * result.absrel
        self.sum_lg10 += n * result.lg10
        self.sum_delta1 += n * result.delta1
        self.sum_delta2 += n * result.delta2
        self.sum_delta3 += n * result.delta3
        self.sum_data_time += n * data_time
        self.sum_gpu_time += n * gpu_time

    def average(self):
        avg = Result()
        avg.update(self.sum_irmse / self.count, self.sum_imae / self.count,
                   self.sum_mse / self.count, self.sum_rmse / self.count,
                   self.sum_mae / self.count, self.sum_absrel / self.count,
                   self.sum_lg10 / self.count, self.sum_delta1 / self.count,
                   self.sum_delta2 / self.count, self.sum_delta3 / self.count,
                   self.sum_gpu_time / self.count,
                   self.sum_data_time / self.count)
        return avg
