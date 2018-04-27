from math import sqrt
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
from typing import Tuple
import numpy as np
from numba import njit, float64, types
from sys import stdout
from sunrgbd_dataloader import square_providers_by_name, no_square
import torch
#   from utils import load_checkpoint, epochs_iteratior

@njit
def compute_errors_and_distances(error_maps,
                                 squares_outputs) -> Tuple[np.array, np.array]:
    errors = np.zeros((error_maps.shape[1:]))
    distances = np.zeros((error_maps.shape[1:]))
    compute_dist_map = False
    im_width, im_height = error_maps[0, 0].shape
    n = squares_outputs.shape[0]
    for i in range(n):
        square, error_map = squares_outputs[i], error_maps[i]
        if compute_dist_map:
            distance_map = np.zeros((im_width, im_height))
        for x in range(im_width):
            for y in range(im_height):
                xmin, xmax, ymin, ymax = square
                # check if point is inside the square
                left_of_square = x < xmin
                right_of_square = xmax < x
                above_square = y < ymin
                below_square = ymax < y

                if not (above_square or below_square or left_of_square
                        or right_of_square):
                    # inside square
                    distance = 0.0
                else:
                    if above_square and not (left_of_square
                                             or right_of_square):
                        distance = ymin - y
                    elif below_square and not (left_of_square
                                               or right_of_square):
                        distance = y - ymax
                    elif left_of_square and not (above_square or below_square):
                        distance = xmin - x
                    elif right_of_square and not (above_square
                                                  or below_square):
                        distance = x - xmax
                    else:
                        if above_square and left_of_square:
                            closest_point = (xmin, ymin)
                        elif above_square and right_of_square:
                            closest_point = (xmax, ymin)
                        elif below_square and left_of_square:
                            closest_point = (xmin, ymax)
                        elif below_square and right_of_square:
                            closest_point = (xmax, ymax)
                        else:
                            raise Exception("We should never be here")

                        x_close, y_close = closest_point
                        x_dist, y_dist = x - x_close, y - y_close
                        distance = sqrt(x_dist**2 + y_dist**2)
                if compute_dist_map:
                    distance_map[x, y] = distance
                error = error_map[0, x, y]
                distances[i, x, y] = distance
                assert distance >= 0
                errors[i, x, y] = error

    return errors.flatten(), distances.flatten()
    #fig = plt.figure()
    #cax = plt.imshow(distance_map)
    #fig.colorbar(cax)
    #plt.show(


@njit(
    types.Tuple((float64[:], float64[:], float64[:]))(float64[:], float64[:],
                                                      float64[:]),
    locals={"distance_min": float64})
def error_statistics_at_distance(
        distances: np.ndarray, errors: np.ndarray, distance_edges: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    assert len(distances) == len(errors)
    n_items = len(distances)
    n_bins = len(distance_edges) - 1
    distance_min, distance_max = distance_edges[0], distance_edges[-1]
    print("sorting")
    sort_idxs = distances.argsort()
    distances = distances[sort_idxs]
    errors = errors[sort_idxs]
    #sorted_distances_errors = sorted(
    #    zip(distances, errors), key=lambda d_e: d_e[0])
    j = 0
    lower_bound = distance_edges[j]
    upper_bound = distance_edges[j + 1]
    #errors_in_bin: List[List[float]] = [[0.0]]
    first_item_idx = 0
    print("entering loop")
    error_means = np.empty((n_bins, ))
    error_stds = np.empty((n_bins, ))
    error_medians = np.empty((n_bins, ))

    while distances[first_item_idx] <= distance_min:
        first_item_idx += 1
    #   print("looping first_item_idx =", first_item_idx, "distance_min =",
    #          distance_min)

    errors_in_bin_j = np.zeros((n_items, ))
    n_errors_in_bin_j = 0
    print(errors.shape)
    print(n_items)
    print(errors_in_bin_j.shape)
    #raise Exception("asdasd")
    for i in range(first_item_idx, n_items):
        distance, error = distances[i], errors[i]
        assert lower_bound <= distance
        if distance <= upper_bound:
            errors_in_bin_j[n_errors_in_bin_j] = error
            n_errors_in_bin_j += 1
        else:
            print("looping, item =", i, ", bin =", j)
            # is above upper bound
            errors_in_bin_j_tmp = errors_in_bin_j[:n_errors_in_bin_j]
            error_means[j] = sqrt((errors_in_bin_j_tmp **2).mean())
            error_stds[j] = errors_in_bin_j_tmp.std()
            error_medians[j] = np.median(errors_in_bin_j_tmp)
            #check if we are at the end
            if distance_max <= distance:
                break
            # else advance to next bin
            j += 1
            n_errors_in_bin_j = 0
            lower_bound = distance_edges[j]
            upper_bound = distance_edges[j + 1]


#    else:
# if we did reach the end without breaking,
# we need to fill upp the remaining part of errors_in_bin list
#while len(errors_in_bin) < n_bins:
#    errors_in_bin.append(
#        [np.nan, np.nan])  # two nans supresses warnings in np.nanstd
    assert len(error_means) == n_bins
    assert len(error_means) == len(error_stds)
    return error_means, error_stds, error_medians

class Evaluator(object):
    def __init__(self, output_size):
        self._errors = []
        self._distances = []
        self.output_size = output_size

    def add_results(self, outputs, labels, squares_output):
        error_maps = torch.abs(outputs.data - labels).cpu().numpy()
        squares_output = squares_output.cpu().numpy()

        errors, distances = compute_errors_and_distances(
            error_maps, squares_output)
        self._errors.append(errors)
        self._distances.append(distances)

    def draw_plots(self):
        distances = np.array(self._distances).flatten()
        errors = np.array(self._errors).flatten()

        print("errors.shape =", errors.shape)
        print("distance.shape", distances.shape)
        plot_labels = ["mean", "meadian"]
        with plt.xkcd():
            print("calling hist2d ...")
            h, xedges, yedges, ax = plt.hist2d(
                distances,
                errors,
                bins=50,
                range=[[1, max(self.output_size) / 2], [0, 1]],
                normed=True)
            error_means, error_stds, error_median = error_statistics_at_distance(
                distances, errors, xedges)
            print("len(error_means) =",len(error_means))
            print("len(error_median) =",len(error_median))
            plt.plot(xedges[1:], error_means)
            plt.plot(xedges[1:], error_median)
            plot_labels = ["mean", "meadian"]
            plt.legend(plot_labels)

    def plot(self):
        self.draw_plots()
        plt.show()

    def save_plot(self, path):
        self.draw_plots()
        plt.savefig(path)


def main(args) -> None:
    model = RGBDRGB()
    model.make_4_ch()
    checkpoint = load_checkpoint(args.checkpoint)
    square_provider_str = checkpoint["square_provider"]
    square_provider = square_providers_by_name[square_provider_str]
    model.load_state_dict(checkpoint["state_dict"])
    model = model.cuda(device=args.device)
    dataset = SUNRGBD(args.data_path, square_provider=square_provider)
    data_loader = DataLoader(dataset, batch_size=args.batch_size)
    model.eval()

    def set_device(obj):
        return obj.cpu() if args.device is None else obj.cuda(
            device=args.device)

    losses_l1 = torch.zeros(len(data_loader))
    losses_l2 = torch.zeros(len(data_loader))

    im_width, im_height = dataset.output_size
    distances = [0.0]
    errors = [0.0]
    l1lossfn = L1Loss()
    l2lossfn = MSELoss()
    for epoch, mini_batch, features, labels, squares_input, squares_output in epochs_iteratior(
            data_loader, args.device, 1):
        outputs = model(features)
        l1_loss = l1lossfn(outputs, labels).cpu()
        l2_loss = l2lossfn(outputs, labels).cpu()
        print(l1_loss)
        losses_l1[mini_batch] = l1_loss.data[0]
        losses_l2[mini_batch] = l2_loss.data[0]

        error_maps = torch.abs(outputs - labels).data.cpu().numpy()
        squares_output = squares_output.cpu().numpy()
        compute_errors_and_distances(error_maps, squares_output, distances,
                                     errors)
        stdout.write(f"\repoch: {epoch} batch: {mini_batch}")
    distances.pop(0)
    errors.pop(0)
    print()
    print("l1 loss =", losses_l1.mean())
    print("l2 loss =", losses_l2.mean())

    with plt.xkcd():
        h, xedges, yedges, ax = plt.hist2d(
            distances,
            errors,
            bins=50,
            range=[[1, max(dataset.output_size) / 2], [0, 1]],
            normed=True)
        error_means, error_stds, error_median = error_statistics_at_distance(
            distances, errors, xedges)
        plt.plot(xedges[1:], error_means)
        plt.plot(xedges[1:], error_median)
        plot_labels = ["mean", "meadian"]
        if args.no_depth_cp is not None:
            model.cpu()
            plot_labels.append("no depth")
            no_depth_checkpoint = load_checkpoint(args.no_depth_cp)
            no_depth_model = RGBDRGB()
            no_depth_model.make_4_ch()
            no_depth_model.load_state_dict(no_depth_checkpoint["state_dict"])
            no_depth_model = set_device(no_depth_model)
            no_depth_model.eval()
            errors = []
            dataset.square_provider = no_square
            for epoch in range(args.epochs):
                for i, mini_batch in enumerate(data_loader):
                    features, labels, squares_input, squares_output = mini_batch
                    n = features.shape[0]
                    if n != args.batch_size:
                        continue
                    features, labels = Variable(
                        set_device(features), volatile=True), Variable(
                            set_device(labels), volatile=True)
                    outputs = no_depth_model(features)
                    loss = loss_fn(outputs, labels).cpu()
                    errors.append(loss.mean().data.numpy()[0])
            mean_error = np.mean(errors)
            print("no depth mean_l1_error =", mean_error)
            print("no depth mean_l2_error =", mean_error)

            plt.plot([0, max(dataset.output_size) / 2], [mean_error] * 2)
        plt.legend(plot_labels)
        # error_stds_scaled = error_stds / 10
        # plt.plot(xedges[1:], error_means + error_stds_scaled)
        # plt.plot(xedges[1:], error_means - error_stds_scaled)

        plt.colorbar(ax)
        plt.title("pixelwise error and distance to measured depth density")
        plt.xlabel("distance (pixels)")
        plt.ylabel("error (absolute)")
        plt.show()


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description=
        "evaluates the given model on a test set using various metrics")
    parser.add_argument(
        "--data-path",
        "-dp",
        help="path to the SUNRGBD dataset to test with",
        type=str)
    parser.add_argument(
        "--checkpoint",
        "-cp",
        help="directory to the model checkpoint to evaluate",
        type=str)
    parser.add_argument(
        "--no-depth-cp",
        help="path to checkpoint of a chackpoint trained with no depth",
        type=str,
        default=None)
    parser.add_argument(
        "--batch-size",
        "-bs",
        help="batch size to use during evaluation",
        default=16,
        type=int)
    parser.add_argument(
        "--device",
        "-dev",
        help=
        "if specified the gpu device to use for evaluation. cpu if not specified",
        default=None,
        type=int)
    parser.add_argument(
        "--epochs",
        "-ep",
        help="number of full passes through the dataset.",
        default=1,
        type=int)
    args = parser.parse_args()
    from model import RGBDRGB
    from SUNRGBD import SUNRGBD
    import torch
    from torch.utils.data import DataLoader
    from torch.autograd import Variable
    from train import loss_fn
    from torch.nn import L1Loss, MSELoss

    main(args)
