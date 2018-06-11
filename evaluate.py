from math import sqrt
import numpy as np
from numba import njit, float64, types, int64, float32
from dataloaders import center_square

pointT = types.UniTuple(float64, 2)
rangeT = types.UniTuple(float64, 2)
squareT = types.UniTuple(float64, 4)
shapeT = types.UniTuple(int64, 2)
imagesT = float32[:, :, :]


@njit(float64(pointT, squareT))
def point_to_square_distance(point, square):
    x, y = point
    xmin, xmax, ymin, ymax = square
    # check if point is inside the square
    left_of_square = x < xmin
    right_of_square = xmax < x
    above_square = y < ymin
    below_square = ymax < y

    if not (above_square or below_square or left_of_square or right_of_square):
        # inside square
        distance = 0.0
    else:
        if above_square and not (left_of_square or right_of_square):
            distance = ymin - y
        elif below_square and not (left_of_square or right_of_square):
            distance = y - ymax
        elif left_of_square and not (above_square or below_square):
            distance = xmin - x
        elif right_of_square and not (above_square or below_square):
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
    return distance


@njit(float64[:, :](shapeT, squareT), locals={"distance_map": float64[:, :]})
def compute_distance_map(image_shape, square):
    distance_map = np.empty(image_shape)
    w, h = image_shape
    for x in range(w):
        for y in range(h):
            distance_map[x, y] = point_to_square_distance((x, y), square)
    return distance_map


@njit(
    types.UniTuple(float64[:], 5)(int64, rangeT, squareT, imagesT, imagesT),
    locals={
        "errors_flat": float64[:, :],
        "distance_map": float64[:]
    })
def create_histogram(bins, distance_range, square, predictions, targets):
    start, end = distance_range
    width, height = targets.shape[1:]
    image_shape = (width, height)
    distance_map = compute_distance_map(image_shape, square).ravel()
    sort_idxs = np.argsort(distance_map)
    distance_map = distance_map[sort_idxs]
    step = (end - start) / bins
    edges = [start + step * i for i in range(1, bins + 1)]
    assert edges[-1] == end, "last edge must be ending edge"
    predictions_flat = np.empty((predictions.shape[0],
                                 predictions.shape[1] * predictions.shape[2]))
    targets_flat = np.empty((targets.shape[0],
                             targets.shape[1] * targets.shape[2]))
    for i in range(targets.shape[0]):
        predictions_flat[i, :] = predictions[i, ...].ravel()[sort_idxs]
        targets_flat[i, :] = targets[i, ...].ravel()[sort_idxs]

    distances = distance_map.ravel()

    i = 0
    # Go to starting index
    while distances[i] <= start:
        i += 1

    hist_mae = np.empty(len(edges))
    hist_rmse = np.empty(len(edges))
    hist_rel = np.empty(len(edges))
    hist_delta1 = np.empty(len(edges))
    for e, edge in enumerate(edges):
        #1. accumulate indexes for edge
        first_i = i
        while i < len(distances) and distances[i] <= edge:
            i += 1

        assert i != first_i, "must have taken some i"

        n_idxs = i - first_i
        #2. extract values for every error maps
        edge_predictions = np.empty(predictions_flat.shape[0] * n_idxs)
        edge_targets = np.empty(targets_flat.shape[0] * n_idxs)
        for j in range(targets_flat.shape[0]):
            edge_predictions[j * n_idxs:(
                j + 1) * n_idxs] = predictions_flat[j, first_i:i]
            edge_targets[j * n_idxs:(
                j + 1) * n_idxs] = targets_flat[j, first_i:i]

        #3. compute metrics for bin
        edge_abs_diff = np.abs(edge_predictions - edge_targets)
        maxRatio = np.maximum(edge_predictions / edge_targets,
                              edge_targets / edge_predictions)
        hist_mae[e] = edge_abs_diff.mean()
        hist_rel[e] = (edge_abs_diff / edge_targets).mean()
        hist_rmse[e] = sqrt((edge_abs_diff**2).mean())
        hist_delta1[e] = (maxRatio < 1.25).astype(float64).mean()

    return np.array(edges), hist_mae, hist_rel, hist_rmse, hist_delta1


class Evaluator(object):
    def __init__(self, output_shape, square_width):
        self._predictions = []
        self._targets = []
        self._distances = []
        self.output_shape = output_shape
        self.square = center_square(output_shape, square_width, square_width)

    @property
    def output_size(self):
        return np.cumprod(self.output_shape)

    def add_results(self, outputs, labels):
        valid_mask = labels > 0
        outputs = outputs.clone()
        labels = labels.clone()
        outputs[~valid_mask] = np.nan
        labels[~valid_mask] = np.nan
        self._predictions.append(outputs.squeeze().cpu().data.numpy())
        self._targets.append(labels.squeeze().cpu().data.numpy())

    def draw_plots(self):
        #Interleave sorted error arrays to create one long sorted error array

        edges, hist_mae, hist_rel, hist_rmse, hist_delta1 = create_histogram(
            50, (0., 100.), self.square, np.asarray(self._predictions),
            np.asarray(self._targets))
        import matplotlib
        if matplotlib.get_backend() != "Agg":
            matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(edges, hist_mae)
        plt.plot(edges, hist_rel)
        plt.plot(edges, hist_rmse)
        plt.plot(edges, hist_delta1)
        plot_labels = ["mae (m)", "rel", "rmse", "delta1"]
        plt.xlabel("distance (px)")
        plt.ylabel("error")
        plt.legend(plot_labels)

    def plot(self):
        import matplotlib
        if matplotlib.get_backend() != "Agg":
            matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        self.draw_plots()
        plt.show()

    def save_plot(self, path):
        self.draw_plots()
        import matplotlib
        if matplotlib.get_backend() != "Agg":
            matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        plt.savefig(path)
