from argparse import ArgumentParser
import torch
from torch import nn
from matplotlib import pyplot as plt
from matplotlib import gridspec
from torch.autograd import Variable
from math import ceil, floor
from typing import Iterator
parser = ArgumentParser(
    description=
    "plots images showing the effect of different transposed convolutions")

def plot_convt_output(X,
                      ax,
                      out_channels=1,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      output_padding=1):
    in_channels = X.shape[1]
    convt = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                               padding, output_padding)
    Y = convt(X)
    ax.imshow(Y[0, 0, ...].data)
    ax.set_title(
        f"Y, ks: {kernel_size}, s: {stride}, p: {padding}, op: {output_padding}"
    )
    ax.set_xlabel(f"{Y.shape[2]} px")
    ax.set_ylabel(f"{Y.shape[3]} px")
    ax.set_xticklabels([])
    ax.set_yticklabels([])





def main():
    args = parser.parse_args()

    x_width = 16
    x_step = 4
    X = Variable(torch.zeros(1, 1, x_width, x_width))
    X[:, :, ::x_step, ::x_step] = 1
    params = [{
        "padding": 0,
        "output_padding": 0,
        "stride": 1
    },{
        "padding": 0,
        "output_padding": 0,
        "stride": 2
    } 
    ,{
        "stride": 2,
        "padding": 1,
        "output_padding": 0
    }, {
        "stride": 2,
        "padding": 1,
        "output_padding": 1
    }]

    def subplot_specsi(n_plots : int) -> Iterator[gridspec.SubplotSpec]:
        cols = 3
        rows = ceil(n_plots / cols)
        last_row_cols = cols - (cols * rows - n_plots)
        gs = gridspec.GridSpec(rows,cols * 2)
        for row in range(rows - 1):
            for col in range(cols):
                col_grid = col * 2
                col_slice = slice(col_grid,col_grid + 2)
                subplotspec = gs[row,col_slice]
                yield subplotspec
        if last_row_cols == 1:
            col_ = floor(cols / 2)
            yield gs[-1,col_:col_ + 2]
        elif last_row_cols == 2:
            col1_slice = slice(1,3)
            col2_slice = slice(3,5)
            yield gs[-1,col1_slice]
            yield gs[-1,col2_slice]
        elif last_row_cols == 3:
            raise NotImplementedError("Implement meeeee")

    subplotspecsi = subplot_specsi(len(params) + 1)
    ax1 = plt.subplot(next(subplotspecsi))
    ax1.imshow(X[0, 0, ...].data)
    ax1.set_title("X")
    ax1.set_xlabel(f"{X.shape[2]} px")
    ax1.set_ylabel(f"{X.shape[3]} px")
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    for conv_params, subplot_spec in zip(params,subplotspecsi):
        ax = plt.subplot(subplot_spec)
        plot_convt_output(X, ax, **conv_params)

    plt.show()

if __name__ == "__main__":
    main()
