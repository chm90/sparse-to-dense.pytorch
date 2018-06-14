import torchviz
from models import ResNet
from argparse import ArgumentParser
parser = ArgumentParser(description="plots the networks used in the report")
parser.add_argument(
    "--render-dir",
    type=str,
    help="Where to save rendered files",
    default="network-renders")
from graphviz import Digraph
from typing import List, Any


def label(*args: Any) -> str:
    string = "|".join(f"<f{i}> {arg} " for i, arg in enumerate(args))
    print(string)
    return string


def main():
    args = parser.parse_args()
    import torch
    from torch.autograd import Variable
    mdl = ResNet(50, "deconv3", in_channels=4, image_shape=(192, 256)).cuda()
    x = Variable(torch.randn(1, mdl.in_channels, *mdl.image_shape).cuda())
    y = mdl(x)
    model_graph = torchviz.make_dot(y.mean(), dict(mdl.named_parameters()))
    model_graph.format = "svg"
    model_graph.render("resnet50.gv", "resnet50_render", view=True)


if __name__ == "__main__":
    main()
