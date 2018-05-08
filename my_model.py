import argparse

import torch
from matplotlib import pyplot as plt
from torch import nn, cat
from torch.autograd import Variable

from torchvision.models import resnet

class deConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(deConv,self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=2,
            padding=1,
            output_padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class RGBDRGB(nn.Module):
    def __init__(self,
                 image_shape,
                 batch_size=64
                 ):
        super(RGBDRGB, self).__init__()
        feature_extractor = resnet.resnet18(pretrained=True)
        self.batch_size = batch_size
        self.add_module("fe", feature_extractor)
        self.image_shape = image_shape
        self.conv_fe = nn.Conv2d(512, 256, (1, 1), bias=False) #for resnet18
        # self.conv_fe = nn.Conv2d(2048,1024, (1, 1), bias=False) #for resnet 50

        # self.bn_fe = nn.BatchNorm2d(1024) #resnet 50
        self.bn_fe = nn.BatchNorm2d(256) # resnet 18
        #        self.up_proj1 = UpProj_Block(256, 128, self.batch_size)
        #        self.up_proj2 = UpProj_Block(128, 64, self.batch_size)
        #        self.up_proj3 = UpProj_Block(64, 32, self.batch_size)
        #        self.up_proj4 = UpProj_Block(32, 16, self.batch_size)
        # for resnet 18
        self.up_proj1 = deConv(256, 128)
        self.up_proj2 = deConv(128, 64)
        self.up_proj3 = deConv(192, 32)
        self.up_proj4 = deConv(96, 16)
        # for resnet 50
        # self.up_proj1 = deConv(1024,512)
        # self.up_proj2 = deConv(512,256)
        # self.up_proj3 = deConv(256 + 512,128)
        # self.up_proj4 = deConv(128 + 256,64)

        # self.conv_last = nn.Conv2d(64 + 64, 1, (3, 3)) #resnet 50 
        self.conv_last = nn.Conv2d(80,1,(3,3)) # resnet 50
        self.relu_last = nn.ReLU(inplace=True)
        self.bilinear = nn.Upsample(size=(image_shape),mode="bilinear")
        self.make_4_ch()

    def make_4_ch(self):
        self.fe.conv1 = nn.Conv2d(
            4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        old_inplanes = self.fe.inplanes
        self.fe.inplanes = 64
        self.fe.layer1 = self.fe._make_layer(resnet.BasicBlock,64,2)
        # self.fe.layer1 = self.fe._make_layer(resnet.Bottleneck, 64, 3) #resnet 50
        self.fe.inplanes = old_inplanes

    def forward(self, x):

        # using same feature extraction and decoder network as in
        # "Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image"
        #print(f"in: {x.shape}")
        x = self.fe.conv1(x)
        x_conv1 = x
        #print(f"conv1 out: {x.shape}")
        x = self.fe.bn1(x)
        x = self.fe.relu(x)
        x = self.fe.maxpool(x)
        #print(f"maxpool out: {x.shape}")
        x = self.fe.layer1(x)
        x_layer1 = x
        #print(f"layer1 out: {x.shape}")
        x = self.fe.layer2(x)
        x_layer2 = x
        #print(f"layer2 out: {x.shape}")
        x = self.fe.layer3(x)
        #print(f"layer3 out: {x.shape}")
        x = self.fe.layer4(x)
        # print(f"layer4 out: {x.shape}")

        x = self.conv_fe(x)
        x = self.bn_fe(x)
        x = self.fe.relu(x)
        # print(f"encoder out: {x.shape}")

        x = self.up_proj1(x)
        #print(f"upProj1 out:{x.shape}")
        x = self.up_proj2(x)
        #print(f"upProj2 out:{x.shape}")
        x = cat([x,x_layer2],dim=1)
        x = self.up_proj3(x)
        #print(f"upProj3 out:{x.shape}")
        x = cat([x,x_layer1],dim=1)
        
        x = self.up_proj4(x)
        #print(f"upProj4 out:{x.shape}")
        x = cat([x,x_conv1],dim=1)
        #print(f"concat last out: {x.shape}")
        x = self.conv_last(x)
        x = self.relu_last(x)
        #print(f"decoder out: {x.shape}")
        x = self.bilinear(x)
        return x


def main():
    parser = argparse.ArgumentParser(
        description="test the model using random input")
    parser.add_argument(
        "--device",
        "-d",
        help="the index of the graphics device to run the test on.",
        type=int)
    model = RGBDRGB()
    model.make_4_ch()
    trash_in = Variable(torch.randn(64, 4, 192, 256))
    #trash_in = trash_in.cuda(device=args.device)
    #model = model.cuda(device=args.device)
    trash_in = trash_in.cpu()
    model = model.cpu()
    #model.cuda(device=args.device)
    trash_out = model.forward(trash_in)
    print(model)
    print("trash_out.shape =", trash_out.shape)
    plt.figure()
    trash_out_np = trash_out.data.numpy()
    trash_out_np = trash_out_np.transpose()
    plt.imshow(trash_out_np[..., 0].squeeze())
    plt.show()


if __name__ == "__main__":
    main()
