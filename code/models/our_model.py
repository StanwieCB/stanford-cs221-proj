import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def conv1x1(input_channels, output_channels, stride=1, padding=0, bias=False):
    return nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, padding=padding, bias=bias)    

def conv3x3(input_channels, output_channels, stride=1, padding=1, bias=False):
    return nn.Sequential(
        nn.ReflectionPad2d(padding),
        nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=0, bias=bias)
    )

def conv5x5(input_channels, output_channels, stride=1, padding=2, bias=False):
    return nn.Sequential(
        nn.ReflectionPad2d(padding),
        nn.Conv2d(input_channels, output_channels, kernel_size=5, stride=stride, padding=0, bias=bias)
    )

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(input_channels, output_channels),
            nn.LeakyReLU(0.1),
            conv3x3(output_channels, output_channels),
            nn.LeakyReLU(0.1),
            conv3x3(output_channels, output_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return F.max_pool2d(self.block(x), kernel_size=2)

class UpBlock(nn.Module):
    def __init__(self, input_channels, output_channels, is_deconv=False, is_out=False):
        super(UpBlock, self).__init__()
        if is_deconv:
            self.up = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingNearest2d(scale_factor=2)
        blocks = [
            conv3x3(input_channels, output_channels),
            nn.LeakyReLU(0.1),
            conv3x3(output_channels, output_channels),
            nn.LeakyReLU(0.1)
        ]
        if is_out:
            blocks += [
                    nn.UpsamplingNearest2d(scale_factor=2),
                    conv3x3(output_channels, output_channels),
                    nn.LeakyReLU(0.1),
                    conv1x1(output_channels, output_channels)
                ]
        self.block = nn.Sequential(*blocks)

    def forward(self, inputs):
        return self.block(self.up(inputs))

class StyleTransferNet221(nn.Module):
    def __init__(self, input_channels=3):
        super(StyleTransferNet221, self).__init__()
        self.style_encoder = nn.Sequential(
            ConvBlock(input_channels, 32),
            ConvBlock(32, 43),
            ConvBlock(43, 57),
            ConvBlock(57, 76),
            ConvBlock(76, 101)
        )
        self.content_encoder = nn.Sequential(
            ConvBlock(input_channels, 32),
            ConvBlock(32, 43),
            ConvBlock(43, 57),
            ConvBlock(57, 76),
            ConvBlock(76, 101)
        )
        self.style_fusion_block = ConvBlock(202, 101)
        self.decoder = nn.Sequential(
            UpBlock(101, 76),
            UpBlock(76, 57),
            UpBlock(57, 43),
            UpBlock(43, 32),
            UpBlock(32, 3, is_out=True)
        )

    # U-Net
    def forward(self, x_c, x_s):
        style_feature = self.style_encoder(x_s)
        content_feature = self.content_encoder(x_c)
        # NCHW
        mixed_feature = torch.cat([style_feature, content_feature], 1)
        out_image = self.decoder(self.style_fusion_block(mixed_feature))
        return style_feature, content_feature, out_image

Vgg16 = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

if __name__ == "__main__":
    test_c = torch.rand(1, 3, 128, 128)
    test_s = torch.rand(1, 3, 128, 128)
    test_vgg = torch.rand(1, 3, 128, 128)
    vgg = Vgg16
    vgg.load_state_dict(torch.load('vgg16.pth'))
    vgg = torch.nn.Sequential(*list(vgg.children())[:31])
    print(vgg(test_vgg).shape)
    net = StyleTransferNet221()
    for feature in net(test_c, test_s):
        print(feature.shape)