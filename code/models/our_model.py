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

class GramMatrix(nn.Module):
    def forward(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

class StyleTransferNet221(nn.Module):
    def __init__(self, vgg):
        super(StyleTransferNet221, self).__init__()
        self.style_encoder = vgg
        self.content_encoder = vgg
        self.style_fusion_block = nn.Sequential(
            ConvBlock(1024, 512),
            ConvBlock(512, 256),
            conv3x3(256, 256),
            nn.LeakyReLU(0.1)
        )
        self.decoder = nn.Sequential(
            UpBlock(256, 128),
            UpBlock(128, 64),
            UpBlock(64, 32),
            UpBlock(32, 3, is_out=True)
        )

        ec_layers = list(self.content_encoder.children())
        self.ec_1 = nn.Sequential(*ec_layers[:4])  # input -> relu1_1
        self.ec_2 = nn.Sequential(*ec_layers[4:11])  # relu1_1 -> relu2_1
        self.ec_3 = nn.Sequential(*ec_layers[11:18])  # relu2_1 -> relu3_1
        self.ec_4 = nn.Sequential(*ec_layers[18:31])  # relu3_1 -> relu4_1

        # fix the content encoder
        for name in ['ec_1', 'ec_2', 'ec_3', 'ec_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        es_layers = list(self.style_encoder.children())
        self.es_1 = nn.Sequential(*es_layers[:4])  # input -> relu1_1
        self.es_2 = nn.Sequential(*es_layers[4:11])  # relu1_1 -> relu2_1
        self.es_3 = nn.Sequential(*es_layers[11:18])  # relu2_1 -> relu3_1
        self.es_4 = nn.Sequential(*es_layers[18:31])  # relu3_1 -> relu4_1

        self.gram = GramMatrix()
        self.loss = nn.MSELoss()

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_style(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'es_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode_content(self, input):
        for i in range(4):
            input = getattr(self, 'ec_{:d}'.format(i + 1))(input)
        return input

    def encoder_basic(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'ec_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def get_Lc(self, input, target):
        assert (input.shape == target.shape)
        assert (target.requires_grad is False)
        return self.loss(input, target)

    def get_Ls(self, input, target):
        # assert (target.requires_grad is False)
        loss = 0.
        for i, _ in enumerate(input):
            assert (input[i].shape == target[i].shape)
            loss += self.loss(self.gram(input[i]), self.gram(target[i])) 
        return loss
    
    def get_Lb(self, input, target):
        assert (input.shape == target.shape)
        assert (target.requires_grad is False)
        return self.loss(input, target)

    # U-Net
    def forward(self, x_c, x_s):
        style_feature = self.encode_style(x_s)
        content_feature = self.encode_content(x_c)
        # NCHW
        mixed_feature = torch.cat([style_feature[-1], content_feature], 1)
        out_image = self.decoder(self.style_fusion_block(mixed_feature))
        out_image_f = self.encoder_basic(out_image)
        
        # print(out_image.shape, out_image_f[-1].shape, content_feature.shape, style_feature[-1].shape)
        loss_c = self.get_Lc(out_image_f[-1], content_feature)
        loss_s = self.get_Ls(out_image_f, style_feature)

        style_feature_ic = self.encode_style(x_c)
        out_image_ic = self.decoder(self.style_fusion_block(
            torch.cat([style_feature_ic[-1], content_feature], 1)))
        
        loss_b = self.get_Lb(out_image_ic, x_c)

        return out_image, loss_c, loss_s, loss_b

if __name__ == "__main__":
    test_c = torch.rand(1, 3, 64, 64)
    test_s = torch.rand(1, 3, 64, 64)
    vgg = Vgg16
    vgg.load_state_dict(torch.load('vgg16.pth'))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    net = StyleTransferNet221(vgg)

    out_image, loss_c, loss_s, loss_b = net(test_c, test_s)
    print(out_image.shape)
    print(loss_c)
    print(loss_s)
    print(loss_b)