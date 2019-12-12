import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import models.our_model as model

def test(args):
    content_transform = transforms.Compose([transforms.Resize(args.content_size),
                                    transforms.ToTensor()])
    style_transform = transforms.Compose([transforms.Resize(args.content_size),
                                    transforms.ToTensor()])

    vgg = model.Vgg16
    # vgg.load_state_dict(torch.load(args.encoder))
    vgg = torch.nn.Sequential(*list(vgg.children())[:31])
    style_model = model.StyleTransferNet221(vgg)

    print('Initializing weight from {}.'.format(args.decoder))
    if args.cuda:
        checkpoint = torch.load(args.decoder)
    else:
        checkpoint = torch.load(args.decoder, map_location=lambda storage, loc: storage)
    style_model.load_state_dict(checkpoint["model_state"])

    if args.cuda:
        style_model.cuda()

    assert(os.path.exists(args.content_folder))
    assert(os.path.exists(args.style_folder))
    content_images = os.listdir(args.content_folder)
    style_images = os.listdir(args.style_folder)
    for content_path in content_images:
        for style_path in style_images:

            content = content_transform(Image.open(os.path.join(args.content_folder, content_path)))
            style = style_transform(Image.open(os.path.join(args.style_folder, style_path)))

            if args.cuda:
                content = content.cuda()
                style = style.cuda()

            content = content.unsqueeze(0)
            style = style.unsqueeze(0)
            with torch.no_grad():
                output= style_model.style_transfer(content, style)

            output = output.cpu()

            if not os.path.exists(args.result_folder):
                os.mkdir(args.result_folder)
                
            output_path = os.path.join(args.result_folder, '{:s}_stylized_{:s}{:s}'.format(
                content_path.split('.')[0], style_path.split('.')[0], args.save_ext))
            print(output_path)
            save_image(output, output_path)

if __name__ == "__main__":
    test_parser = argparse.ArgumentParser()
    test_parser.add_argument("--content-folder", type=str, default="./images/test-content",
                                    help="path to content folder"
                                    "containing another folder with all the training images")
    test_parser.add_argument("--style-folder", type=str, default="./images/test-style",
                                    help="path to style folder")
    test_parser.add_argument("--result-folder", type=str, default='./results',
                                    help="path to save validation results")
    test_parser.add_argument("--encoder", type=str, default="models/vgg16.pth",
                                    help="encoder weight to load")
    test_parser.add_argument("--decoder", type=str, default="models/train_saved/",
                                    help="decoder weight to load")
    test_parser.add_argument("--content-size", type=int, default=512,
                                    help="size of content images, default is 512 X 512")
    test_parser.add_argument("--style-size", type=int, default=512,
                                    help="size of style images, default is 512 X 512")
    test_parser.add_argument("--cuda", type=int, default=1, 
                                    help="set it to 1 for running on GPU, 0 for CPU")
    test_parser.add_argument("--save-ext", type=str, default='.jpg', 
                                    help="output image extension")
    args = test_parser.parse_args()
    print(args)
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("ERROR: cuda is not available, try running on CPU")

    test(args)