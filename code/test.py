##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


import os
import sys
import time
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

from utils import utils
from model import Net, Vgg16

from global_var import Options



def optimize(args):
    """    Gatys et al. CVPR 2017
    ref: Image Style Transfer Using Convolutional Neural Networks
    """
    # load the content and style target
    content_image = utils.tensor_load_rgbimage(args.content_image, size=args.content_size, keep_asp=True)
    content_image = content_image.unsqueeze(0)
    content_image = Variable(utils.preprocess_batch(content_image), requires_grad=False)
    content_image = utils.subtract_imagenet_mean_batch(content_image)
    style_image = utils.tensor_load_rgbimage(args.style_image, size=args.style_size)
    style_image = style_image.unsqueeze(0)    
    style_image = Variable(utils.preprocess_batch(style_image), requires_grad=False)
    style_image = utils.subtract_imagenet_mean_batch(style_image)

    # load the pre-trained vgg-16 and extract features
    vgg = Vgg16()
    utils.init_vgg16(args.vgg_model_dir)
    vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, "vgg16.weight")))
    if args.cuda:
        content_image = content_image.cuda()
        style_image = style_image.cuda()
        vgg.cuda()
    features_content = vgg(content_image)
    f_xc_c = Variable(features_content[1].data, requires_grad=False)
    features_style = vgg(style_image)
    gram_style = [utils.gram_matrix(y) for y in features_style]
    # init optimizer
    output = Variable(content_image.data, requires_grad=True)
    optimizer = Adam([output], lr=args.lr)
    mse_loss = torch.nn.MSELoss()
    # optimizing the images
    tbar = trange(args.iters)
    for e in tbar:
        utils.imagenet_clamp_batch(output, 0, 255)
        optimizer.zero_grad()
        features_y = vgg(output)
        content_loss = args.content_weight * mse_loss(features_y[1], f_xc_c)

        style_loss = 0.
        for m in range(len(features_y)):
            gram_y = utils.gram_matrix(features_y[m])
            gram_s = Variable(gram_style[m].data, requires_grad=False)
            style_loss += args.style_weight * mse_loss(gram_y, gram_s)

        total_loss = content_loss + style_loss
        total_loss.backward()
        optimizer.step()
        tbar.set_description(total_loss.data.cpu().numpy()[0])
    # save the image    
    output = utils.add_imagenet_mean_batch(output)
    utils.tensor_save_bgrimage(output.data[0], args.output_image, args.cuda)



def check_paths(args):
    try:
        if not os.path.exists(args.vgg_model_dir):
            os.makedirs(args.vgg_model_dir)
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def evaluate(args):
    content_image = utils.tensor_load_rgbimage(args.content_image, size=args.content_size, keep_asp=True)
    content_image = content_image.unsqueeze(0)
    style = utils.tensor_load_rgbimage(args.style_image, size=args.style_size)
    style = style.unsqueeze(0)    
    style = utils.preprocess_batch(style)

    style_model = Net(ngf=args.ngf)
    model_dict = torch.load(args.model)
    model_dict_clone = model_dict.copy()
    for key, value in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]
    style_model.load_state_dict(model_dict, False)

    if args.cuda:
        style_model.cuda()
        content_image = content_image.cuda()
        style = style.cuda()

    style_v = Variable(style)

    content_image = Variable(utils.preprocess_batch(content_image))
    style_model.setTarget(style_v)

    output = style_model(content_image)
    #output = utils.color_match(output, style_v)
    utils.tensor_save_bgrimage(output.data[0], args.output_image, args.cuda)


def fast_evaluate(args, basedir, contents, idx = 0):
    # basedir to save the data
    style_model = Net(ngf=args.ngf)
    style_model.load_state_dict(torch.load(args.model), False)
    style_model.eval()
    if args.cuda:
        style_model.cuda()
    
    style_loader = StyleLoader(args.style_folder, args.style_size, 
        cuda=args.cuda)

    for content_image in contents:
        idx += 1
        content_image = utils.tensor_load_rgbimage(content_image, size=args.content_size, keep_asp=True).unsqueeze(0)
        if args.cuda:
            content_image = content_image.cuda()
        content_image = Variable(utils.preprocess_batch(content_image))

        for isx in range(style_loader.size()):
            style_v = Variable(style_loader.get(isx).data)
            style_model.setTarget(style_v)
            output = style_model(content_image)
            filename = os.path.join(basedir, "{}_{}.png".format(idx, isx+1))
            utils.tensor_save_bgrimage(output.data[0], filename, args.cuda)
            print(filename)

# figure out the experiments type
parser = argparse.ArgumentParser()
eval_arg = parser.add_argument_group('Test')
eval_arg.add_argument("--ngf", type=int, default=128,
                                help="number of generator filter channels, default 128")
eval_arg.add_argument("--content-image", type=str, required=True,
                                help="path to content image you want to stylize")
eval_arg.add_argument("--style-image", type=str, default="images/9styles/candy.jpg",
                                help="path to style-image")
eval_arg.add_argument("--content-size", type=int, default=512,
                                help="factor for scaling down the content image")
eval_arg.add_argument("--style-size", type=int, default=512,
                                help="size of style-image, default is the original size of style image")
eval_arg.add_argument("--style-folder", type=str, default="images/9styles/",
                                help="path to style-folder")
eval_arg.add_argument("--output-image", type=str, default="output.jpg",
                                help="path for saving the output image")
eval_arg.add_argument("--model", type=str, required=True,
                                help="saved model to be used for stylizing the image")
eval_arg.add_argument("--cuda", type=int, default=1,
                                help="set it to 1 for running on GPU, 0 for CPU")    
eval_arg.add_argument("--vgg-model-dir", type=str, default="models/",
                                help="directory for vgg, if model is not present in the directory it is downloaded")
parser.parse_args()


# if args.subcommand is None:
    # raise ValueError("ERROR: specify the experiment type")
if eval_arg.cuda and not torch.cuda.is_available():
    raise ValueError("ERROR: cuda is not available, try running on CPU")
evaluate(eval_arg)
