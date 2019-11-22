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
import argparse
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


def train(args):
    check_paths(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        kwargs = {'num_workers': 0, 'pin_memory': False}
    else:
        kwargs = {}

    transform = transforms.Compose([transforms.Scale(args.image_size),
                                    transforms.CenterCrop(args.image_size),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.mul(255))])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, **kwargs)

    style_model = Net(ngf=args.ngf)
    if args.resume is not None:
        print('Resuming, initializing using weight from {}.'.format(args.resume))
        style_model.load_state_dict(torch.load(args.resume))
    print(style_model)
    optimizer = Adam(style_model.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16()
    utils.init_vgg16(args.vgg_model_dir)
    vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, "vgg16.weight")))

    if args.cuda:
        style_model.cuda()
        vgg.cuda()

    style_loader = utils.StyleLoader(args.style_folder, args.style_size)

    tbar = trange(args.epochs)
    for e in tbar:
        style_model.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()
            x = Variable(utils.preprocess_batch(x))
            if args.cuda:
                x = x.cuda()

            style_v = style_loader.get(batch_id)
            style_model.setTarget(style_v)

            style_v = utils.subtract_imagenet_mean_batch(style_v)
            features_style = vgg(style_v)
            gram_style = [utils.gram_matrix(y) for y in features_style]

            y = style_model(x)
            xc = Variable(x.data.clone())

            y = utils.subtract_imagenet_mean_batch(y)
            xc = utils.subtract_imagenet_mean_batch(xc)

            features_y = vgg(y)
            features_xc = vgg(xc)

            f_xc_c = Variable(features_xc[1].data, requires_grad=False)

            content_loss = args.content_weight * mse_loss(features_y[1], f_xc_c)

            style_loss = 0.
            for m in range(len(features_y)):
                gram_y = utils.gram_matrix(features_y[m])
                gram_s = Variable(gram_style[m].data, requires_grad=False).repeat(args.batch_size, 1, 1, 1)
                style_loss += args.style_weight * mse_loss(gram_y, gram_s[:n_batch, :, :])

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.data[0]
            agg_style_loss += style_loss.data[0]

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                agg_content_loss / (batch_id + 1),
                                agg_style_loss / (batch_id + 1),
                                (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                tbar.set_description(mesg)

            
            if (batch_id + 1) % (4 * args.log_interval) == 0:
                # save model
                style_model.eval()
                style_model.cpu()
                save_model_filename = "Epoch_" + str(e) + "iters_" + str(count) + "_" + \
                    str(time.ctime()).replace(' ', '_') + "_" + str(
                    args.content_weight) + "_" + str(args.style_weight) + ".model"
                save_model_path = os.path.join(args.save_model_dir, save_model_filename)
                torch.save(style_model.state_dict(), save_model_path)
                style_model.train()
                style_model.cuda()
                tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)

    # save model
    style_model.eval()
    style_model.cpu()
    save_model_filename = "Final_epoch_" + str(args.epochs) + "_" + \
        str(time.ctime()).replace(' ', '_') + "_" + str(
        args.content_weight) + "_" + str(args.style_weight) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(style_model.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def check_paths(args):
    try:
        if not os.path.exists(args.vgg_model_dir):
            os.makedirs(args.vgg_model_dir)
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


parser = argparse.ArgumentParser()
train_arg = parser.add_argument_group('Train')
train_arg.add_argument("--ngf", type=int, default=128,
                                help="number of generator filter channels, default 128")
train_arg.add_argument("--epochs", type=int, default=2,
                                help="number of training epochs, default is 2")
train_arg.add_argument("--batch-size", type=int, default=4,
                                help="batch size for training, default is 4")
train_arg.add_argument("--dataset", type=str, default="dataset/",
                                help="path to training dataset, the path should point to a folder "
                                "containing another folder with all the training images")
train_arg.add_argument("--style-folder", type=str, default="images/9styles/",
                                help="path to style-folder")
train_arg.add_argument("--vgg-model-dir", type=str, default="models/",
                                help="directory for vgg, if model is not present in the directory it is downloaded")
train_arg.add_argument("--save-model-dir", type=str, default="models/",
                                help="path to folder where trained model will be saved.")
train_arg.add_argument("--image-size", type=int, default=256,
                                help="size of training images, default is 256 X 256")
train_arg.add_argument("--style-size", type=int, default=512,
                                help="size of style-image, default is the original size of style image")
train_arg.add_argument("--cuda", type=int, default=1, 
                                help="set it to 1 for running on GPU, 0 for CPU")
train_arg.add_argument("--seed", type=int, default=42, 
                                help="random seed for training")
train_arg.add_argument("--content-weight", type=float, default=1.0,
                                help="weight for content-loss, default is 1.0")
train_arg.add_argument("--style-weight", type=float, default=5.0,
                                help="weight for style-loss, default is 5.0")
train_arg.add_argument("--lr", type=float, default=1e-3,
                                help="learning rate, default is 0.001")
train_arg.add_argument("--log-interval", type=int, default=500,
                                help="number of images after which the training loss is logged, default is 500")
train_arg.add_argument("--resume", type=str, default=None,
                                help="resume if needed")
# args = Options().parse()
# if args.subcommand is None:
    # raise ValueError("ERROR: specify the experiment type")
if train_arg.cuda and not torch.cuda.is_available():
    raise ValueError("ERROR: cuda is not available, try running on CPU")


# if args.subcommand == "train":
    # Training the model 
train(train_arg)
