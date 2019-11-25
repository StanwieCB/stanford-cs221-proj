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

from utils import util
from models import our_model

def train(args):
    check_paths(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        kwargs = {'num_workers': 0, 'pin_memory': False}
    else:
        kwargs = {}

    # loaders
    transform = transforms.Compose([transforms.Scale(args.image_size),
                                    transforms.CenterCrop(args.image_size),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.mul(255))])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, **kwargs)
    style_loader = util.StyleLoader(args.style_folder, args.image_size, args.batch_size)

    # net and trainers
    style_model = our_model.StyleTransferNet221()
    if args.resume is not None:
        print('Resuming, initializing using weight from {}.'.format(args.resume))
        style_model.load_state_dict(torch.load(args.resume))
    
    optimizer = Adam(style_model.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = our_model.Vgg16
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = torch.nn.Sequential(*list(vgg.children())[:31])

    if args.cuda:
        style_model.cuda()
        vgg.cuda()

    tbar = trange(args.epochs)
    for e in tbar:
        style_model.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        agg_basic_loss = 0.
        count = 0
        for batch_id, (content_images, _) in enumerate(train_loader):
            n_batch = len(content_images)
            count += n_batch
            optimizer.zero_grad()
            content_images = Variable(util.preprocess_batch(content_images))
            if args.cuda:
                content_images = content_images.cuda()
            style_images = style_loader.get(batch_id)
            style_feature, content_feature, out_images = style_model(content_images, style_images)
            # for debug
            # print(content_images.shape, style_images.shape, out_images.shape)
            _, content_feature_o, _ = style_model(out_images, style_images)

            # L_c
            content_loss = args.content_weight * mse_loss(content_feature_o, content_feature)
            print(content_loss)
            style_feature_o, _, _ = style_model(content_images, out_images)

            gram_style_f = util.gram_matrix(style_feature)
            gram_style_o = util.gram_matrix(style_feature_o)
            # L_s
            style_loss = args.style_weight * mse_loss(gram_style_f, gram_style_o)
            print(style_loss)

            out_images = util.subtract_imagenet_mean_batch(out_images)
            content_images_clone = content_images.clone()
            content_images_clone = util.subtract_imagenet_mean_batch(content_images_clone)

            features_vgg_o = vgg(out_images)
            features_vgg_c = vgg(content_images_clone)
            #L_b
            basic_loss = args.basic_weight * mse_loss(features_vgg_o, features_vgg_c)
            
            total_loss = content_loss + style_loss + basic_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()
            agg_basic_loss += basic_loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\t \
                    basic: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                agg_content_loss / (batch_id + 1),
                                agg_style_loss / (batch_id + 1),
                                agg_basic_loss / (batch_id + 1),
                                (agg_content_loss + agg_style_loss + agg_basic_loss) / (batch_id + 1)
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
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    train_parser = argparse.ArgumentParser()
    train_parser.add_argument("--epochs", type=int, default=2,
                                    help="number of training epochs, default is 2")
    train_parser.add_argument("--batch-size", type=int, default=4,
                                    help="batch size for training, default is 4")
    train_parser.add_argument("--dataset", type=str, default="/home/dataset/small",
                                    help="path to training dataset, the path should point to a folder "
                                    "containing another folder with all the training images")
    train_parser.add_argument("--style-folder", type=str, default="/home/dataset/21styles/",
                                    help="path to style-folder")
    train_parser.add_argument("--vgg", type=str, default="models/vgg16.pth",
                                    help="directory for vgg, if model is not present in the directory it is downloaded")
    train_parser.add_argument("--save-model-dir", type=str, default="models/train_saved",
                                    help="path to folder where trained model will be saved.")
    train_parser.add_argument("--image-size", type=int, default=256,
                                    help="size of training images, default is 256 X 256")
    train_parser.add_argument("--cuda", type=int, default=1, 
                                    help="set it to 1 for running on GPU, 0 for CPU")
    train_parser.add_argument("--seed", type=int, default=42, 
                                    help="random seed for training")
    train_parser.add_argument("--content-weight", type=float, default=1.0,
                                    help="weight for content-loss, default is 1.0")
    train_parser.add_argument("--style-weight", type=float, default=5.0,
                                    help="weight for style-loss, default is 5.0")
    train_parser.add_argument("--basic-weight", type=float, default=4.0,
                                    help="weight for style-loss, default is 4.0")
    train_parser.add_argument("--lr", type=float, default=1e-4,
                                    help="learning rate, default is 0.0001")
    train_parser.add_argument("--log-interval", type=int, default=5,
                                    help="number of images after which the training loss is logged, default is 500")
    train_parser.add_argument("--resume", type=str, default=None,
                                    help="resume if needed")
    args = train_parser.parse_args()
    print(args)
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("ERROR: cuda is not available, try running on CPU")

    train(args)
