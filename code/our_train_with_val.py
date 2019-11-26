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
    transform = transforms.Compose([transforms.Resize(args.image_size),
                                    transforms.CenterCrop(args.image_size),
                                    transforms.ToTensor()])
    train_dataset = util.FlatFolderDataset(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, **kwargs)
    val_dataset = util.FlatFolderDataset(args.val_folder, transform)
    val_loader = DataLoader(val_dataset, batch_size=1)
    style_dataset = util.FlatFolderDataset(args.style_folder, transform)
    style_loader = DataLoader(style_dataset, batch_size=args.batch_size, **kwargs)

    # net and trainers
    vgg = our_model.Vgg16
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = torch.nn.Sequential(*list(vgg.children())[:31])
    style_model = our_model.StyleTransferNet221(vgg)
    
    optimizer = Adam(style_model.parameters(), args.lr)
    
    count = 0
    start_time = str(time.ctime()).replace(' ', '_')
    if args.resume is not None:
        print('Resuming, initializing using weight from {}.'.format(args.resume))
        checkpoint = torch.load(args.resume)
        style_model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        count = checkpoint["iter"]
        start_time = checkpoint["start_time"]

    if args.cuda:
        style_model.cuda()

    tbar = trange(args.epochs)
    for e in tbar:
        style_model.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        agg_basic_loss = 0.
        for batch_id, content_images in enumerate(train_loader):
            n_batch = len(content_images)
            count += n_batch
            optimizer.zero_grad()
                
            style_images = style_dataset[batch_id%len(style_loader)]
            style_images = style_images.expand_as(content_images)
            content_images = util.preprocess_batch(content_images)
            style_images = util.preprocess_batch(style_images)
            if args.cuda:
                content_images = content_images.cuda()
                style_images = style_images.cuda()
            
            # print(content_images.shape)
            # print(style_images.shape)
            # print(content_images[0])
            # print("style")
            # print(style_images[0])
            _, content_loss, style_loss, basic_loss = style_model(content_images, style_images)

            # for debug
            # print(out_images.shape)
            content_loss = content_loss * args.content_weight
            style_loss = content_loss * args.style_weight
            basic_loss = content_loss * args.basic_weight

            total_loss = content_loss + style_loss + basic_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()
            agg_basic_loss += basic_loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\t \
                    basic: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), count, len(train_dataset),
                                agg_content_loss / (batch_id + 1),
                                agg_style_loss / (batch_id + 1),
                                agg_basic_loss / (batch_id + 1),
                                (agg_content_loss + agg_style_loss + agg_basic_loss) / (batch_id + 1)
                )
                tbar.set_description(mesg)
            
            if (batch_id + 1) % (4 * args.log_interval) == 0:
                # save model
                style_model.eval()
                state = {
                        "iter": count + 1,
                        "model_state": style_model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "start_time": start_time
                }
                save_model_filename = "iters_" + str(count) + "_" + \
                    start_time + "_" + str(
                    args.content_weight) + "_" + str(args.style_weight) + ".model"
                save_model_path = os.path.join(args.save_model_dir, save_model_filename)
                
                torch.save(state, save_model_path)
                
                print("Checkpoint, trained model saved at", save_model_path)
                
                # validation
                with torch.no_grad():
                    for i, val_images in enumerate(val_dataset):
                        style_images = style_dataset[i%len(style_dataset)]
                        # print(val_images.shape)
                        # print(style_images.shape)
                        style_images = style_images[None,:,:]
                        style_images = util.preprocess_batch(style_images)
                        style_images = style_images.cuda()
                        val_images = val_images[None,:,:]
                        val_images = util.preprocess_batch(val_images)
                        val_images = val_images.cuda()
                        
                        out_images, _, _, _ = style_model(val_images, style_images)
                        
                        # safe save
                        path = os.path.join(args.result_folder, str(count))
                        if not os.path.exists(path):
                            os.makedirs(path)
                            
                        val_save_path = os.path.join(path, str(i)+'_input.jpg')
                        output_save_path = os.path.join(path, str(i)+'_output.jpg')
                        style_save_path = os.path.join(path, str(i)+'_style.jpg')
                        
                        val_images = val_images.cpu()
                        style_images = style_images.cpu()
                        out_images = out_images.cpu()
                        
                        # if preprossed, than save bgr, otherwise rgb
                        util.tensor_save_bgrimage(val_images[0], val_save_path)
                        util.tensor_save_bgrimage(out_images[0], output_save_path)
                        util.tensor_save_bgrimage(style_images[0], style_save_path)
                        
                style_model.train()
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
    train_parser.add_argument("--dataset", type=str, default="/home/dataset/small/small",
                                    help="path to training dataset, the path should point to a folder "
                                    "containing another folder with all the training images")
    train_parser.add_argument("--style-folder", type=str, default="/home/dataset/21styles/",
                                    help="path to style-folder")
    train_parser.add_argument("--val-folder", type=str, default="/home/dataset/val/",
                                    help="path to val-folder")
    train_parser.add_argument("--vgg", type=str, default="models/vgg16.pth",
                                    help="directory for vgg, if model is not present in the directory it is downloaded")
    train_parser.add_argument("--save-model-dir", type=str, default="models/train_saved",
                                    help="path to folder where trained model will be saved.")
    train_parser.add_argument("--image-size", type=int, default=512,
                                    help="size of training images, default is 512 X 512")
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
    train_parser.add_argument("--log-interval", type=int, default=500,
                                    help="number of images after which the training loss is logged, default is 500")
    train_parser.add_argument("--resume", type=str, default=None,
                                    help="resume if needed")
    train_parser.add_argument("--result-folder", type=str, default='./results',
                                    help="path to save validation results")
    args = train_parser.parse_args()
    print(args)
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("ERROR: cuda is not available, try running on CPU")

    train(args)
