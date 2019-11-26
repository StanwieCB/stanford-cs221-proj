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

transform = transforms.Compose([transforms.Resize(512),
                                    transforms.CenterCrop(512),
                                    transforms.ToTensor()])
val_dataset = util.FlatFolderDataset('/home/dataset/val', transform)
style_dataset = util.FlatFolderDataset('/home/dataset/21styles', transform)

print(len(style_dataset))
for i, val_images in enumerate(val_dataset):
    print(i%len(style_dataset))
    style_images = style_dataset[i%len(style_dataset)]
    
    style_images = style_images[None,:,:]
    val_images = val_images[None,:,:]
    val_images = util.preprocess_batch(val_images)
    style_images = util.preprocess_batch(style_images)
    
    style_save_path = os.path.join('test-val/', str(i)+'_style.jpg')
    val_save_path = os.path.join('test-val/', str(i)+'_val.jpg')
    if not os.path.exists('test-val/'):
        os.makedirs('test-val/')
    util.tensor_save_bgrimage(val_images[0], val_save_path)
    util.tensor_save_bgrimage(style_images[0], style_save_path)