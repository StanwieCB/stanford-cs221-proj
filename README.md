# Stanford CS221 Fall 2019 Project - Artistic Neural Styel Transfer

Team Member: [Yiheng Zhang](https://stanwiecb.github.io/hankzhang.github.io/) | [Yuyan Wang](https://github.com/theawen) | [Yichen Li](https://github.com/AntheaLi)

## Code Structure
`/models`: our deep learning model and saved pre-trained model

`/results`: example results of the pre-trained model

`/images`: example images

`/utils`: utilities used for training and testing

`train.py`: train code for our project

`test.py`: test code for our project

## Training
Following packages are prerequisites for our model:
* PyTorch >= 1.0.0
* Python >= 3.6
* torchvision
* tqdm
* numpy

To train our model, you need to specify the content images for training by specifing `--dataset PATH_TO_YOUR_CONTENT_IMAGES` and the style images by specifing  `--style-folder PATH_TO_YOUR_STYLE_IMAGES`. Furthermore, you have to download the pretrained encoder weights [vgg16.pth](https://drive.google.com/open?id=18HWJ-go9XSks4h65G-vWFNOKHcysa8wO) and put it in `./models`.

E.g. `python train --dataset /home/dataset/small/ --style-folder /home/dataset/21styles/`

The trained model will be saved in `models/train_saved` by default. Resuming training is also supported.

For more argument options, please refer to `train_parser` in `train.py`

## Testing
To train our model, you need to specify a pretrained model by `--decoder PATH_TO_MODEL` the content images for training by specifing `--content-folder PATH_TO_YOUR_CONTENT_IMAGES` and the style images by specifing  `--style-folder PATH_TO_YOUR_STYLE_IMAGES`. This will iterating all possible combination of content images and style images.

We've provided a pretrained model at [trained.model](https://drive.google.com/open?id=1-zNgMWVGW4FFeBprilayCL23KOjLek9L).

E.g. `python train --decoder ./models/train_saved/trained.model --content-folder /images/content/ --style-folder /images/style/`

The results will be saved at `/results` by default.


For more argument options, please refer to `test_parser` in `test.py`