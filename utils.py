import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
import torchvision


def parse_args():
    desc = 'Pytorch Implementation of \'C2AIR: Consolidated Compact Aerial Image Haze Removal\''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_path_test', type=str, default='./Testing_Images/')
    parser.add_argument('--data_name', type=str, default='Aerial', choices=['Aerial'])
    parser.add_argument('--save_path', type=str, default='result')
    parser.add_argument('--expansion_factor', type=float, default=2.66, help='factor of channel expansion for GDFN')
    parser.add_argument('--num_refinement', type=int, default=4, help='number of channels for refinement stage')
    parser.add_argument('--num_iter', type=int, default=9057800, help='iterations of training')
    parser.add_argument('--batch_size', nargs='+', type=int, default=[1, 1, 1, 1, 1, 1],
                        help='batch size of loading images for progressive learning')
    parser.add_argument('--patch_size', nargs='+', type=int, default=[256, 256, 256, 256, 256, 256],
                        help='patch size of each image for progressive learning')
    parser.add_argument('--lr', type=float, default=0.00003, help='initial learning rate')
    parser.add_argument('--milestone', nargs='+', type=int, default=[276000],
                        help='when to change patch size and batch size')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=-1, help='random seed (-1 for no manual seed)')
    # model_file is None means training stage, else means testing stage
    parser.add_argument('--model_file', type=str, default='./result/Aerial.pth', help='path of pre-trained model file')
    parser.add_argument('--finetune', default=True, help='path of pre-trained model file')

    return init_args(parser.parse_args())


class Config(object):
    def __init__(self, args):
        
        self.data_path_test = args.data_path_test
        self.data_name = args.data_name
        self.save_path = args.save_path
        self.expansion_factor = args.expansion_factor
        self.num_refinement = args.num_refinement
        self.num_iter = args.num_iter
        self.batch_size = args.batch_size
        self.patch_size = args.patch_size
        self.lr = args.lr
        self.milestone = args.milestone
        self.workers = args.workers
        self.model_file = args.model_file
        self.finetune = args.finetune


def init_args(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    return Config(args)


def pad_image_needed(img, size):
    width, height = T.get_image_size(img)
    if width < size[1]:
        img = T.pad(img, [size[1] - width, 0], padding_mode='reflect')
    if height < size[0]:
        img = T.pad(img, [0, size[0] - height], padding_mode='reflect')
    return img


class RainDataset(Dataset):
    def __init__(self, data_path, data_path_test, data_name, data_type, patch_size=None, length=None):
        super().__init__()
        self.data_name, self.data_type, self.patch_size = data_name, data_type, patch_size
        self.rain_images = sorted(glob.glob('{}/*.png'.format(data_path)))
        self.rain_images_test = sorted(glob.glob('{}/*.png'.format(data_path_test)))
        # self.rain_images = sorted(glob.glob('{}/{}/{}/inp/*.png'.format(data_path, data_name, data_type)))
        # self.norain_images = sorted(glob.glob('{}/{}/{}/tar/*.png'.format(data_path, data_name, data_type)))
        # make sure the length of training and testing different
        self.num = len(self.rain_images)
        self.num_test = len(self.rain_images_test)
        self.sample_num = length if data_type == 'train' else self.num

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):

        
        if self.data_type == 'train':
            image_name = os.path.basename(self.rain_images[idx % self.num])

            imag = np.array(Image.open(self.rain_images[idx % self.num]))
            r,c,ch = imag.shape
            width = c//2

            rain = T.to_tensor(imag[:,:width,:])
            norain = T.to_tensor(imag[:,width:width*2,:])
            # mask = T.to_tensor(imag[:,width*2:width*3,:])

            # print('rain shape..............',rain.shape)
            # print('mask shape..............',mask.shape)
            # exit(0)
            # norain = T.to_tensor(Image.open(self.norain_images[idx % self.num]))
            h, w = rain.shape[1:]

            # make sure the image could be cropped
            rain = pad_image_needed(rain, (self.patch_size, self.patch_size))
            norain = pad_image_needed(norain, (self.patch_size, self.patch_size))
            # mask = pad_image_needed(mask, (self.patch_size, self.patch_size))
            i, j, th, tw = RandomCrop.get_params(rain, (self.patch_size, self.patch_size))
            rain = T.crop(rain, i, j, th, tw)
            norain = T.crop(norain, i, j, th, tw)
            # mask = T.crop(mask, i, j, th, tw)
            if torch.rand(1) < 0.5:
                rain = T.hflip(rain)
                norain = T.hflip(norain)
                # mask = T.hflip(mask)
            if torch.rand(1) < 0.5:
                rain = T.vflip(rain)
                norain = T.vflip(norain)
                # mask = T.vflip(mask)


        else:
            image_name = os.path.basename(self.rain_images_test[idx % self.num_test])

            imag = np.array(Image.open(self.rain_images_test[idx % self.num_test]))

            r,c,ch = imag.shape
            width = c//2

            rain = T.to_tensor(imag[:,:width,:])
            norain = T.to_tensor(imag[:,width:width*2,:])
            # mask = T.to_tensor(imag[:,width*2:width*3,:])

            # print(rain.shape)
            # exit(0)
            # norain = T.to_tensor(Image.open(self.norain_images[idx % self.num]))
            h, w = rain.shape[1:]
            # padding in case images are not multiples of 8
            # new_h, new_w = ((h + 8) // 8) * 8, ((w + 8) // 8) * 8
            # pad_h = new_h - h if h % 8 != 0 else 0
            # pad_w = new_w - w if w % 8 != 0 else 0
            # rain = F.pad(rain, (0, pad_w, 0, pad_h), 'reflect')
            # mask = F.pad(mask, (0, pad_w, 0, pad_h), 'reflect')
            # norain = F.pad(norain, (0, pad_w, 0, pad_h), 'reflect')
        return rain, norain, image_name, h, w


class RainDataset_REALWORLD(Dataset):
    def __init__(self, data_path, data_path_test, data_name, data_type, patch_size=None, length=None):
        super().__init__()
        self.data_name, self.data_type, self.patch_size = data_name, data_type, patch_size
        self.rain_images = sorted(glob.glob('{}/*.png'.format(data_path)))
        self.rain_images_test = sorted(glob.glob('{}/*.png'.format(data_path_test)))
        # self.rain_images = sorted(glob.glob('{}/{}/{}/inp/*.png'.format(data_path, data_name, data_type)))
        # self.norain_images = sorted(glob.glob('{}/{}/{}/tar/*.png'.format(data_path, data_name, data_type)))
        # make sure the length of training and testing different
        self.num = len(self.rain_images)
        self.num_test = len(self.rain_images_test)
        self.sample_num = length if data_type == 'train' else self.num

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):

        
        if self.data_type == 'train':
            image_name = os.path.basename(self.rain_images[idx % self.num])

            imag = np.array(Image.open(self.rain_images[idx % self.num]))
            r,c,ch = imag.shape
            width = c//2

            rain = T.to_tensor(imag[:,:width,:])
            norain = T.to_tensor(imag[:,width:width*2,:])
            # mask = T.to_tensor(imag[:,width*2:width*3,:])

            # print('rain shape..............',rain.shape)
            # print('mask shape..............',mask.shape)
            # exit(0)
            # norain = T.to_tensor(Image.open(self.norain_images[idx % self.num]))
            h, w = rain.shape[1:]

            # make sure the image could be cropped
            rain = pad_image_needed(rain, (self.patch_size, self.patch_size))
            norain = pad_image_needed(norain, (self.patch_size, self.patch_size))
            # mask = pad_image_needed(mask, (self.patch_size, self.patch_size))
            i, j, th, tw = RandomCrop.get_params(rain, (self.patch_size, self.patch_size))
            rain = T.crop(rain, i, j, th, tw)
            norain = T.crop(norain, i, j, th, tw)
            # mask = T.crop(mask, i, j, th, tw)
            if torch.rand(1) < 0.5:
                rain = T.hflip(rain)
                norain = T.hflip(norain)
                # mask = T.hflip(mask)
            if torch.rand(1) < 0.5:
                rain = T.vflip(rain)
                norain = T.vflip(norain)
                # mask = T.vflip(mask)


        else:
            image_name = os.path.basename(self.rain_images_test[idx % self.num_test])

            imag = np.array(Image.open(self.rain_images_test[idx % self.num_test]))

            # r,c,ch = imag.shape
            # width = c//2

            rain = T.to_tensor(imag)
            # norain = T.to_tensor(imag[:,width:width*2,:])
            # mask = T.to_tensor(imag[:,width*2:width*3,:])

            # print(rain.shape)
            # exit(0)
            # norain = T.to_tensor(Image.open(self.norain_images[idx % self.num]))
            h, w = rain.shape[1:]
            # padding in case images are not multiples of 8
            # new_h, new_w = ((h + 8) // 8) * 8, ((w + 8) // 8) * 8
            # pad_h = new_h - h if h % 8 != 0 else 0
            # pad_w = new_w - w if w % 8 != 0 else 0
            # rain = F.pad(rain, (0, pad_w, 0, pad_h), 'reflect')
            # mask = F.pad(mask, (0, pad_w, 0, pad_h), 'reflect')
            # norain = F.pad(norain, (0, pad_w, 0, pad_h), 'reflect')
        return rain, image_name, h, w
def rgb_to_y(x):
    rgb_to_grey = torch.tensor([0.256789, 0.504129, 0.097906], dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return torch.sum(x * rgb_to_grey, dim=1, keepdim=True).add(16.0)


def psnr(x, y, data_range=255.0):
    x, y = x / data_range, y / data_range
    mse = torch.mean((x - y) ** 2)
    score = - 10 * torch.log10(mse)
    return score


def ssim(x, y, kernel_size=11, kernel_sigma=1.5, data_range=255.0, k1=0.01, k2=0.03):
    x, y = x / data_range, y / data_range
    # average pool image if the size is large enough
    f = max(1, round(min(x.size()[-2:]) / 256))
    if f > 1:
        x, y = F.avg_pool2d(x, kernel_size=f), F.avg_pool2d(y, kernel_size=f)

    # gaussian filter
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device)
    coords -= (kernel_size - 1) / 2.0
    g = coords ** 2
    g = (- (g.unsqueeze(0) + g.unsqueeze(1)) / (2 * kernel_sigma ** 2)).exp()
    g /= g.sum()
    kernel = g.unsqueeze(0).repeat(x.size(1), 1, 1, 1)

    # compute
    c1, c2 = k1 ** 2, k2 ** 2
    n_channels = x.size(1)
    mu_x = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu_y = F.conv2d(y, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu_xx, mu_yy, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sigma_xx = F.conv2d(x ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xx
    sigma_yy = F.conv2d(y ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_yy
    sigma_xy = F.conv2d(x * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xy

    # contrast sensitivity (CS) with alpha = beta = gamma = 1.
    cs = (2.0 * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)
    # structural similarity (SSIM)
    ss = (2.0 * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs
    return ss.mean()


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

