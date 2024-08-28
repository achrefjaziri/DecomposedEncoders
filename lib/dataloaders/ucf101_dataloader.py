import torch
from torch.utils import data
import pandas as pd
from torchvision import transforms
import torch.nn as nn
import pickle
import os
import random
import numbers
import math
import collections
import numpy as np
from PIL import ImageOps, Image
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
from dataloaders.lbp_rg_transfo import LBP, NormalizedRG  #dataloaders.


class Padding:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, img):
        return ImageOps.expand(img, border=self.pad, fill=0)

class Scale:
    def __init__(self, size, interpolation=Image.NEAREST):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgmap):
        # assert len(imgmap) > 1 # list of images
        img1 = imgmap[0]
        if isinstance(self.size, int):
            w, h = img1.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return imgmap
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return [i.resize((ow, oh), self.interpolation) for i in imgmap]
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return [i.resize((ow, oh), self.interpolation) for i in imgmap]
        else:
            return [i.resize(self.size, self.interpolation) for i in imgmap]


class CenterCrop:
    def __init__(self, size, consistent=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgmap):
        img1 = imgmap[0]
        w, h = img1.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return [i.crop((x1, y1, x1 + tw, y1 + th)) for i in imgmap]


class RandomCropWithProb:
    def __init__(self, size, p=0.8, consistent=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.consistent = consistent
        self.threshold = p

    def __call__(self, imgmap):
        img1 = imgmap[0]
        w, h = img1.size
        if self.size is not None:
            th, tw = self.size
            if w == tw and h == th:
                return imgmap
            if self.consistent:
                if random.random() < self.threshold:
                    x1 = random.randint(0, w - tw)
                    y1 = random.randint(0, h - th)
                else:
                    x1 = int(round((w - tw) / 2.))
                    y1 = int(round((h - th) / 2.))
                return [i.crop((x1, y1, x1 + tw, y1 + th)) for i in imgmap]
            else:
                result = []
                for i in imgmap:
                    if random.random() < self.threshold:
                        x1 = random.randint(0, w - tw)
                        y1 = random.randint(0, h - th)
                    else:
                        x1 = int(round((w - tw) / 2.))
                        y1 = int(round((h - th) / 2.))
                    result.append(i.crop((x1, y1, x1 + tw, y1 + th)))
                return result
        else:
            return imgmap


class RandomCrop:
    def __init__(self, size, consistent=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.consistent = consistent

    def __call__(self, imgmap, flowmap=None):
        img1 = imgmap[0]
        w, h = img1.size
        if self.size is not None:
            th, tw = self.size
            if w == tw and h == th:
                return imgmap
            if not flowmap:
                if self.consistent:
                    x1 = random.randint(0, w - tw)
                    y1 = random.randint(0, h - th)
                    return [i.crop((x1, y1, x1 + tw, y1 + th)) for i in imgmap]
                else:
                    result = []
                    for i in imgmap:
                        x1 = random.randint(0, w - tw)
                        y1 = random.randint(0, h - th)
                        result.append(i.crop((x1, y1, x1 + tw, y1 + th)))
                    return result
            elif flowmap is not None:
                assert (not self.consistent)
                result = []
                for idx, i in enumerate(imgmap):
                    proposal = []
                    for j in range(3):  # number of proposal: use the one with largest optical flow
                        x = random.randint(0, w - tw)
                        y = random.randint(0, h - th)
                        proposal.append([x, y, abs(np.mean(flowmap[idx, y: y + th, x: x + tw, :]))])
                    [x1, y1, _] = max(proposal, key=lambda x: x[-1])
                    result.append(i.crop((x1, y1, x1 + tw, y1 + th)))
                return result
            else:
                raise ValueError('wrong case')
        else:
            return imgmap


class RandomSizedCrop:
    def __init__(self, size, interpolation=Image.BILINEAR, consistent=True, p=1.0):
        self.size = size
        self.interpolation = interpolation
        self.consistent = consistent
        self.threshold = p

    def __call__(self, imgmap):
        img1 = imgmap[0]
        if random.random() < self.threshold:  # do RandomSizedCrop
            for attempt in range(10):
                area = img1.size[0] * img1.size[1]
                target_area = random.uniform(0.5, 1) * area
                aspect_ratio = random.uniform(3. / 4, 4. / 3)

                w = int(round(math.sqrt(target_area * aspect_ratio)))
                h = int(round(math.sqrt(target_area / aspect_ratio)))

                if self.consistent:
                    if random.random() < 0.5:
                        w, h = h, w
                    if w <= img1.size[0] and h <= img1.size[1]:
                        x1 = random.randint(0, img1.size[0] - w)
                        y1 = random.randint(0, img1.size[1] - h)

                        imgmap = [i.crop((x1, y1, x1 + w, y1 + h)) for i in imgmap]
                        for i in imgmap: assert (i.size == (w, h))

                        return [i.resize((self.size, self.size), self.interpolation) for i in imgmap]
                else:
                    result = []
                    for i in imgmap:
                        if random.random() < 0.5:
                            w, h = h, w
                        if w <= img1.size[0] and h <= img1.size[1]:
                            x1 = random.randint(0, img1.size[0] - w)
                            y1 = random.randint(0, img1.size[1] - h)
                            result.append(i.crop((x1, y1, x1 + w, y1 + h)))
                            assert (result[-1].size == (w, h))
                        else:
                            result.append(i)

                    assert len(result) == len(imgmap)
                    return [i.resize((self.size, self.size), self.interpolation) for i in result]

            # Fallback
            scale = Scale(self.size, interpolation=self.interpolation)
            crop = CenterCrop(self.size)
            return crop(scale(imgmap))
        else:  # don't do RandomSizedCrop, do CenterCrop
            crop = CenterCrop(self.size)
            return crop(imgmap)


class RandomHorizontalFlip:
    def __init__(self, consistent=True, command=None):
        self.consistent = consistent
        if command == 'left':
            self.threshold = 0
        elif command == 'right':
            self.threshold = 1
        else:
            self.threshold = 0.5

    def __call__(self, imgmap):
        if self.consistent:
            if random.random() < self.threshold:
                return [i.transpose(Image.FLIP_LEFT_RIGHT) for i in imgmap]
            else:
                return imgmap
        else:
            result = []
            for i in imgmap:
                if random.random() < self.threshold:
                    result.append(i.transpose(Image.FLIP_LEFT_RIGHT))
                else:
                    result.append(i)
            assert len(result) == len(imgmap)
            return result


class RandomGray:
    '''Actually it is a channel splitting, not strictly grayscale images'''

    def __init__(self, consistent=True, p=0.5):
        self.consistent = consistent
        self.p = p  # probability to apply grayscale

    def __call__(self, imgmap):
        if self.consistent:
            if random.random() < self.p:
                return [self.grayscale(i) for i in imgmap]
            else:
                return imgmap
        else:
            result = []
            for i in imgmap:
                if random.random() < self.p:
                    result.append(self.grayscale(i))
                else:
                    result.append(i)
            assert len(result) == len(imgmap)
            return result

    def grayscale(self, img):
        channel = np.random.choice(3)
        np_img = np.array(img)[:, :, channel]
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
        return img


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image. --modified from pytorch source code
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, consistent=False, p=1.0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        self.consistent = consistent
        self.threshold = p

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = torchvision.transforms.Compose(transforms)

        return transform

    def __call__(self, imgmap):
        if random.random() < self.threshold:  # do ColorJitter
            if self.consistent:
                transform = self.get_params(self.brightness, self.contrast,
                                            self.saturation, self.hue)
                return [transform(i) for i in imgmap]
            else:
                result = []
                for img in imgmap:
                    transform = self.get_params(self.brightness, self.contrast,
                                                self.saturation, self.hue)
                    result.append(transform(img))
                return result
        else:  # don't do ColorJitter, do nothing
            return imgmap

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


class RandomRotation:
    def __init__(self, consistent=True, degree=15, p=1.0):
        self.consistent = consistent
        self.degree = degree
        self.threshold = p

    def __call__(self, imgmap):
        if random.random() < self.threshold:  # do RandomRotation
            if self.consistent:
                deg = np.random.randint(-self.degree, self.degree, 1)[0]
                return [i.rotate(deg, expand=True) for i in imgmap]
            else:
                return [i.rotate(np.random.randint(-self.degree, self.degree, 1)[0], expand=True) for i in imgmap]
        else:  # don't do RandomRotation, do nothing
            return imgmap


class ToTensor:
    def __call__(self, imgmap):
        totensor = transforms.ToTensor()
        return [totensor(i) for i in imgmap]


class Normalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, imgmap):
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        return [normalize(i) for i in imgmap]


def lbp_lambda(x):
    lbp_transform = LBP(radius=3, points=24)
    # print('shape in lbp_lambda',x.shape)
    img_out = torch.Tensor(lbp_transform(x[0].detach().numpy()))
    img_out = torch.unsqueeze(img_out, 0)
    return img_out


def rg_lambda(x):
    rg_norm = NormalizedRG(conf=False)
    # print('shape i lbp_lambda',x.shape)
    img_out = torch.Tensor(rg_norm(x.permute(1, 2, 0).detach().numpy())).permute(2, 0, 1)
    # img_out=torch.unsqueeze(img_out, 0)
    return img_out


class LBP_transformation:
    def __init__(self, conf=False):
        self.conf = conf

    def __call__(self, imgmap):
        return [lbp_lambda(i) for i in imgmap]


class RG_Normalization:
    def __init__(self, conf=False):
        self.conf = conf

    def __call__(self, imgmap):
        return [rg_lambda(i) for i in imgmap]


def pil_loader(path):
    with open(path.replace('/home/ajaziri/Thesis_work/src/vision/main','/data/aj_data'), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class UCF101_3d(data.Dataset):
    def __init__(self,
                 mode='train',
                 transform=None,
                 seq_len=10,
                 num_seq=5,
                 downsample=3,
                 epsilon=5,
                 which_split=1,
                 return_label=False,
                 varying_speeds=False):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.which_split = which_split
        self.return_label = return_label
        self.varying_speeds = varying_speeds
        path = '/data/aj_data/data/UCF101_data'
        # splits
        if mode == 'train':
            split = os.path.join(path, 'train_split%02d.csv') % self.which_split
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):  # use val for test
            split = os.path.join(path, 'test_split%02d.csv') % self.which_split
            video_info = pd.read_csv(split, header=None)
        else:
            raise ValueError('wrong mode')

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}
        action_file = os.path.join(path, 'splits_classification', 'classInd.txt')
        action_df = pd.read_csv(action_file, sep=' ', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
            vpath, vlen = row
            if vlen - self.num_seq * self.seq_len * self.downsample <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

        if mode == 'val': self.video_info = self.video_info.sample(frac=0.3)
        # shuffle not required due to external sampler

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        if vlen - self.num_seq * self.seq_len * self.downsample <= 0: return [None]
        n = 1
        start_idx = np.random.choice(range(vlen - self.num_seq * self.seq_len * self.downsample), n)
        seq_idx = np.expand_dims(np.arange(self.num_seq), -1) * self.downsample * self.seq_len + start_idx
        seq_idx_block = seq_idx + np.expand_dims(np.arange(self.seq_len), 0) * self.downsample
        return [seq_idx_block, vpath]

    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]

        if self.varying_speeds:
            while True:
                self.downsample = random.choice([1, 1, 1, 3, 5, 6, 10, 11, 12])
                if not (vlen - self.num_seq * self.seq_len * self.downsample <= 0):
                    break
            # print("xxxx",vlen,self.seq_len * self.downsample)
            if self.downsample == 1:
                pace_label = 0
            elif self.downsample in [3, 5, 6]:
                pace_label = 1
            else:
                pace_label = 2

        items = self.idx_sampler(vlen, vpath)
        if items is None: print(vpath)

        idx_block, vpath = items
        assert idx_block.shape == (self.num_seq, self.seq_len)
        idx_block = idx_block.reshape(self.num_seq * self.seq_len)

        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i + 1))) for i in idx_block]
        t_seq = self.transform(seq)  # apply same transform

        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1, 2)

        if self.return_label:
            try:
                vname = vpath.split('/')[-3]
                vid = self.encode_action(vname)
            except:
                vname = vpath.split('/')[-2]
                vid = self.encode_action(vname)
            label = torch.LongTensor([vid])
            return t_seq[0], label[0]-1

        if self.varying_speeds:
            return t_seq[0], pace_label

        return t_seq[0],torch.LongTensor([0])


    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return action code'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]



class multi_UCF101_3d(data.Dataset):
    def __init__(self,
                 mode='train',
                 vanilla_transform=None, rg_transform=None, lbp_transform=None, wavelet_transform=None,
                 seq_len=10,
                 num_seq=5,
                 downsample=3,
                 epsilon=5,
                 which_split=1,
                 return_label=False,
                 varying_speeds=False):
        self.mode = mode
        self.vanilla_transform = vanilla_transform
        self.rg_transform = rg_transform
        self.lbp_transform = lbp_transform
        self.wavelet_transform = wavelet_transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.which_split = which_split
        self.return_label = return_label
        self.varying_speeds = varying_speeds
        path = '/data/aj_data/data/UCF101_data'
        # splits
        if mode == 'train':
            split = os.path.join(path, 'train_split%02d.csv') % self.which_split
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):  # use val for test
            split = os.path.join(path, 'test_split%02d.csv') % self.which_split
            video_info = pd.read_csv(split, header=None)
        else:
            raise ValueError('wrong mode')

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}
        action_file = os.path.join(path, 'splits_classification', 'classInd.txt')
        action_df = pd.read_csv(action_file, sep=' ', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
            vpath, vlen = row
            if vlen - self.num_seq * self.seq_len * self.downsample <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

        if mode == 'val': self.video_info = self.video_info.sample(frac=0.3)
        # shuffle not required due to external sampler

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        if vlen - self.num_seq * self.seq_len * self.downsample <= 0: return [None]
        n = 1
        start_idx = np.random.choice(range(vlen - self.num_seq * self.seq_len * self.downsample), n)
        seq_idx = np.expand_dims(np.arange(self.num_seq), -1) * self.downsample * self.seq_len + start_idx
        seq_idx_block = seq_idx + np.expand_dims(np.arange(self.seq_len), 0) * self.downsample
        return [seq_idx_block, vpath]

    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]



        items = self.idx_sampler(vlen, vpath)
        if items is None: print(vpath)

        idx_block, vpath = items
        assert idx_block.shape == (self.num_seq, self.seq_len)
        idx_block = idx_block.reshape(self.num_seq * self.seq_len)

        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i + 1))) for i in idx_block]
        t_seq_vanilla = self.vanilla_transform(seq)  # apply same transform

        (C, H, W) = t_seq_vanilla[0].size()
        t_seq_vanilla = torch.stack(t_seq_vanilla, 0)
        t_seq_vanilla = t_seq_vanilla.view(self.num_seq, self.seq_len, C, H, W).transpose(1, 2)

        t_seq_rg = self.rg_transform(seq)  # apply same transform

        (C, H, W) = t_seq_rg[0].size()
        t_seq_rg = torch.stack(t_seq_rg, 0)
        t_seq_rg = t_seq_rg.view(self.num_seq, self.seq_len, C, H, W).transpose(1, 2)

        t_seq_lbp = self.lbp_transform(seq)  # apply same transform

        (C, H, W) = t_seq_lbp[0].size()
        t_seq_lbp = torch.stack(t_seq_lbp, 0)
        t_seq_lbp = t_seq_lbp.view(self.num_seq, self.seq_len, C, H, W).transpose(1, 2)

        t_seq_wavelet = self.wavelet_transform(seq)  # apply same transform

        (C, H, W) = t_seq_wavelet[0].size()
        t_seq_wavelet = torch.stack(t_seq_wavelet, 0)
        t_seq_wavelet = t_seq_wavelet.view(self.num_seq, self.seq_len, C, H, W).transpose(1, 2)


        if self.return_label:
            try:
                vname = vpath.split('/')[-3]
                vid = self.encode_action(vname)
            except:
                vname = vpath.split('/')[-2]
                vid = self.encode_action(vname)
            label = torch.LongTensor([vid])
            return t_seq_vanilla[0],t_seq_rg[0],t_seq_lbp[0],t_seq_wavelet[0], label[0]-1



        return t_seq_vanilla[0],t_seq_rg[0],t_seq_lbp[0],t_seq_wavelet[0],torch.LongTensor([0])


    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return action code'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]



def get_ucf101_dataloader(args, rank=None, world_size=None):
    # TODO some hardcoded variables need to change!!
    if args.train_mode == 'so_mvment':
        crop_size = args.input_dim
        scale_size = args.resize_size
    else:
        crop_size = args.input_dim
        scale_size = args.resize_size


    scale_size = args.input_dim


    train_ops =[
        #RandomHorizontalFlip(consistent=True),
        Scale(size=(scale_size, scale_size)),
        #RandomCrop(size=scale_size, consistent=True),
        # RandomGray(consistent=False, p=0.5),
        #ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.10, p=1.0, consistent=False),
        ToTensor(),
        #Normalize()
    ]
    test_ops=[
        Scale(size=(scale_size, scale_size )),
        #RandomCrop(size=scale_size, consistent=True),
        # RandomGray(consistent=False, p=0.5),
        ToTensor(),
        #RG_Normalization()
        # Normalize()
    ]
    if args.input_mode=='lbp':
        train_ops.append(LBP_transformation())
        test_ops.append(LBP_transformation())

    elif args.input_mode=='rgNorm':
        train_ops.append(RG_Normalization())
        test_ops.append(RG_Normalization())

    train_transform = transforms.Compose(train_ops)
    test_transform = transforms.Compose(test_ops)
    # CREATION OF DATALOADERS
    seq_length_var= 32
    train_dataset = UCF101_3d(mode='train',
                              transform=train_transform,
                              seq_len=seq_length_var,
                              num_seq=1,  # NUMBER OF SEQUENCES, ARTEFACT FROM DPC CODE, KEEP SET TO 1!
                              downsample=3,
                              return_label=args.test_mode,
                             varying_speeds=False)  # FRAME RATE DOWNSAMPLING: FPS = 30/downsample

    if world_size != None:
        # distribute the trainig data on multiple GPU. This currently works only when we are using the full dataset
        print('Distirubted Loading...')
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False,
                                     drop_last=False)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=16,
            sampler=sampler)
    elif args.nb_samples < 1.0:
        # if sample represents the percentage of the dataset used for training 1.0 means the whole dataset
        valid_size = args.nb_samples
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(valid_size * num_train))
        train_idx, throwaway_idx = indices[:split], indices[split:]
        train_sampler = SubsetRandomSampler(train_idx)
        # default dataset loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=16
        )  # shuffle=True,
        print('finished sampling from train', len(train_loader))
    else:

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=32,
                                                   pin_memory=True,
                                                   drop_last=True)

        # CREATION OF DATALOADERS
    test_dataset = UCF101_3d(mode='test',
                             transform=test_transform,
                             seq_len=seq_length_var,
                             num_seq=1,  # NUMBER OF SEQUENCES, ARTEFACT FROM DPC CODE, KEEP SET TO 1!
                             downsample=3,
                             return_label=args.test_mode,
                             varying_speeds=False)  # FRAME RATE DOWNSAMPLING: FPS = 30/downsample

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=32
    )
    # TODO create train/val split
    return (
        train_loader,
        train_dataset,
        test_loader,
        test_dataset,
    )
def get_multi_modal_ucf101_dataloader(args, rank=None, world_size=None):



    scale_size = args.input_dim


    train_ops =[
        #RandomHorizontalFlip(consistent=True),
        Scale(size=(scale_size, scale_size)),

        ToTensor(),
        #Normalize()
    ]
    test_ops=[
        Scale(size=(scale_size, scale_size )),
        ToTensor(),
    ]

    train_ops_lbp = [
        Scale(size=(scale_size, scale_size)),
        ToTensor(),
        LBP_transformation()

    ]
    test_ops_lbp = [
        Scale(size=(scale_size, scale_size)),
        ToTensor(),
        LBP_transformation()

    ]

    train_ops_rg = [
        Scale(size=(scale_size, scale_size)),
        ToTensor(),
        RG_Normalization()
    ]
    test_ops_rg = [
        Scale(size=(scale_size, scale_size)),
        ToTensor(),
        RG_Normalization()

    ]

    train_ops_wavelet = [
        Scale(size=(256, 256)),
        ToTensor(),
    ]
    test_ops_wavelet = [
        Scale(size=(256, 256)),
        ToTensor(),
    ]


    train_transform_vanilla = transforms.Compose(train_ops)
    test_transform_vanilla = transforms.Compose(test_ops)

    train_transform_rg = transforms.Compose(train_ops_rg)
    test_transform_rg = transforms.Compose(test_ops_rg)

    train_transform_lbp = transforms.Compose(train_ops_lbp)
    test_transform_lbp = transforms.Compose(test_ops_lbp)

    train_transform_wavelet = transforms.Compose(train_ops_wavelet)
    test_transform_wavelet = transforms.Compose(test_ops_wavelet)
    # CREATION OF DATALOADERS
    train_dataset = multi_UCF101_3d(mode='train',
                              vanilla_transform=train_transform_vanilla,rg_transform=train_transform_rg,lbp_transform=train_transform_lbp,wavelet_transform=train_transform_wavelet,
                              seq_len=16,
                              num_seq=1,  # NUMBER OF SEQUENCES, ARTEFACT FROM DPC CODE, KEEP SET TO 1!
                              downsample=3,
                              return_label=args.test_mode,
                             varying_speeds=False)  # FRAME RATE DOWNSAMPLING: FPS = 30/downsample

    if world_size != None:
        # distribute the trainig data on multiple GPU. This currently works only when we are using the full dataset
        print('Distirubted Loading...')
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False,
                                     drop_last=False)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=16,
            sampler=sampler)

    else:
        sampler = torch.utils.data.RandomSampler(train_dataset)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=32,
                                                   pin_memory=True,
                                                   drop_last=True)

        # CREATION OF DATALOADERS
    test_dataset = multi_UCF101_3d(mode='test',
                             vanilla_transform=test_transform_vanilla, rg_transform=test_transform_rg,
                             lbp_transform=test_transform_lbp, wavelet_transform=test_transform_wavelet,
                             seq_len=16,
                             num_seq=1,  # NUMBER OF SEQUENCES, ARTEFACT FROM DPC CODE, KEEP SET TO 1!
                             downsample=3,
                             return_label=args.test_mode,
                             varying_speeds=False)  # FRAME RATE DOWNSAMPLING: FPS = 30/downsample

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=32
    )

    # TODO create train/val split

    return (
        train_loader,
        train_dataset,
        test_loader,
        test_dataset,
    )



if __name__ == "__main__":
    """
    ucf_data_dir = "/home/ajaziri/Thesis_work/src/vision/main/data/UCF101_data/UCF-101"
    ucf_label_dir = "/home/ajaziri/Thesis_work/src/vision/main/data/UCF101_data/ucfTrainTestlist"
    frames_per_clip = 16
    step_between_clips = 1
    batch_size = 4

    tfs = transforms.Compose([
        # TODO: this should be done by a video-level transfrom when PyTorch provides transforms.ToTensor() for video
        # scale in [0, 1] of type float
        transforms.Lambda(lambda x: x / 255.),
        # reshape into (T, C, H, W) for easier convolutions
        transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
        # rescale to the most common size
        transforms.Lambda(lambda x: nn.functional.interpolate(x, (240, 320))),
    ])


    def custom_collate(batch):
        filtered_batch = []
        for video, _, label in batch:
            filtered_batch.append((video, label))
        return torch.utils.data.dataloader.default_collate(filtered_batch)


    # create train loader (allowing batches and other extras)
    train_dataset = UCF101(ucf_data_dir, ucf_label_dir, frames_per_clip=frames_per_clip,
                           step_between_clips=step_between_clips, train=True, transform=tfs)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=custom_collate)
    # create test loader (allowing batches and other extras)
    test_dataset = UCF101(ucf_data_dir, ucf_label_dir, frames_per_clip=frames_per_clip,
                          step_between_clips=step_between_clips, train=False, transform=tfs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                              collate_fn=custom_collate)

    print(f"Total number of train samples: {len(train_dataset)}")
    print(f"Total number of test samples: {len(test_dataset)}")
    print(f"Total number of (train) batches: {len(train_loader)}")
    print(f"Total number of (test) batches: {len(test_loader)}")

    for i,(vid,label) in enumerate(train_loader):
        print('one video',vid.shape,label.shape)
        break
    """

    transform = transforms.Compose([
        RandomHorizontalFlip(consistent=True),
        RandomCrop(size=224, consistent=True),
        Scale(size=(120, 120)),
        # RandomGray(consistent=False, p=0.5),
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.10, p=1.0, consistent=False),
        ToTensor(),
        # Normalize()
    ])

    # CREATION OF DATALOADERS
    dataset = UCF101_3d(mode='train',
                        transform=transform,
                        seq_len=16,
                        num_seq=1,  # NUMBER OF SEQUENCES, ARTEFACT FROM DPC CODE, KEEP SET TO 1!
                        downsample=3,
                        varying_speeds=True)  # FRAME RATE DOWNSAMPLING: FPS = 30/downsample
    sampler = data.RandomSampler(dataset)

    data_loader = data.DataLoader(dataset,
                                  batch_size=4,
                                  sampler=sampler,
                                  shuffle=False,
                                  num_workers=8,
                                  pin_memory=True,
                                  drop_last=True)

    print('"%s" dataset size: %d' % ("train", len(dataset)))
    for i, (seq, label) in enumerate(data_loader):
        print(seq.shape, label.shape, label)
        """
                for idx in range(seq.shape[3]):
            img = seq[0,0,:,idx]
            print(img.shape)
            plt.Figure()

            plt.imshow(img.permute(1,2,0).detach().cpu().numpy())
            plt.savefig(f"./view_examples/motion_img_example{idx}.png")
        
        """
