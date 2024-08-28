#go to the folder
#get the index that im allowed to sample from
#pull a sequence of 16/8 consecutive frames and pick a crop from it

"""
This Dataloder loads an input image and its corresponding segmentation mask.
The images are loaded from  an input directory which contains two directories /inputs and /gsround_truths.
This dataloader expects that the all images in the input folder have a ground truth map with the same naming convention.
"""
import numpy as np
from PIL import Image
import json
from dataloaders.ucf101_dataloader import RandomCrop,RandomHorizontalFlip,RG_Normalization,ToTensor,ColorJitter,Scale,LBP_transformation,CenterCrop #dataloaders.
from PIL import ImageOps
import torch.nn.functional as nnf
import glob
import torch
import os
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from random import randint
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import cv2
import copy
import random
from PIL import ImageFile

WEATHER = ['SUN','CLOUD','RAIN','SNOW','FOG']
CONTINUAL_DATA_MODE = ['WEATHER','LIGHT']

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def cv2_loader(path):
    return  cv2.imread(path,cv2.IMREAD_GRAYSCALE)

def get_padding_values(h,patch_size):
    n = h // patch_size
    if (h % patch_size) ==0:
        return 0,0
    else:
        additional_pixels = (n+1)* patch_size - h
        if (additional_pixels % 2) ==0:
            return additional_pixels//2,additional_pixels//2
        else:
            return (additional_pixels // 2)+1, (additional_pixels // 2)

def remove_padding(input_img,x1,y1,x2,y2):
    if x1 * y1 * x2 * y2 != 0:
        input_img = input_img[:, x1:-y1, x2:-y2]
    else:
        if x1 * y1 * x2 != 0:
            input_img = input_img[:, x1:y1, x2:-y2]
        elif x1 * y1 * y2 != 0:
            input_img = input_img[:, x1:-y1, :-y2]
        elif x1 * x2 * y2 != 0:
            input_img = input_img[:, x1:, x2:-y2]
        elif y1 * x2 * y2 != 0:
            input_img = input_img[:, :-y1, x2:-y2]
        elif x1 * y1 != 0:
            input_img = input_img[:, x1:-y1, :]
        elif x1 * y2 != 0:
            input_img = input_img[:, x1:, :-y2]
        elif x1 * x2 != 0:
            input_img = input_img[:, x1:, x2:]
        elif y1 * x2 != 0:
            input_img = input_img[:, :-y1, x2:]
        elif y1 * y2 != 0:
            input_img = input_img[:, :-y1, :-y2]
        elif x2 * y2 != 0:
            input_img = input_img[:, :, x2:-y2]
        elif x1 != 0:
            input_img = input_img[:, x1:, :]
        elif x2 != 0:
            input_img = input_img[:, :, x2:]
        elif y1 != 0:
            input_img = input_img[:, :-y1, :]
        elif y2 != 0:
            input_img = input_img[:, :, :-y2]
   #print(input_img.shape)
    return input_img

def multi_target_one_hot_vector(y, n_dims=None):
    ''' Take integer tensor y with n dims and convert it to 1-hot representation with n+1 dims. '''


    y_tensor = y.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot

class EndlessRunnerLoader(Dataset):
    def __init__(self, args=None, mode='train',weather_modes=None,transformations=None,seq_len=16,return_label=False,varying_speeds=False):
        super(EndlessRunnerLoader, self).__init__()

        self.data_mode = 'weather'#args.continual_data_mode
        self.return_label = return_label
        self.seq_length =seq_len


        self.transform = transformations  # apply same transform


        # all file names
        if mode =="train":
            self.image_arr_path = '/data/aj_data/data/Endless_runner/Train_IncrementalEnvironments'
            #os.path.join(data_path,Train_)#glob.glob(str(configs['TRAINING_DIR']) + str("/images/*"))
        elif mode =="test":
            #TODO add test data
            self.image_arr_path = '/data/aj_data/data/Endless_runner/Test_IncrementalEnvironments'

        with open(os.path.join(self.image_arr_path,'Sequence.json')) as f:
            d = json.load(f)


        self.image_arr = glob.glob(os.path.join(self.image_arr_path,'Color','0' ,'*'))[:]
        self.image_arr.sort()

        dict_arrays = {'SUN':self.image_arr[:d[1]['Sequence']['ImageCounter']-1],
                'CLOUD': self.image_arr[d[1]['Sequence']['ImageCounter']:d[2]['Sequence']['ImageCounter'] - 1], #RAIN
                'RAIN':self.image_arr[d[2]['Sequence']['ImageCounter']:d[3]['Sequence']['ImageCounter'] - 1],
                'SNOW':self.image_arr[d[3]['Sequence']['ImageCounter']:d[4]['Sequence']['ImageCounter'] - 1],
                'FOG': self.image_arr[d[4]['Sequence']['ImageCounter']:]}
        self.image_arr =[]
        if self.data_mode =='weather':
            for mode in weather_modes:
                self.image_arr = self.image_arr + dict_arrays[mode]
        # Calculate len

        self.data_len = int(len(self.image_arr)/(self.seq_length))
    def __getitem__(self, index):
        #for seq length
        img_indx = index * self.seq_length
        seq_paths = self.image_arr[img_indx:img_indx+16]

        seq = [pil_loader(img_path) for img_path in seq_paths]
        t_seq = self.transform(seq)  # apply same transform

        t_seq = torch.stack(t_seq, 0).permute(1,0,2,3)
        #t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1, 2)
        if self.return_label:
            labels = [cv2_loader(img_path.replace('Color', 'Mask').replace('jpeg', 'png')) for img_path in seq_paths]
            w = 400
            h = 400
            center = [labels[0].shape[0] / 2, labels[0].shape[1] / 2]
            x = center[1] - w / 2
            y = center[0] - h / 2
            # print("shape one label",torch.from_numpy(labels[0]).unsqueeze(0).shape)

            labels_seq_orig = [torch.from_numpy(label[int(y):int(y + h), int(x):int(x + w)]).unsqueeze(0) for label in
                               labels]  # apply same transform

            labels_seq_orig = torch.stack(labels_seq_orig, 0).permute(1, 0, 2, 3)

            labels_seq = torch.zeros_like(labels_seq_orig)

            for i in range(10, 25):
                labels_seq[labels_seq_orig == i] = 1  # Tree

            for i in range(45, 60):
                labels_seq[labels_seq_orig == i] = 2  # Car
            for i in range(88, 105):
                labels_seq[labels_seq_orig == i] = 3  # People

            for i in range(125, 145):
                labels_seq[labels_seq_orig == i] = 4  # Street Lamps

            return t_seq,labels_seq

        #mask_path = img_path.replace("inputs","ground_truths") #This probably needs to be changed depending on our structure
        else:
            return t_seq,torch.LongTensor([0])

    def __len__(self):
        return int(len(self.image_arr)/(self.seq_length))



class MultiEndlessRunnerLoader(Dataset):
    def __init__(self, args=None, mode='train',weather_modes=None,vanilla_transform=None,rg_transform=None,lbp_transform=None,wavelet_transform=None,seq_len=16,return_label=False):
        super(MultiEndlessRunnerLoader, self).__init__()

        self.data_mode = 'weather'#args.continual_data_mode
        self.return_label = return_label
        self.seq_length =seq_len

        self.vanilla_transform = vanilla_transform
        self.rg_transform = rg_transform
        self.lbp_transform = lbp_transform
        self.wavelet_transform = wavelet_transform

        # all file names
        if mode =="train":
            self.image_arr_path = '/data/aj_data/data/Endless_runner/Train_IncrementalInstance_LightIntensity'
            #'/home/ajaziri/Thesis_work/src/vision/main/data/Endless_runner/Train_IncrementalEnvironments'
            #os.path.join(data_path,Train_)#glob.glob(str(configs['TRAINING_DIR']) + str("/images/*"))
        elif mode =="test":
            #TODO add test data
            self.image_arr_path = '/data/aj_data/data/Endless_runner/Test_IncrementalInstance_LightIntensity'
                                  #'/home/ajaziri/Thesis_work/src/vision/main/data/Endless_runner/Test_IncrementalEnvironments'

        with open(os.path.join(self.image_arr_path,'Sequence.json')) as f:
            d = json.load(f)


        self.image_arr = glob.glob(os.path.join(self.image_arr_path,'Color','0' ,'*'))[:]
        self.image_arr.sort()

        dict_arrays = {'SUN':self.image_arr[:d[1]['Sequence']['ImageCounter']-1],
                'CLOUD': self.image_arr[d[1]['Sequence']['ImageCounter']:d[2]['Sequence']['ImageCounter'] - 1], #RAIN
                'RAIN':self.image_arr[d[2]['Sequence']['ImageCounter']:d[3]['Sequence']['ImageCounter'] - 1],
                'SNOW':self.image_arr[d[3]['Sequence']['ImageCounter']:d[4]['Sequence']['ImageCounter'] - 1],
                'FOG': self.image_arr[d[4]['Sequence']['ImageCounter']:]}
        self.image_arr =[]
        if self.data_mode =='weather':
            for mode in weather_modes:
                self.image_arr = self.image_arr + dict_arrays[mode]
        # Calculate len

        self.data_len = int(len(self.image_arr)/(self.seq_length))
    def __getitem__(self, index):
        #for seq length
        img_indx = index * self.seq_length
        seq_paths = self.image_arr[img_indx:img_indx+16]

        seq = [pil_loader(img_path) for img_path in seq_paths]
        t_seq_vanilla = self.vanilla_transform(seq)  # apply same transform

        t_seq_vanilla= torch.stack(t_seq_vanilla, 0).permute(1,0,2,3)

        t_seq_lbp = self.lbp_transform(seq)  # apply same transform
        t_seq_lbp = torch.stack(t_seq_lbp, 0).permute(1, 0, 2, 3)

        t_seq_rg = self.rg_transform(seq)  # apply same transform
        t_seq_rg = torch.stack(t_seq_rg, 0).permute(1, 0, 2, 3)

        t_seq_wavelet = self.wavelet_transform(seq)  # apply same transform

        t_seq_wavelet = torch.stack(t_seq_wavelet, 0).permute(1, 0, 2, 3)

        #t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1, 2)
        if self.return_label:
            #we are cropping becausse we need faster computations



            labels = [cv2_loader(img_path.replace('Color','Mask').replace('jpeg','png')) for img_path in seq_paths]
            w = 400
            h=400
            center = [labels[0].shape[0] / 2,labels[0].shape[1] / 2]
            x = center[1] - w / 2
            y = center[0] - h / 2
            #print("shape one label",torch.from_numpy(labels[0]).unsqueeze(0).shape)

            labels_seq_orig = [torch.from_numpy(label[int(y):int(y + h), int(x):int(x + w)]).unsqueeze(0) for label in labels]  # apply same transform

            labels_seq_orig = torch.stack(labels_seq_orig, 0).permute(1, 0, 2, 3)

            labels_seq = torch.zeros_like(labels_seq_orig)

            for i in range(10, 25):
                labels_seq[labels_seq_orig == i] = 1  # Tree

            for i in range(45, 60):
                labels_seq[labels_seq_orig == i] = 2  # Car
            for i in range(88, 105):
                labels_seq[labels_seq_orig == i] = 3  # People

            for i in range(125, 145):
                labels_seq[labels_seq_orig == i] = 4  # Street Lamps

            return t_seq_vanilla,t_seq_rg,t_seq_lbp,t_seq_wavelet,labels_seq

        #mask_path = img_path.replace("inputs","ground_truths") #This probably needs to be changed depending on our structure
        else:
            return t_seq_vanilla,t_seq_rg,t_seq_lbp,t_seq_wavelet,torch.LongTensor([0])

    def __len__(self):
        return int(len(self.image_arr)/(self.seq_length))




def get_endlessrunner_dataloader(args, rank=None, world_size=None,er_modes=['SUN']):
    if args.train_mode == 'so_mvment':
        crop_size = args.input_dim
        scale_size = args.resize_size
    else:
        crop_size = args.input_dim
        scale_size = args.resize_size



    if not args.test_mode:
        train_ops =[
            RandomHorizontalFlip(consistent=True),
            #Scale(size=(scale_size, scale_size)),
            RandomCrop(size=crop_size, consistent=True),
            # RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.10, p=1.0, consistent=False),
            ToTensor(),
            # Normalize()
        ]
        test_ops = [
            Scale(size=(scale_size, scale_size)),
            # RandomCrop(size=crop_size, consistent=True),
            # RandomGray(consistent=False, p=0.5),
            ToTensor(),
            # RG_Normalization()
            # Normalize()
        ]
    else:
        train_ops = [
            CenterCrop(400),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.10, p=1.0, consistent=False),
            ToTensor(),
            # Normalize()
        ]
        test_ops = [
            CenterCrop(400),
            ToTensor(),
        ]

    #test_ops=[
    #    Scale(size=(scale_size, scale_size)),
        #RandomCrop(size=crop_size, consistent=True),
        # RandomGray(consistent=False, p=0.5),
    #    ToTensor(),
        #RG_Normalization()
        # Normalize()
    #]
    if args.input_mode=='lbp':
        train_ops.append(LBP_transformation())
        test_ops.append(LBP_transformation())

    elif args.input_mode=='rgNorm':
        train_ops.append(RG_Normalization())
        test_ops.append(RG_Normalization())

    train_transform = transforms.Compose(train_ops)
    test_transform = transforms.Compose(test_ops)
    # CREATION OF DATALOADERS
    train_dataset = EndlessRunnerLoader(weather_modes=er_modes,mode='train', transformations=train_transform,seq_len=16,
                              return_label=args.test_mode,)
    # FRAME RATE DOWNSAMPLING: FPS = 30/downsample

    if world_size != None:
        # distribute the trainig data on multiple GPU. This currently works only when we are using the full dataset
        print('Distirubted Loading...')
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False,
                                     drop_last=False)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            sampler=sampler)

    else:
        sampler = torch.utils.data.RandomSampler(train_dataset)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=16,
                                                   pin_memory=True,
                                                   drop_last=True)

        # CREATION OF DATALOADERS
    test_dataset = EndlessRunnerLoader(weather_modes=er_modes,mode='test', transformations=test_transform,seq_len=16,
                              return_label=args.test_mode,
                              varying_speeds=args.varied_video_speed)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16
    )

    # TODO create train/val split

    return (
        train_loader,
        train_dataset,
        test_loader,
        test_dataset,
    )


def get_multi_modal_endlessrunner_dataloader(args, rank=None, world_size=None,er_modes=['SUN']):


    train_ops_vanilla = [
        CenterCrop(400),
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.10, p=1.0, consistent=False),
        ToTensor(),
    ]
    test_ops_vanilla = [
        CenterCrop(400),
        ToTensor(),
    ]

    train_ops_rg = [
        CenterCrop(400),
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.10, p=1.0, consistent=False),
        ToTensor(),
        RG_Normalization()
    ]
    test_ops_rg = [
        CenterCrop(400),
        ToTensor(),
        RG_Normalization()
    ]

    train_ops_lbp = [
        CenterCrop(400),
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.10, p=1.0, consistent=False),
        ToTensor(),
        LBP_transformation()
    ]
    test_ops_lbp = [
        CenterCrop(400),
        ToTensor(),
        LBP_transformation()
    ]

    train_ops_wavelet = [
        CenterCrop(400),
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.10, p=1.0, consistent=False),
        ToTensor(),
    ]
    test_ops_wavelet = [
        CenterCrop(400),
        ToTensor(),
    ]

    train_transform_vanilla = transforms.Compose(train_ops_vanilla)
    test_transform_vanilla = transforms.Compose(test_ops_vanilla)

    train_transform_rg = transforms.Compose(train_ops_rg)
    test_transform_rg = transforms.Compose(test_ops_rg)

    train_transform_lbp = transforms.Compose(train_ops_lbp)
    test_transform_lbp = transforms.Compose(test_ops_lbp)

    train_transform_wavelet = transforms.Compose(train_ops_wavelet)
    test_transform_wavelet = transforms.Compose(test_ops_wavelet)
    # CREATION OF DATALOADERS
    train_dataset = MultiEndlessRunnerLoader(weather_modes=er_modes,mode='train', vanilla_transform=train_transform_vanilla,
                                             rg_transform=train_transform_rg,lbp_transform=train_transform_lbp,wavelet_transform=train_transform_wavelet,seq_len=16,
                              return_label=args.test_mode)
    # FRAME RATE DOWNSAMPLING: FPS = 30/downsample
    if world_size != None:
        # distribute the trainig data on multiple GPU. This currently works only when we are using the full dataset
        print('Distirubted Loading...')
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False,
                                     drop_last=False)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            sampler=sampler)

    else:
        sampler = torch.utils.data.RandomSampler(train_dataset)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=16,
                                                   pin_memory=True,
                                                   drop_last=True)

        # CREATION OF DATALOADERS
    test_dataset = MultiEndlessRunnerLoader(weather_modes=er_modes,mode='test',vanilla_transform=test_transform_vanilla,
                                             rg_transform=test_transform_rg,lbp_transform=test_transform_lbp,wavelet_transform=test_transform_wavelet,seq_len=16,
                              return_label=args.test_mode)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16
    )

    # TODO create train/val split

    return (
        train_loader,
        train_dataset,
        test_loader,
        test_dataset,
    )


def transform_inputs(imgs,labels,patch_size = 240):

    # get padding values for each dimension
    x1, y1 = get_padding_values(imgs.size(1), patch_size)
    x2, y2 = get_padding_values(imgs.size(2), patch_size)
    # pad images
    new_img_tensor = F.pad(imgs, (x2, y2, x1, y1))
    new_label_tensor = F.pad(labels, (x2, y2, x1, y1))

    #print("new img tensor",new_img_tensor.shape)

    patches_imgs =  new_img_tensor.data.unfold(0, new_img_tensor.shape[0], new_img_tensor.shape[0]).unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches_labels = new_label_tensor.data.unfold(0, 1, 1).unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)

    # save shape of images patches
    shape_of_img_patches = patches_imgs.data.cpu().numpy().shape
    shape_of_labels_patches = patches_labels.data.cpu().numpy().shape

    # print('shape of image patches', shape_of_img_patches)

    # flatten patches
    patches_imgs = torch.flatten(patches_imgs, start_dim=0, end_dim=2)
    patches_labels = torch.flatten(patches_labels, start_dim=0, end_dim=2)

    #print("xxxxxxxxxxxx",patches_labels.shape,patches_imgs.shape)
    classification_labels=[]
    for i in range(patches_labels.shape[0]):
        current_patch = patches_labels[i]

        uniques, counts = np.unique(current_patch.cpu().numpy(), return_counts=True)

        if uniques.shape[0] > 1:  # if  there is more than one label in patch
            if uniques[0] == 0:  # ignore the background label
                index_max = np.argmax(counts[1:]) + 1
            else:
                index_max = np.argmax(counts)

            if counts[index_max] > 100:

                label = uniques[index_max]
            else:
                label = 0

        else:
            label = uniques[0]

        classification_labels.append(label)
    target = torch.from_numpy(np.array(classification_labels))
    one_hot =multi_target_one_hot_vector(target,n_dims=5)
    return patches_imgs,one_hot
def transform_batch(data,label,patch_size = 120):
    patches_all = []
    labels_all = []
    #print('transform batch',data.shape,label.shape)
    for idx in range(data.shape[2]):
        out1, out2 = transform_inputs(data[0, :, idx], label[0, :, idx],patch_size = 120)
        patches_all.append(out1)
        labels_all.append(out2)

    patches_all = torch.stack(patches_all, dim=0).permute(1, 2, 0, 3, 4)
    labels_all = torch.stack(labels_all, dim=0).permute(1, 0, 2)
    summed = torch.sum(labels_all, dim=1)
    summed[summed > 0] = 1

    return patches_all,summed

if __name__ == "__main__":
    #first seq is normal sunny
    #second seq is cloudy
    #third seq is raining
    #fourth seq is snowing
    #fifth seq is foggy
    #TODO make a get data

    import  matplotlib.pyplot as plt
    scale_size =256
    train_ops = [
        #RandomHorizontalFlip(consistent=True),
        #Scale(size=(scale_size, scale_size)),
        #RandomCrop(size=120, consistent=True),
        # RandomGray(consistent=False, p=0.5),
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.10, p=1.0, consistent=False),
        ToTensor(),
        # Normalize()
    ]
    train_transform = transforms.Compose(train_ops)

    Dataset_test = EndlessRunnerLoader(weather_modes=['SUN'],mode='test',transformations=train_transform,seq_len=16,return_label=True)

    test_load = \
        torch.utils.data.DataLoader(dataset=Dataset_test,
                                    num_workers=2, batch_size=1, shuffle=True)
    for batch, (data,label) in enumerate(test_load):
        input_seq = label.detach().cpu().numpy()
    #    target_img = data['gt'][0]

        print('image', data.shape,label.shape)
        patches_all =[]
        labels_all=[]
        for idx in range(data.shape[2]):
            out1,out2 =transform_inputs(data[0,:,idx],label[0,:,idx])
            patches_all.append(out1)
            labels_all.append(out2)

        patches_all = torch.stack(patches_all, dim=0).permute(1,2,0,3,4)
        labels_all = torch.stack(labels_all, dim=0).permute(1,0,2)
        summed = torch.sum(labels_all,dim=1)
        summed[summed>0]=1
        #print(summed.shape,patches_all.shape)
        #print('out final', patches_all.shape,labels_all.shape,summed.shape
        #      )
        #for indx_i mg in range(16):
        #    plt.Figure()
        #    plt.imshow(input_seq[0,0,indx_img])
        #    plt.savefig(f"./view_examples/endlessrunner{indx_img}.png")
        #TODO Transform to one hot vectors for multi classification

        #TODO multi target classification



