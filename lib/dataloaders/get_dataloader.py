from torchvision.transforms import transforms
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch
from torch.utils import data
import os
import numpy as np
import torchvision
import argparse
from PIL import Image
from dataloaders.lbp_rg_transfo import LBP, NormalizedRG  #
from dataloaders.codebrim_dataloader import get_codebrim_dataloader
from dataloaders.ucf101_dataloader import get_ucf101_dataloader
from dataloaders.gtsrb_dataloader import GTSRB

from dataloaders.endless_runner_dataloader import get_endlessrunner_dataloader


def get_dataloader(args, rank=None, world_size=None, er_modes=['SUN']):
    if args.dataset == 'MNIST':
        train_loader, train_dataset, test_loader, test_dataset = get_MNIST_dataloader(args, rank, world_size)
    elif args.dataset == 'FashionMNIST':
        train_loader, train_dataset, test_loader, test_dataset = get_FashionMNIST_dataloader(args, rank, world_size)

    elif args.dataset == 'KuzushijiMNIST':
        train_loader, train_dataset, test_loader, test_dataset = get_KuzushijiMNIST_dataloder(args, rank, world_size)

    elif args.dataset == 'CIFAR10':
        train_loader, train_dataset, test_loader, test_dataset = get_CIFAR10_dataloader(args, rank, world_size)
    elif args.dataset == 'GTSRB':
        train_loader, train_dataset, test_loader, test_dataset = get_GTSRB_dataloder(args, rank, world_size)
    elif args.dataset == 'CIFAR100':
        train_loader, train_dataset, test_loader, test_dataset = get_CIFAR100_dataloader(args, rank, world_size)
    elif args.dataset == 'SVHN':
        train_loader, train_dataset, test_loader, test_dataset = get_SVHN_dataloder(args, rank, world_size)

    elif args.dataset == 'STL10':
        train_loader, train_dataset, test_loader, test_dataset = get_stl10_dataloader(args, rank, world_size)

    elif args.dataset == 'ImageNet':
        train_loader, train_dataset, test_loader, test_dataset = get_ImageNet_dataloader(args, rank, world_size)

    elif args.dataset == 'CODEBRIM':
        train_loader, train_dataset, test_loader, test_dataset = get_codebrim_dataloader(args, rank, world_size)

    elif args.dataset == 'UCF101':
        train_loader, train_dataset, test_loader, test_dataset = get_ucf101_dataloader(args, rank, world_size)
    elif args.dataset == 'ER':  # EndlessRunner Dataset
        train_loader, train_dataset, test_loader, test_dataset = get_endlessrunner_dataloader(args, rank, world_size,
                                                                                              er_modes=er_modes)

    else:
        print('No valid dataset is specified')

    # embed()
    # raise Exception()
    return (
        train_loader,
        train_dataset,
        test_loader,
        test_dataset,
    )


class Stl10MultiModalDataset(torchvision.datasets.STL10):
    def __init__(self, base_folder, split="train", transform=None, download=True, vanilla_transform=None,
                 rg_transform=None, lbp_transform=None, wavelet_transform=None):
        super().__init__(root=base_folder, split=split, download=download, transform=transform)
        self.vanilla_transform = vanilla_transform
        self.rg_transform = rg_transform
        self.lbp_transform = lbp_transform
        self.wavelet_transform = wavelet_transform

    def __getitem__(self, index):
        image, label = self.data[index], self.labels[index]
        image = np.uint8(image).transpose(1, 2, 0)
        image = Image.fromarray(image, 'RGB')

        rg_img = self.rg_transform(image)
        vanilla_img = self.vanilla_transform(image)
        lbp_img = self.lbp_transform(image)
        wavelt_img = self.wavelet_transform(image)

        # print(image.shape)
        # label = torch.ToTensor(label)
        return vanilla_img, rg_img, lbp_img, wavelt_img, label


def get_multi_modal_stl10_dataloader(args, args_rg, args_lbp, args_wavelet):
    base_folder = os.path.join(args.data_input_dir, "stl10_binary")
    aug_vanilla = {
        "stl10": {
            "randcrop": args.input_dim,  # 64
            "flip": True,
            "resize": args.resize_input,
            "lbp": True if args.input_mode == 'lbp' else False,
            "rgNorm": True if args.input_mode == 'rgNorm' else False,
            'resize_size': args.resize_size,
            "pad": False,
            "grayscale": True if args.input_ch == 1 else False,
            "lbp": True if args.input_mode == 'lbp' else False,
            "mean": [0.4313, 0.4156, 0.3663],  # values for train+unsupervised combined
            "std": [0.2683, 0.2610, 0.2687],
            "bw_mean": [0.4120],  # values for train+unsupervised combined
            "bw_std": [0.2570],
        }  # values for labeled train set: mean [0.4469, 0.4400, 0.4069], std [0.2603, 0.2566, 0.2713]
    }

    aug_rg = {
        "stl10": {
            "randcrop": args_rg.input_dim,  # 64
            "flip": True,
            "resize": args_rg.resize_input,
            "lbp": True if args_rg.input_mode == 'lbp' else False,
            "rgNorm": True if args_rg.input_mode == 'rgNorm' else False,
            'resize_size': args_rg.resize_size,
            "pad": False,
            "grayscale": True if args_rg.input_ch == 1 else False,
            "lbp": True if args_rg.input_mode == 'lbp' else False,
            "mean": [0.4313, 0.4156, 0.3663],  # values for train+unsupervised combined
            "std": [0.2683, 0.2610, 0.2687],
            "bw_mean": [0.4120],  # values for train+unsupervised combined
            "bw_std": [0.2570],
        }  # values for labeled train set: mean [0.4469, 0.4400, 0.4069], std [0.2603, 0.2566, 0.2713]
    }

    aug_lbp = {
        "stl10": {
            "randcrop": args_lbp.input_dim,  # 64
            "flip": True,
            "resize": args_lbp.resize_input,
            "lbp": True if args_lbp.input_mode == 'lbp' else False,
            "rgNorm": True if args_lbp.input_mode == 'rgNorm' else False,
            'resize_size': args_lbp.resize_size,
            "pad": False,
            "grayscale": True if args_lbp.input_ch == 1 else False,
            "lbp": True if args_lbp.input_mode == 'lbp' else False,
            "mean": [0.4313, 0.4156, 0.3663],  # values for train+unsupervised combined
            "std": [0.2683, 0.2610, 0.2687],
            "bw_mean": [0.4120],  # values for train+unsupervised combined
            "bw_std": [0.2570],
        }  # values for labeled train set: mean [0.4469, 0.4400, 0.4069], std [0.2603, 0.2566, 0.2713]
    }

    aug_wavelet = {
        "stl10": {
            "randcrop": args_wavelet.input_dim,  # 64
            "flip": True,
            "resize": args_wavelet.resize_input,
            "lbp": True if args_wavelet.input_mode == 'lbp' else False,
            "rgNorm": True if args_wavelet.input_mode == 'rgNorm' else False,
            'resize_size': args_wavelet.resize_size,
            "pad": False,
            "grayscale": True if args_wavelet.input_ch == 1 else False,
            "lbp": True if args_wavelet.input_mode == 'lbp' else False,
            "mean": [0.4313, 0.4156, 0.3663],  # values for train+unsupervised combined
            "std": [0.2683, 0.2610, 0.2687],
            "bw_mean": [0.4120],  # values for train+unsupervised combined
            "bw_std": [0.2570],
        }  # values for labeled train set: mean [0.4469, 0.4400, 0.4069], std [0.2603, 0.2566, 0.2713]
    }

    transform_train_vanilla = transforms.Compose(
        [get_transforms(eval=False, aug=aug_vanilla["stl10"])]
    )
    transform_valid_vanilla = transforms.Compose(
        [get_transforms(eval=True, aug=aug_vanilla["stl10"])]
    )

    transform_train_rg = transforms.Compose(
        [get_transforms(eval=False, aug=aug_rg["stl10"])]
    )
    transform_valid_rg = transforms.Compose(
        [get_transforms(eval=True, aug=aug_rg["stl10"])]
    )

    transform_train_lbp = transforms.Compose(
        [get_transforms(eval=False, aug=aug_lbp["stl10"])]
    )
    transform_valid_lbp = transforms.Compose(
        [get_transforms(eval=True, aug=aug_lbp["stl10"])]
    )

    transform_train_wavelet = transforms.Compose(
        [get_transforms(eval=False, aug=aug_wavelet["stl10"])]
    )
    transform_valid_wavelet = transforms.Compose(
        [get_transforms(eval=True, aug=aug_wavelet["stl10"])]
    )

    # in other cases we use the unlabled data
    train_dataset = Stl10MultiModalDataset(
        base_folder,
        split="train",
        transform=None,
        download=True,
        vanilla_transform=transform_train_vanilla, rg_transform=transform_train_rg, lbp_transform=transform_train_lbp,
        wavelet_transform=transform_train_wavelet

    )  # set download to True to get the dataset

    test_dataset = Stl10MultiModalDataset(
        base_folder, split="test", transform=None, download=True,
        vanilla_transform=transform_valid_vanilla, rg_transform=transform_valid_rg, lbp_transform=transform_valid_lbp,
        wavelet_transform=transform_valid_wavelet
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
    )

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


def get_stl10_dataloader(args, rank=None, world_size=None):
    base_folder = os.path.join(args.data_input_dir, "stl10_binary")

    aug = {
        "stl10": {
            "randcrop": args.input_dim,  # 64
            "flip": True,
            "resize": args.resize_input,
            "lbp": True if args.input_mode == 'lbp' else False,
            "rgNorm": True if args.input_mode == 'rgNorm' else False,
            'resize_size': args.resize_size,
            "pad": False,
            "grayscale": True if args.input_ch == 1 else False,
            "lbp": True if args.input_mode == 'lbp' else False,
            "mean": [0.4313, 0.4156, 0.3663],  # values for train+unsupervised combined
            "std": [0.2683, 0.2610, 0.2687],
            "bw_mean": [0.4120],  # values for train+unsupervised combined
            "bw_std": [0.2570],
        }  # values for labeled train set: mean [0.4469, 0.4400, 0.4069], std [0.2603, 0.2566, 0.2713]
    }
    transform_train = transforms.Compose(
        [get_transforms(eval=False, aug=aug["stl10"])]
    )
    transform_valid = transforms.Compose(
        [get_transforms(eval=True, aug=aug["stl10"])]
    )

    if False:
        train_dataset = Stl10AugmentedDataset(
            base_folder, split="train", download=True
        )

        test_dataset = torchvision.datasets.STL10(
            base_folder, split="test", transform=transform_valid, download=True
        )
    else:
        if args.train_mode == 'CE' or args.train_mode == 'predSim' or args.test_mode:  # in these train modes we need labled data
            train_dataset = torchvision.datasets.STL10(
                base_folder, split="train", transform=transform_train, download=True
            )
        else:
            # in other cases we use the unlabled data
            train_dataset = torchvision.datasets.STL10(
                base_folder,
                split="unlabeled",
                transform=transform_train,
                download=True,
            )  # set download to True to get the dataset

        test_dataset = torchvision.datasets.STL10(
            base_folder, split="test", transform=transform_valid, download=True
        )

    if float(args.nb_samples) < 1.0:
        # if sample represents the percentage of the dataset used for training 1.0 means the whole dataset
        valid_size = float(args.nb_samples)
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(valid_size * num_train))
        train_idx, throwaway_idx = indices[:split], indices[split:]
        train_sampler = SubsetRandomSampler(train_idx)
        args.batch_size_multiGPU = args.batch_size
        # default dataset loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=16
        )  # shuffle=True,

    elif world_size != None:
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
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=16,
        )

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


def get_MNIST_dataloader(args, rank=None, world_size=None):
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.device != 'cpu' else {}
    base_folder = os.path.join(args.data_input_dir, "MNIST")

    train_transform = transforms.Compose([
        transforms.Resize((args.input_dim, args.input_dim)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(args.input_dim, padding=2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    if args.cutout:
        train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
    train_dataset = datasets.MNIST(base_folder, train=True, download=True, transform=train_transform)

    if args.nb_samples < 1.0:
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

    elif world_size != None:
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

        train_loader = torch.utils.data.DataLoader(train_dataset, sampler=None, batch_size=args.batch_size,
                                                   shuffle=args.classes_per_batch == 0, **kwargs)

    test_dataset = datasets.MNIST(base_folder, train=False,
                                  transform=transforms.Compose([
                                      transforms.Resize((args.input_dim, args.input_dim)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ]))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    return (train_loader, train_dataset, test_loader, test_dataset)


def get_FashionMNIST_dataloader(args, rank=None, world_size=None):
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.device != 'cpu' else {}
    base_folder = os.path.join(args.data_input_dir, "FashionMNIST")

    train_transform = transforms.Compose([
        transforms.Resize((args.input_dim, args.input_dim)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(args.input_dim, padding=2),
        transforms.ToTensor(),
        transforms.Normalize((0.286,), (0.353,))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((args.input_dim, args.input_dim)),
        transforms.ToTensor(),
        transforms.Normalize((0.286,), (0.353,))])
    if args.cutout:
        train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))

    train_dataset = datasets.FashionMNIST(base_folder, train=True, download=True,
                                          transform=train_transform)
    test_dataset = datasets.FashionMNIST(base_folder, train=False,
                                         transform=test_transform)

    if args.nb_samples < 1.0:
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

    elif world_size != None:
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
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=None,
            batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size, shuffle=False, **kwargs)

    return (train_loader, train_dataset, test_loader, test_dataset)


def get_KuzushijiMNIST_dataloder(args, rank=None, world_size=None):
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.device != 'cpu' else {}
    base_folder = os.path.join(args.data_input_dir, "KuzushijiMNIST")

    train_transform = transforms.Compose([
        transforms.Resize((args.input_dim, args.input_dim)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(args.input_dim, padding=2),
        transforms.ToTensor(),
        transforms.Normalize((0.1904,), (0.3475,))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((args.input_dim, args.input_dim)),
        transforms.ToTensor(),
        transforms.Normalize((0.1904,), (0.3475,))
    ])

    if args.cutout:
        train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
    train_dataset = KuzushijiMNIST(base_folder, train=True, download=True, transform=train_transform)
    test_dataset = KuzushijiMNIST(base_folder, train=False,
                                  transform=test_transform)

    if args.nb_samples < 1.0:
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

    elif world_size != None:
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
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=None,
            batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False, **kwargs)

    return (train_loader, train_dataset, test_loader, test_dataset)


def get_GTSRB_dataloder(args, rank=None, world_size=None):
    # TODO fix the import here
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.device != 'cpu' else {}
    base_folder = os.path.join(args.data_input_dir, "GTSRB")

    # input_dim = 32
    # input_ch = 3
    # num_classes = 10
    #train_transform = transforms.Compose(
    #    [transforms.Resize((args.input_dim, args.input_dim)), transforms.ToTensor(), transforms.RandomHorizontalFlip(),
    #     transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))])
    trans = [
        transforms.Resize((args.input_dim +14, args.input_dim+14)),
        transforms.RandomCrop((args.input_dim, args.input_dim)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214),
                             (0.2724, 0.2608, 0.2669))
    ]
    if args.input_mode=='lbp':
        trans.append(transforms.Lambda(lbp_lambda))
    if args.input_mode=='rgNorm':
        trans.append(transforms.Lambda(rg_lambda))

    transform = transforms.Compose(trans)





    #test_transform = transforms.Compose([transforms.Resize((args.input_dim, args.input_dim)), transforms.ToTensor(),
    #                                     transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))])

    #if args.input_ch == 1:
    #    train_transform.transforms.append(transforms.Grayscale())
    #    test_transform.transforms.append(transforms.Grayscale())

    train_dataset = GTSRB(
        root_dir=args.data_input_dir, train=True, transform=transform)
    test_dataset = GTSRB(
        root_dir=args.data_input_dir, train=False, transform=transform)
    if args.nb_samples < 1.0:
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

        print('finished sampling from train',len(train))

    elif world_size != None:
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
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=None,
            batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False, **kwargs)
    return (train_loader, train_dataset, test_loader, test_dataset)


def get_SVHN_dataloder(args, rank=None, world_size=None):
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.device != 'cpu' else {}
    base_folder = os.path.join(args.data_input_dir, "SVHN")

    # input_dim = 32
    # input_ch = 3
    # num_classes = 10
    train_transform = transforms.Compose([
        transforms.Resize((args.input_dim, args.input_dim)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(args.input_dim, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.431, 0.430, 0.446), (0.197, 0.198, 0.199))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((args.input_dim, args.input_dim)),

        transforms.ToTensor(),
        transforms.Normalize((0.431, 0.430, 0.446), (0.197, 0.198, 0.199))
    ])

    if args.input_ch == 1:
        train_transform.transforms.append(transforms.Grayscale())
        test_transform.transforms.append(transforms.Grayscale())
    if args.cutout:
        train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
    train_dataset = torch.utils.data.ConcatDataset((
        datasets.SVHN(base_folder, split='train', download=True, transform=train_transform),
        datasets.SVHN(base_folder, split='extra', download=True, transform=train_transform)))
    if args.nb_samples < 1.0:
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

    elif world_size != None:
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
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=None,
            batch_size=args.batch_size, shuffle=True, **kwargs)
    test_dataset = datasets.SVHN(base_folder, split='test', download=True,
                                 transform=test_transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False, **kwargs)
    return (train_loader, train_dataset, test_loader, test_dataset)


def get_CIFAR10_dataloader(args, rank=None, world_size=None):
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.device != 'cpu' else {}
    base_folder = os.path.join(args.data_input_dir, "CIFAR10")

    # input_dim = 32
    # input_ch = 3
    # num_classes = 10
    train_transform = transforms.Compose([
        transforms.Resize((args.input_dim, args.input_dim)),
        transforms.RandomCrop(args.input_dim, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((args.input_dim, args.input_dim)),
        transforms.ToTensor(),
        transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284))
    ])

    if args.input_ch == 1:
        train_transform.transforms.append(transforms.Grayscale())
        test_transform.transforms.append(transforms.Grayscale())
    if args.cutout:
        train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
    train_dataset = datasets.CIFAR10(base_folder, train=True, download=True, transform=train_transform)

    if args.nb_samples < 1.0:
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

    elif world_size != None:
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
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=None,
            batch_size=args.batch_size, shuffle=True, **kwargs)

    test_dataset = datasets.CIFAR10(base_folder, train=False,
                                    transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    return (train_loader, train_dataset, test_loader, test_dataset)


def get_CIFAR100_dataloader(args, rank=None, world_size=None):
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    base_folder = os.path.join(args.data_input_dir, "CIFAR100")

    # input_dim = 32
    # input_ch = 3
    # num_classes = 100
    train_transform = transforms.Compose([
        transforms.Resize(args.resize_size),
        transforms.RandomCrop(args.input_dim),  # , padding=4
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.438, 0.418, 0.377), (0.300, 0.287, 0.294))
    ])
    test_transform = transforms.Compose([
        transforms.Resize(args.resize_size),
        transforms.RandomCrop(args.input_dim),  # , padding=4
        transforms.ToTensor(),
        transforms.Normalize((0.438, 0.418, 0.377), (0.300, 0.287, 0.294))
    ])

    if args.input_ch == 1:
        train_transform.transforms.append(transforms.Grayscale())
        test_transform.transforms.append(transforms.Grayscale())

    if args.cutout:
        train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
    train_dataset = datasets.CIFAR100(base_folder, train=True, download=True, transform=train_transform)
    if args.nb_samples < 1.0:
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

    elif world_size != None:
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
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=None,
            batch_size=args.batch_size, shuffle=True, **kwargs)

    test_dataset = datasets.CIFAR100(base_folder, train=False,
                                     transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    return (train_loader, train_dataset, test_loader, test_dataset)




class KuzushijiMNIST(datasets.MNIST):
    urls = [
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz'
    ]


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


def get_transforms(eval=False, aug=None):
    trans = []

    if aug["resize"]:
        trans.append(transforms.Resize(aug["resize_size"]))

    if aug["pad"]:
        trans.append(transforms.Pad(aug["pad_size"], fill=0, padding_mode='constant'))

    if aug["randcrop"] and not eval:
        trans.append(transforms.RandomCrop(aug["randcrop"]))

    if aug["randcrop"] and eval:
        trans.append(transforms.CenterCrop(aug["randcrop"]))

    if aug["flip"] and not eval:
        trans.append(transforms.RandomHorizontalFlip())

    if aug["grayscale"]:
        trans.append(transforms.Grayscale())
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=aug["bw_mean"], std=aug["bw_std"]))
    elif aug["mean"]:
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=aug["mean"], std=aug["std"]))
    else:
        trans.append(transforms.ToTensor())

    if aug["lbp"] and aug["grayscale"]:
        trans.append(transforms.Lambda(lbp_lambda))
    if aug["rgNorm"] and not aug["grayscale"]:
        trans.append(transforms.Lambda(rg_lambda))
    trans = transforms.Compose(trans)
    return trans


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch local error training')
    parser.add_argument(
        "--dataset",
        default='SVHN',
        help="Available Datasets: MNIST,FashionMNIST,KuzushijiMNIST,CIFAR10,CIFAR100,GTSRB,SVHN,STL10,ImageNet",
    )
    parser.add_argument(
        "--input_dim",
        default=32,
        help="Size of the input image. Recommended: 224 for ImageNet,64 for STL and 32 for the rest ",
    )

    parser.add_argument(
        "--input_ch",
        default=1,
        help="Number of input channels. This parameter does not matter for one channel datasets like MNIST",
    )
    parser.add_argument(
        "--num_classes",
        default=10,
        help="Number of classes for the training data (needed for the supervised models) ",
    )
    parser.add_argument(
        "--data_input_dir",
        default='../data',
        help="Boolean to decide whether to use special weight initialization (delta orthogonal)",
    )
    parser.add_argument(
        "--cutout",
        default=False,
        help="Boolean to decide whether to use cutout regularization",
    )
    parser.add_argument(
        "--nb_samples",
        default=1.0,
        help="Percentage of images in the training set from the whole set",
    )
    parser.add_argument(
        "--batch_size",
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--classes_per_batch",
        default=0,
        help="Classes per Batch. ",
    )
    parser.add_argument('--train_mode', default='hinge',
                        help='ll')

    parser.add_argument(
        "--cuda",
        default=False,
        help="Boolean to decide whether to use special weight initialization (delta orthogonal)",
    )
    args = parser.parse_args()
    train_loader, _, _, _ = get_dataloader(args)
    for idx, (img, label) in enumerate(train_loader):
        print(img.shape)
    # vgg.save_model(epoch=3)
    # vgg.load_models(epoch=3)
    # vgg.load_models_incomplete(epoch=599,single_load=True)
