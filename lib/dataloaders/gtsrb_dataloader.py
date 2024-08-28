import torch
import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


class GTSRB(Dataset):
    base_folder = 'GTSRB'

    def __init__(self, root_dir, train=False, transform=None):
        """
        Args:
            train (bool): Load trainingset or test set.
            root_dir (string): Directory containing GTSRB folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir

        self.sub_directory = 'trainingset' if train else 'testset'
        self.csv_file_name = 'training.csv' if train else 'test.csv'

        csv_file_path = os.path.join(
            root_dir, self.base_folder, self.sub_directory, self.csv_file_name)

        self.csv_data = pd.read_csv(csv_file_path)

        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.base_folder, self.sub_directory,
                                self.csv_data.iloc[idx, 0])
        img = Image.open(img_path)

        classId = self.csv_data.iloc[idx, 1]

        if self.transform is not None:
            img = self.transform(img)
        #label = torch.LongTensor(classId])

        return img, classId

if __name__=="__main__":



    transform = transforms.Compose([
        #transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3)),
        #transforms.RandomAdjustSharpness(sharpness_factor=2),
        #transforms.RandomAutocontrast(),
        transforms.Resize((48, 48)),
        transforms.RandomCrop((32, 32)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214),
                             (0.2724, 0.2608, 0.2669))
    ])

    # Create Datasets
    trainset = GTSRB(
        root_dir='/home/ajaziri/Thesis_work/src/vision/main/data', train=True, transform=transform)
    testset = GTSRB(
        root_dir='/home/ajaziri/Thesis_work/src/vision/main/data', train=False, transform=transform)

    # Load Datasets
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2)

    for i, (seq, label) in enumerate(trainloader):
        print(seq.shape, label.shape, label)
        """
                for idx in range(seq.shape[3]):
            img = seq[0,0,:,idx]
            print(img.shape)
            plt.Figure()

            plt.imshow(img.permute(1,2,0).detach().cpu().numpy())
            plt.savefig(f"./view_examples/motion_img_example{idx}.png")

        """

