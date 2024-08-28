# Few imports

import torch
import torchvision
from torchvision import transforms
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split

import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report





# Define the transformations. To begin with, we shall keep it minimum - only resizing the images and converting them to PyTorch tensors

data_transforms = transforms.Compose([
    transforms.Resize([64, 64]),
    transforms.ToTensor()
    ])


# Defining hyperparameters

BATCH_SIZE = 256
learning_rate = 0.001
EPOCHS = 15
numClasses = 43


class GTSRB(Dataset):
    base_folder = 'GTSRB'

    def __init__(self, root_dir, train=False, transform=None,csv_data=None):
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
        if train:
            self.csv_data = csv_data
        else:
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



# Function to calculate accuracy

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


# Function to perform training of the model

def train(model, loader, opt, criterion):
    epoch_loss = 0
    epoch_acc = 0

    # Train the model
    model.train()

    for (images, labels) in loader:
        images = images.cuda()
        labels = labels.cuda()

        # Training pass
        opt.zero_grad()

        output, _ = model(images)
        loss = criterion(output, labels)

        # Backpropagation
        loss.backward()

        # Calculate accuracy
        acc = calculate_accuracy(output, labels)

        # Optimizing weights
        opt.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(loader), epoch_acc / len(loader)


# Function to perform evaluation on the trained model

def evaluate(model, loader, opt, criterion):
    epoch_loss = 0
    epoch_acc = 0

    # Evaluate the model
    model.eval()

    with torch.no_grad():
        for (images, labels) in loader:
            images = images.cuda()
            labels = labels.cuda()

            # Run predictions
            output, _ = model(images)
            loss = criterion(output, labels)

            # Calculate accuracy
            acc = calculate_accuracy(output, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(loader), epoch_acc / len(loader)


class AlexnetTS(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 1000),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(in_features=1000, out_features=256),
            nn.ReLU(inplace=True),

            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.features(x)
        h = x.view(x.shape[0], -1)
        #print("shape input",h.shape)
        x = self.classifier(h)
        return x, h




# Function to count the number of parameters in the model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__=="__main__":

    # Define path of training data
    path_csv = '/home/ajaziri/Thesis_work/src/vision/main/data/GTSRB/trainingset/training.csv'
    csv_data = pd.read_csv(path_csv)

    train_csv, val_csv = train_test_split(csv_data, test_size=0.2)



    print(f"Number of training samples = {len(train_csv)}")
    print(f"Number of validation samples = {len(val_csv)}")

    # Create Datasets
    trainset = GTSRB(
        root_dir='/home/ajaziri/Thesis_work/src/vision/main/data', train=True, transform=data_transforms,csv_data=train_csv)

    valset = GTSRB(
        root_dir='/home/ajaziri/Thesis_work/src/vision/main/data', train=True, transform=data_transforms,
        csv_data=val_csv)
    testset = GTSRB(
        root_dir='/home/ajaziri/Thesis_work/src/vision/main/data', train=False, transform=data_transforms)

    # Load Datasets
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    # Load Datasets
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    # Initialize the model


    model = AlexnetTS(numClasses)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    # Define optimizer and criterion functions

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # If CUDA is available, convert model and loss to cuda variables

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    # Perform training

    # List to save training and val loss and accuracies
    train_loss_list = [0] * EPOCHS
    train_acc_list = [0] * EPOCHS
    val_loss_list = [0] * EPOCHS
    val_acc_list = [0] * EPOCHS

    for epoch in range(EPOCHS):
        print("Epoch-%d: " % (epoch))

        train_start_time = time.monotonic()
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        train_end_time = time.monotonic()
        print("Training: Loss = %.4f, Accuracy = %.4f, Time = %.2f seconds" % (
            train_loss, train_acc, train_end_time - train_start_time))
        val_start_time = time.monotonic()
        val_loss, val_acc = evaluate(model, val_loader, optimizer, criterion)
        val_end_time = time.monotonic()

        train_loss_list[epoch] = train_loss
        train_acc_list[epoch] = train_acc
        val_loss_list[epoch] = val_loss
        val_acc_list[epoch] = val_acc


        print("Validation: Loss = %.4f, Accuracy = %.4f, Time = %.2f seconds" % (
        val_loss, val_acc, val_end_time - val_start_time))
        print("")

    val_start_time = time.monotonic()

    test_loss, test_acc = evaluate(model, testloader, optimizer, criterion)
    val_end_time = time.monotonic()

    print("Test: Loss = %.4f, Accuracy = %.4f, Time = %.2f seconds" % (
        test_loss, test_acc, val_end_time - val_start_time))
    print("")





