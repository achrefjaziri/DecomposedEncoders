"""
This Dataloder loads an input image and its corresponding segmentation mask.
The images are loaded from  an input directory which contains two directories /inputs and /gsround_truths.
This dataloader expects that the all images in the input folder have a ground truth map with the same naming convention.
"""
import numpy as np
import pandas
from PIL import Image
from PIL import ImageOps
import glob
import torch
from torchvision.transforms import transforms
from torch.utils.data.dataset import Dataset
import array
import cv2
from PIL import ImageFile

class rdk_loader(Dataset):
    def __init__(self, csv_file, mode='train',coherence=1.0):
        super(rdk_loader, self).__init__()
        # all file names
        self.path_csv = csv_file
        self.df = pandas.read_csv(csv_file,delimiter='\t')
        print(self.df)
        self.mode =mode
        self.df= self.df.loc[self.df['split'] == self.mode]
        self.df= self.df.loc[self.df['coherence'] == coherence]

        self.transform= transforms.Compose([
            transforms.Resize(120),
            transforms.ToTensor(),
        ])
    def __getitem__(self, index):
        current_row = self.df.iloc[index]
        sequence_path = current_row['Seq_path'].replace('/home/ajaziri/Thesis_work/src/vision/main','/data/aj_data')
        label = current_row['label']
        path_all_imgs = glob.glob(sequence_path+ '/*')

        imgs = [Image.open(img_path).convert('RGB') for img_path in path_all_imgs]
        #print(len(imgs),path_all_imgs,sequence_path)
        seq_tensor = torch.stack([self.transform(img) for img in imgs]).permute(1,0,2,3)
        return seq_tensor,int(label)


    def __len__(self):
        return len(self.df.index)




if __name__ == "__main__":


    Dataset_test = rdk_loader('/home/ajaziri/Thesis_work/src/vision/main/data/rdk/rdk_labels.csv', mode='test',coherence=1.0)

    test_load = \
        torch.utils.data.DataLoader(dataset=Dataset_test,
                                    num_workers=16, batch_size=1, shuffle=False)
    for batch, (imgs,label) in enumerate(test_load):


        print('image', imgs.shape,label)


