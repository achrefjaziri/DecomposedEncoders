import cv2
import numpy as np
import scipy
import random
import kornia
from skimage import feature

from kornia.augmentation.container.image import ImageSequential, ParamItem
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms




import torch
import torchvision
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset, TensorDataset
import torchvision.transforms as transforms

import numpy as np
import os
import random
# import cv2
import xml.etree.ElementTree as ElementTree
from PIL import Image, ImageOps
import pandas as pd

from tqdm import tqdm

import torch
import torchvision
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset, TensorDataset
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from PIL import Image
import random
import scipy

# from skimage import color, exposure, transform

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def load_image_np(path):
    img = Image.open(path)
    img = img.convert('RGB')
    return np.asarray(img)


NO_ROI = 0
APPLY_ROI = 1
RETURN_ROI = 2


class Roi_image_loader():
    def __init__(self, path, roi_mode=NO_ROI):
        self.rois = {}
        self.roi_mode = roi_mode
        if os.path.isdir(path):
            # if (path).is_dir():
            root = path
            classes = [d.name for d in os.scandir(root) if d.is_dir()]
            for c in classes:
                csv_path = root + c + "/GT-" + c + ".csv"
                self.read_rois_from_csv(root + c + "/", csv_path)
        elif path.endswith(".csv"):
            splited_path = path.split('/')
            root = path[:-len(splited_path[-1])]
            self.read_rois_from_csv(root, path)

    def read_rois_from_csv(self, root, path, verbose=False):
        data = pd.read_csv(path, sep=";")
        for i in range(data.shape[0]):
            filename = data['Filename'][i]
            rois = (data['Roi.X1'][i], data['Roi.Y1'][i], data['Roi.X2'][i], data['Roi.Y2'][i])
            self.rois.update({root + filename: rois})
            if verbose:
                print(root + filename)

    def __call__(self, path):
        with open(path, 'rb') as f:
            img = load_image_np(f)
            if self.roi_mode == NO_ROI:
                return img
            crop_area = self.rois[path]
            if self.roi_mode == APPLY_ROI:
                return img[crop_area]
            return img, crop_area


RED_c = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]  # Circle with red margin
RED_cf = [14, 17]  # Filled Red
BLUE_cf = [33, 34, 35, 36, 37, 38, 39, 40]  # Blue filled circle
RED_t = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]  # Triangle Red Surrounding upword
RED_t2 = [13]  # Triangle red(vorfahrt)
GRAY = [6, 32, 41, 42]  # Gray
YELLOW = [12]  # yellow rectangle(vorfahrt)




class Contrastive_Tensor_Dataset_From_PIL(Dataset):
    def __init__(self, inputs, targets, dynamic_transforms, target_transforms):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.inputs = inputs
        self.targets = targets
        self.dynamic_transforms = dynamic_transforms
        self.target_transforms = target_transforms
        self.aug_list =  ImageSequential(
            kornia.augmentation.ColorJiggle(0.15, 0.15, 0.15, 0.15, p=1.0),
            kornia.filters.MedianBlur((3, 3)),

            kornia.augmentation.RandomAffine(360, p=0.5),

        )


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        next_input = self.inputs[idx]
        next_target = self.targets[idx]





        if self.dynamic_transforms:
            anchor = self.dynamic_transforms(next_input)
            anchor = self.aug_list(anchor)
            positive_example = self.dynamic_transforms(next_input)
            positive_example = self.aug_list(positive_example)



            random_idx = random.randint(0,len(self.inputs))
            neg_input = self.inputs[random_idx]
            neg_example = self.dynamic_transforms(neg_input)
            neg_example = self.aug_list(neg_example)




            return anchor,positive_example,neg_example
        else: return next_input

class CustomDataset(Dataset):
    def __init__(self,paths):
        self.data_left = []
        for path in paths:
            file_list = glob.glob(path + "/*")

            self.data_left +=file_list



        self.tw = 224
        self.th = 224

        self.aug_list = ImageSequential(
            kornia.augmentation.ColorJiggle(0.15, 0.15, 0.15, 0.15, p=1.0),
            kornia.filters.MedianBlur((3, 3)),

            kornia.augmentation.RandomAffine(360, p=0.5),

        )

        self.img_transformations = transforms.Compose([
            transforms.ColorJitter(brightness=(0,0.15),contrast=(0,0.3),saturation=(0,0.1),hue=.3),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.ToTensor()])








    def clean_up_data(self):

        for idx,img_path_left in enumerate(self.data_left):
            print(idx)
            try:
                img_path_right = img_path_left.replace('left','right').replace('FF005096','FF005097')
                img_left = cv2.imread(img_path_left)
                img_tensor_left = torch.from_numpy(img_left)
                img_tensor_left = img_tensor_left.permute(2, 0, 1)


                img_right = cv2.imread(img_path_right)
                img_tensor_right = torch.from_numpy(img_right)
                img_tensor_right = img_tensor_right.permute(2, 0, 1)

                if img_tensor_left is None or img_tensor_right is None:
                    print('None tensors', img_path_right, img_path_left)
                    self.data_left.pop(idx)
            except:
                print('paths broken',img_path_right,img_path_left)
                self.data_left.pop(idx)







    def __len__(self):
        return len(self.data_left)
    def __getitem__(self, idx):


        img_tensor_left,img_tensor_right = None,None
        idx_data = idx
        while img_tensor_left is None and img_tensor_right is None:

            #try:
            img_path_left = self.data_left[idx_data]
            img_path_right = img_path_left.replace('left','right').replace('FF005096','FF005097')

            img_left = Image.open(img_path_left)

            img_right = Image.open(img_path_right)

            w, h = img_left.size

            x1 = random.randint(0, w - self.tw)
            y1 = random.randint(0, h - self.th)

            img_left =img_left.crop((x1, y1, x1 + self.tw, y1 + self.th))
            img_right =img_right.crop((x1, y1, x1 + self.tw, y1 + self.th))


            img_left = np.asarray(img_left)
            img_right = np.asarray(img_right)


            img_tensor_left = torch.from_numpy(img_left)
            #

            #img_tensor_left = self.img_transformations(img_left)
            img_tensor_left = img_tensor_left.permute(2, 0, 1)

            #img_tensor_right = self.img_transformations(img_right)

            img_tensor_right = torch.from_numpy(img_right)
            img_tensor_right = img_tensor_right.permute(2, 0, 1)


        #except:
            #    print('im here!!')
            #    idx_data = random.randint(0, len(self.data_left))



        return img_tensor_left, img_tensor_right



if __name__=="__main__":
    import matplotlib.pyplot as plt
    dataset = CustomDataset(['/data/resist_data/datasets/photos_stereo_greece/mission_1/1647518564/left',
                             '/data/resist_data/datasets/photos_stereo_greece/mission_3/left',
                             '/data/resist_data/datasets/photos_stereo_greece/mission_4_3m/left',
                             '/data/resist_data/datasets/photos_stereo_greece/mission_5/left',
                             '/data/resist_data/datasets/photos_stereo_greece/mission_5_re/left'])
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True,num_workers=8)
    img_tensor_left, img_tensor_right = next(iter(data_loader))
    for i,(img_tensor_left, img_tensor_right) in enumerate(data_loader):
        print('shapes', i,img_tensor_left.shape, img_tensor_right.shape)


        for idx in range(img_tensor_left.shape[0]):
            img = img_tensor_left[idx]#[0]
            print(img.shape)
            plt.Figure()

            plt.imshow(img.permute(1, 2, 0).detach().cpu().numpy())
            plt.savefig(f"./view_examples/img_tensor_left{idx}.png")

            img = img_tensor_right[idx]#[0]
            print(img.shape)
            plt.Figure()

            plt.imshow(img.permute(1, 2, 0).detach().cpu().numpy())
            plt.savefig(f"./view_examples/img_tensor_right{idx}.png")
        break

    
