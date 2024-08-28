import torchvision.transforms as transforms
import torch
import numpy as np
from dataloaders.lbp_rg_transfo import LBP,NormalizedRG #dataloaders


def lbp_lambda(x):
    lbp_transform = LBP(radius=3, points=24)
    #print('shape in lbp_lambda',x.shape)
    #img_out = torch.Tensor(lbp_transform(x[0].detach().cpu().numpy()))
    all_imgs = []
    transform = transforms.Grayscale()
    for i in range(x.shape[0]):
        current_img_gray = transform(x[i])
        #print("grayscaled shape",current_img_gray.shape)
        current_img = current_img_gray[0].detach().cpu().numpy()
        #print("current image",current_img.shape)
        current_img = np.expand_dims(lbp_transform(current_img), axis=0)
        #print("after lbp transfo",current_img.shape)
        all_imgs.append(current_img)
    all_imgs = np.asarray(all_imgs)
    #print("shape all images", all_imgs.shape)
    img_out = torch.Tensor(all_imgs)

    #img_out=torch.unsqueeze(img_out, 0)
    return img_out

def rg_lambda(x):
    rg_norm = NormalizedRG(conf=False)
    #print('shape i rg_lambda',x.shape)
    all_imgs = []
    for i in range(x.shape[0]):
      current_img= x[i].permute(1, 2, 0).detach().cpu().numpy()
      all_imgs.append(rg_norm(current_img))
    all_imgs = np.asarray(all_imgs)
    #print("shape all images",all_imgs.shape)
    img_out = torch.Tensor(all_imgs).permute(0,3,1,2)
    #print("shape all images final",all_imgs.shape)

    return img_out