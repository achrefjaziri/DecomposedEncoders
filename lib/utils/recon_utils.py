import os.path

from tqdm import tqdm
import torch
import imageio
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image


def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(model, dataloader, dataset, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    train_loss = running_loss / counter
    return train_loss


def validate(model, dataloader, dataset, device, criterion):
    model.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
            counter += 1
            data = data[0]
            data = data.to(device)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()

            # save the last batch input and output of every epoch
            if i == int(len(dataset) / dataloader.batch_size) - 1:
                recon_images = reconstruction
    val_loss = running_loss / counter
    return val_loss, recon_images


def train_dec(encoder,model, dataloader, dataset, device, optimizer, criterion,mode='vanilla'):
    model.train()
    encoder.eval()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        if mode=='vanilla':
            idx = 0
        elif mode =='rgNorm':
            idx =1
        elif mode=='lbp':
            idx=2
        elif mode=='dtcwt':
            idx=3
        data_input = data[idx]
        data_input = data_input.to(device)

        data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            output_dic = encoder(data_input)

        img_encodings = output_dic['img_encoding'].detach()  # Shape [128, 1024, 7, 7]
        img_encodings = img_encodings.view(img_encodings.size(0), -1)
        reconstruction, mu, logvar = model(img_encodings)
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    train_loss = running_loss / counter
    return train_loss


def validate_dec(encoder,model, dataloader, dataset, device, criterion,mode='vanilla'):
    model.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
            counter += 1
            if mode == 'vanilla':
                idx = 0
            elif mode == 'rgNorm':
                idx = 1
            elif mode == 'lbp':
                idx = 2
            elif mode == 'dtcwt':
                idx = 3
            data_input = data[idx]
            data_input = data_input.to(device)

            data = data[0]
            data = data.to(device)
            with torch.no_grad():
                output_dic = encoder(data_input)

            img_encodings = output_dic['img_encoding'].detach()  # Shape [128, 1024, 7, 7]
            img_encodings = img_encodings.view(img_encodings.size(0), -1)


            reconstruction, mu, logvar = model(img_encodings)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()

            # save the last batch input and output of every epoch
            if i == int(len(dataset) / dataloader.batch_size) - 1:
                recon_images = reconstruction
    val_loss = running_loss / counter
    return val_loss, recon_images



def train_all(encoder,model, dataloader, dataset, device, optimizer, criterion,encoder2=None, encoder3=None, encoder4=None):
    model.train()
    encoder.eval()
    running_loss = 0.0
    counter = 0
    for i, (vanilla_img, rg_img, lbp_img, wavelet_img, label) in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        vanilla_img = vanilla_img.to(device)  # shape (4,1,64,64)
        rg_img = rg_img.to(device)  # shape (4,1,64,64)
        lbp_img = lbp_img.to(device)  # shape (4,1,64,64)
        wavelet_img = wavelet_img.to(device)  # shape (4,1,64,64)
        optimizer.zero_grad()

        with torch.no_grad():
            #output_dic2 = encoder2(rg_img.detach().contiguous())
            output_dic3 = encoder3(lbp_img.detach().contiguous())
            output_dic = encoder(wavelet_img.detach().contiguous())
            if encoder4 is not None:
                output_dic4 = encoder4(vanilla_img.detach().contiguous())

        img_encodings = output_dic['img_encoding'].detach()  # Shape [128, 1024, 7, 7]

        #img_encodings2 = output_dic2['img_encoding'].detach()  # Shape [128, 1024, 7, 7]
        img_encodings3 = output_dic3['img_encoding'].detach()  # Shape [128, 1024, 7, 7]
        if encoder4 is not None:
            img_encodings4 = output_dic4['img_encoding'].detach()  # Shape [128, 1024, 7, 7]
        # print("img encodings before reshape xxxxxxxxxxxxxxxxxxxxxx",img_encodings.shape,img_encodings2.shape,img_encodings3.shape)

        img_encodings = img_encodings.view(img_encodings.size(0), -1)
        #img_encodings2 = img_encodings2.view(img_encodings2.size(0), -1)
        img_encodings3 = img_encodings3.view(img_encodings3.size(0), -1)

        if encoder4 is None:
            imgs_encodings_all = torch.cat([img_encodings,  img_encodings3], dim=1)
        else:
            img_encodings4 = img_encodings4.view(img_encodings4.size(0), -1)
            imgs_encodings_all = torch.cat([img_encodings,  img_encodings3, img_encodings4],
                                           dim=1)

        #print('img encodings all',imgs_encodings_all.shape)
        reconstruction, mu, logvar = model(imgs_encodings_all)
        bce_loss = criterion(reconstruction, vanilla_img)
        loss = final_loss(bce_loss, mu, logvar)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    train_loss = running_loss / counter
    return train_loss


def validate_all(encoder,model, dataloader, dataset, device, criterion,encoder2=None, encoder3=None, encoder4=None):
    model.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i,  (vanilla_img, rg_img, lbp_img, wavelet_img, label) in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
            counter += 1
            vanilla_img = vanilla_img.to(device)  # shape (4,1,64,64)
            rg_img = rg_img.to(device)  # shape (4,1,64,64)
            lbp_img = lbp_img.to(device)  # shape (4,1,64,64)
            wavelet_img = wavelet_img.to(device)  # shape (4,1,64,64)

            with torch.no_grad():
                #output_dic2 = encoder2(rg_img.detach().contiguous())
                output_dic3 = encoder3(lbp_img.detach().contiguous())
                output_dic = encoder(wavelet_img.detach().contiguous())
                if encoder4 is not None:
                    output_dic4 = encoder4(vanilla_img.detach().contiguous())

            img_encodings = output_dic['img_encoding'].detach()  # Shape [128, 1024, 7, 7]

            #img_encodings2 = output_dic2['img_encoding'].detach()  # Shape [128, 1024, 7, 7]
            img_encodings3 = output_dic3['img_encoding'].detach()  # Shape [128, 1024, 7, 7]
            if encoder4 is not None:
                img_encodings4 = output_dic4['img_encoding'].detach()  # Shape [128, 1024, 7, 7]
            # print("img encodings before reshape xxxxxxxxxxxxxxxxxxxxxx",img_encodings.shape,img_encodings2.shape,img_encodings3.shape)

            img_encodings = img_encodings.view(img_encodings.size(0), -1)
            #img_encodings2 = img_encodings2.view(img_encodings2.size(0), -1)
            img_encodings3 = img_encodings3.view(img_encodings3.size(0), -1)

            if encoder4 is None:
                imgs_encodings_all = torch.cat([img_encodings,  img_encodings3], dim=1) #img_encodings2,
            else:
                img_encodings4 = img_encodings4.view(img_encodings4.size(0), -1)
                imgs_encodings_all = torch.cat([img_encodings,  img_encodings3, img_encodings4],
                                               dim=1)


            reconstruction, mu, logvar = model(imgs_encodings_all)
            bce_loss = criterion(reconstruction, vanilla_img)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()

            # save the last batch input and output of every epoch
            if i == int(len(dataset) / dataloader.batch_size) - 1:
                recon_images = reconstruction
    val_loss = running_loss / counter
    return val_loss, recon_images


to_pil_image = transforms.ToPILImage()
def image_to_vid(images,save_dir):
    imgs = [np.array(to_pil_image(img)) for img in images]
    imageio.mimsave(os.path.join(save_dir,'generated_images.gif'), imgs)
def save_reconstructed_images(recon_images, epoch,save_dir):
    save_image(recon_images.cpu(), os.path.join(save_dir,f"output{epoch}.jpg"))
def save_loss_plot(train_loss, valid_loss,save_dir):
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(valid_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir,'loss.jpg'))
    plt.show()