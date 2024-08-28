import argparse
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
import math
import cv2
import numpy as np
# from utils.model_utils import similarity_matrix
from models.ContrastiveLoss import ContrastiveLoss
from models.FA import LinearFA
from utils.model_utils import similarity_matrix, to_one_hot, reshape_to_patches
from pytorch_wavelets import DTCWTForward, DTCWTInverse


class CPC(nn.Module):
    '''
    A block containing nn.Conv2d -> nn.BatchNorm2d -> nn.ReLU -> nn.Dropou2d
    The block can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.

    Args:
        ch_in (int): Number of input features maps.
        ch_out (int): Number of output features maps.
        kernel_size (int): Kernel size in Conv2d.
        stride (int): Stride in Conv2d.
        padding (int): Padding in Conv2d.
        num_classes (int): Number of classes (used in local prediction loss).
        dim_out (int): Feature map height/width for input (and output).
        dropout (float): Dropout rate, if None, read from args.dropout.
        bias (bool): True if to use trainable bias.
        pre_act (bool): True if to apply layer order nn.BatchNorm2d -> nn.ReLU -> nn.Dropou2d -> nn.Conv2d (used for PreActResNet).
        post_act (bool): True if to apply layer order nn.Conv2d -> nn.BatchNorm2d -> nn.ReLU -> nn.Dropou2d.
    '''

    def __init__(self, conv_layers, ch_in, ch_out, opt, block_idx):
        super(CPC, self).__init__()

        self.encoder_num = block_idx
        self.opt = opt
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.opt = opt
        self.overlap = int(opt.overlap_factor)

        self.patch_size = int(opt.patch_size)

        #
        self.patch_average_pool_out_dim = 1

        self.asymmetric_W_pred = self.opt.asymmetric_W_pred
        self.prediction_steps = self.opt.prediction_step  # Number of prediction steps in the future

        # Layer
        self.main_branch = conv_layers

        #
        self.patch_average_pool_out_dim = 1

        in_channels_loss = self.ch_out
        out_channels = self.ch_out

        if self.opt.input_mode == "dtcwt":
            self.xfm = DTCWTForward(J=3, biort='near_sym_b',
                                    qshift='qshift_b')

        # Loss module; is always present, but only gets used when training CLAPPVision modules
        # in_channels_loss: z, out_channels: c

        self.loss = ContrastiveLoss(
            opt,
            in_channels=in_channels_loss,  # z
            out_channels=out_channels,  # c
            prediction_steps=self.prediction_steps,
            save_vars=False
        )
        if self.asymmetric_W_pred:
            self.loss_mirror = ContrastiveLoss(
                opt,
                in_channels=in_channels_loss,
                out_channels=out_channels,
                prediction_steps=self.prediction_steps
            )

        self.optimizer = optim.Adam(self.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay,
                                    amsgrad=opt.optim == 'amsgrad')

        # ([6272, 128, 16, 16]) is the shape of the latents passed to the next layer.
        # ([128, 256, 7, 7]) shape of the encodings passed to the loss function or the classifier: each [:,:,x,y] represents the latent of a specific patch of the orginal image.

        self.clear_stats()

    def clear_stats(self):
        if not self.opt.no_print_stats:
            self.loss_sim = 0.0
            self.loss_pred = 0.0
            self.correct = 0
            self.examples = 0

    def set_learning_rate(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def optim_zero_grad(self):
        self.optimizer.zero_grad()

    def optim_step(self):
        self.optimizer.step()

    def forward(self, x, n_patches_y=None, n_patches_x=None):
        # x: eitherinput dims b, c, Y, X or (if coming from lower module which did unfolding, as variable z): b * n_patches_y * n_patches_x, c, y, x
        # Input preparation, i.e unfolding into patches. Usually only needed for first module. More complicated for experimental increasing_patch_size option.
        # print('ENCODER NUMBER',self.encoder_num)
        # x enters as (4,1,64,64) - (batch_size,channels,height,width)
        if self.opt.dataset != 'UCF101' and self.opt.dataset != 'ER':
            if self.opt.input_mode == "dtcwt":
                x, Yh = self.xfm(x)
            x = (  # b, c, y, x
                x.unfold(2, self.patch_size,
                         self.patch_size // self.overlap)  # b, c, n_patches_y, x, patch_size
                .unfold(3, self.patch_size,
                        self.patch_size // self.overlap)  # b, c, n_patches_y, n_patches_x, patch_size, patch_size
                .permute(0, 2, 3, 1, 4, 5)  # b, n_patches_y, n_patches_x, c, patch_size, patch_size
            )
            # x gets reshaped to (batch_size,7,7,1,16,16) #where patch_size is 16 and overlap is 2
            n_patches_y = x.shape[1]
            n_patches_x = x.shape[2]
            x = x.reshape(
                x.shape[0] * x.shape[1] * x.shape[2], x.shape[3], x.shape[4], x.shape[5]
            )  # b * n_patches_y * n_patches_x, c, patch_size, patch_size
        else:
            # x has shape (batch_size,channels,seq_length,img_size,img_size)
            x = x.permute(0, 2, 1, 3, 4)
            n_patches_y = x.shape[1]
            n_patches_x = 1
            x = x.reshape(
                x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]
            )  # b * n_patches_y * n_patches_x, c, patch_size, patch_size
            if self.opt.input_mode == "dtcwt":
                x, Yh = self.xfm(x)

        # without recurrence split does not really matter, arbitrily choose 1
        split_ind = 1
        # 1. Apply encoding weights of model (e.g. conv2d layer)
        # print('Shape of inputs',x.shape,x.requires_grad)
        # cur_device = x_orig.get_device()
        # x_i = torch.rand(x_input.shape, requires_grad=True).to(cur_device)
        z = self.main_branch(x)  # .clone().detach() # b * n_patches_y * n_patches_x, c, y, x
        # Optional extra conv layer with downsampling (stride > 1) here to increase receptive field size ###
        dec = z
        # Pool over patch
        # in original CPC/GIM, pooling is done over whole patch, i.e. output shape 1 by 1
        out = F.adaptive_avg_pool2d(dec,
                                    self.patch_average_pool_out_dim)  # b * n_patches_y(_reduced) * n_patches_x(_reduced) (* n_extra_patches * n_extra_patches), c, x_pooled, y_pooled

        # for  self.patch_average_pool_out_dim=1: out has shape (198,128) we pooled each (16,16) to 1 for layer 0.

        # Flatten over channel and pooled patch dimensions x_pooled, y_pooled:
        out = out.reshape(out.shape[0],
                          -1)  # b * n_patches_y(_reduced) * n_patches_x(_reduced) (* n_extra_patches * n_extra_patches),  c * y_pooled * x_pooled
        # out has shape (4,7,7,128) we just seperated 198 to 3 dims

        n_p_x, n_p_y = n_patches_x, n_patches_y
        out = out.reshape(-1, n_p_y, n_p_x, out.shape[
            1])  # b, n_patches_y, n_patches_x, c * y_pooled * x_pooled OR  b * n_patches_y(_reduced) * n_patches_x(_reduced), n_extra_patches, n_extra_patches, c * y_pooled * x_pooled
        out = out.permute(0, 3, 1,
                          2).contiguous()  # b, c * y_pooled * x_pooled, n_patches_y, n_patches_x  OR  b * n_patches_y(_reduced) * n_patches_x(_reduced), c * y_pooled * x_pooled, n_extra_patches, n_extra_patches
        output_dic = {'next_layer_input': z, 'rep': out, 'n_patches_y': n_patches_y, 'n_patches_x': n_patches_x}
        return output_dic  # out, z, n_patches_y, n_patches_x

        # If pre-activation, apply batchnorm->nonlin->dropout

    # crop feature map such that the loss always predicts/averages over same amount of patches (as the last one)
    def random_spatial_crop(self, out, n_patches_x, n_patches_y):
        n_patches_x_crop = n_patches_x // (self.max_patch_size // self.patch_size_eff)
        n_patches_y_crop = n_patches_y // (self.max_patch_size // self.patch_size_eff)
        if n_patches_x == n_patches_x_crop:
            posx = 0
        else:
            posx = np.random.randint(0, n_patches_x - n_patches_x_crop + 1)
        if n_patches_y == n_patches_y_crop:
            posy = 0
        else:
            posy = np.random.randint(0, n_patches_y - n_patches_y_crop + 1)
        out = out[:, :, posy:posy + n_patches_y_crop, posx:posx + n_patches_x_crop]
        return out

    def evaluate_loss(self, outs, y, cur_idx=None, y_onehot=None, gating=None):
        '''
        outs: list
        label: Tensor of shape (TBD,)
        '''
        # print('im in evaluate loss now',outs[cur_idx].shape ,label[cur_idx],cur_idx)

        accuracy = torch.zeros(1)
        gating_out = None

        if self.asymmetric_W_pred:  # u = z*W_pred*c -> u = drop_grad(z)*W_pred1*c + z*W_pred2*drop_grad(c)
            # the outs we are using are the mean pooled ones  so (4,nb_filters,7,7).
            loss, loss_gated, _ = self.loss(outs[cur_idx], outs[cur_idx].clone().detach(),
                                            gating=gating)  # z, detach(c)

            loss_mirror, loss_mirror_gated, _ = self.loss_mirror(outs[cur_idx].clone().detach(), outs[cur_idx],
                                                                 gating=gating)  # detach(z), c

            loss = loss + loss_mirror
            loss_gated = loss_gated + loss_mirror_gated
        else:
            loss, loss_gated, gating_out = self.loss(outs[cur_idx], outs[cur_idx],
                                                     gating=gating)  # z, c  #outs is a list that contains tensors of shape (4,nb_channels,7,7)
        output_dic = {'loss': loss, 'loss_gated': loss_gated, 'accuracy': accuracy, 'gatting_out': gating_out}
        return output_dic  # loss, loss_gated, accuracy, gating_out

    def train_step(self, x, n_patches_y=None, n_patches_x=None):
        '''
        A training step for the model
        x: represents the model input
        reps: the representation from the previous layers. Needed in case of reccurence (still work in progress..)
        t: only apply rec if iteration is not the first one (is not entered if recurrence is off since then t = 0)
        n_patches_y, n_patches_x: number of patches on x and y
        label: label of current image
        idx: idx of the current module in the hierarchy
        '''
        self.optimizer.zero_grad()
        output_dic = self.forward(
            x, n_patches_y, n_patches_x
        )
        z, h, n_patches_y, n_patches_x = output_dic['next_layer_input'], output_dic['rep'], output_dic['n_patches_y'], \
                                         output_dic['n_patches_x']
        # out: mean pooled per patch for each z. For example z shape is (198,128,16,16)  the mean pooling is (4,128,7,7)  where 198=4*7*7 = batch_size*img_patch_size*img_patch_size

        if self.asymmetric_W_pred:  # u = z*W_pred*c -> u = drop_grad(z)*W_pred1*c + z*W_pred2*drop_grad(c)
            # the outs we are using are the mean pooled ones  so (4,nb_filters,7,7).
            loss, loss_gated, _ = self.loss(h, h.clone().detach())  # z, detach(c)

            loss_mirror, loss_mirror_gated, _ = self.loss_mirror(h.clone().detach(), h)  # detach(z), c

            loss = loss + loss_mirror
            loss_gated = loss_gated + loss_mirror_gated
        else:
            loss, loss_gated, gating_out = self.loss(h,
                                                     h)  # z, c  #outs is a list that contains tensors of shape (4,nb_channels,7,7)

        # print('Loss calculation done...',loss)

        # print('updating weights...')

        loss.backward()  # retain_graph=True

        self.optimizer.step()
        next_layer_input = z.clone().detach()  # full module output # This sets requires_grad to False

        return {'next_layer_input': next_layer_input, 'rep': h, 'loss': loss, 'n_patches_y': n_patches_y,
                'n_patches_x': n_patches_x}
        # next_layer_input, h, loss, n_patches_y, n_patches_x

    def add_filter(self, writer, idx, epoch):
        filter = self.main_branch[0].weight.data.clone()
        print(filter.shape)
        res = cv2.resize(filter[10, 0].detach().cpu().numpy(), dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
        # writer.add_images(f'Filter for layer {idx} after {epoch} epochs', res,dataformats='HW')
        print("filter added")
