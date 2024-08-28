import argparse

import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
import math
import cv2
import numpy as np
import kornia
from skimage import feature

# import kornia.augmentation.container.ImageSequential as ImageSequential
from kornia.augmentation.container.image import ImageSequential, ParamItem
#from utils.model_utils import similarity_matrix
from models.ContrastiveLoss import ContrastiveLoss
from models.DorsalLoss import DorsalLoss
from models.FA import LinearFA
from utils.model_utils import similarity_matrix, to_one_hot,reshape_to_patches,prepare_labels
from pytorch_wavelets import DTCWTForward, DTCWTInverse


def get_scaling_factor(cfg,encoder_num):
    pooling_counter = 0
    for idx,val in enumerate(cfg):
        if val=='M':
            pooling_counter+=1
        else:
            encoder_num -=1
        if encoder_num<0:
            break

    return (2**pooling_counter)



class SingleLinearLayer(nn.Module):
    """
    Convenience function defining a single block consisting of a fully connected (linear) layer followed by
    batch normalization and a rectified linear unit activation function.
    """
    def __init__(self, l, fan_in, fan_out, batch_norm=1e-5):
        super(SingleLinearLayer, self).__init__()

        self.fclayer = nn.Sequential(OrderedDict([
            ('fc' + str(l), nn.Linear(fan_in, fan_out, bias=False)),
        ]))

        if batch_norm > 0.0:
            self.fclayer.add_module('bn' + str(l), nn.BatchNorm1d(fan_out, eps=batch_norm))

        self.fclayer.add_module('act' + str(l), nn.ReLU())

    def forward(self, x):
        x = self.fclayer(x)
        return x

class LocalLossBlockConv(nn.Module):
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

    def __init__(self, ch_in, ch_out, kernel_size, stride, padding, num_classes, dim_out,opt,block_idx,
                 dropout=None, bias=None, pre_act=False, post_act=True,patch_size=16,   overlap_factor=2,cfg=None):
        super(LocalLossBlockConv, self).__init__()
        self.testing_bool = True
        self.encoder_num = block_idx
        self.opt = opt
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.opt = opt
        self.num_classes = num_classes
        self.dropout_p = float(opt.dropout) if dropout is None else dropout
        self.bias = True if bias is None else bias
        self.pre_act = pre_act
        self.post_act = post_act

        scaling_factor = get_scaling_factor(cfg, self.encoder_num)
        # Layer
        self.main_branch = self.make_layers(ch_in,ch_out,kernel_size, stride, padding)

        if self.opt.train_mode == 'hinge' or self.opt.train_mode =='var_hinge':
            # Training each layer seperatly according to CLAPP loss. Paper:https://arxiv.org/abs/2010.08262

            self.overlap = int(opt.overlap_factor)

            self.patch_size = int(opt.patch_size)

            #
            self.patch_average_pool_out_dim = 1

            self.asymmetric_W_pred = self.opt.asymmetric_W_pred
            self.prediction_steps = self.opt.prediction_step #Number of prediction steps in the future
            in_channels_loss = self.ch_out
            out_channels = self.ch_out

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

            self.optimizer_hebb = optim.Adam(self.parameters(), lr=0.01/64)

            if self.opt.input_mode == "dtcwt":
                self.xfm = DTCWTForward(J=3, biort='near_sym_b',
                                        qshift='qshift_b')


                #([6272, 128, 16, 16]) is the shape of the latents passed to the next layer.
            # ([128, 256, 7, 7]) shape of the encodings passed to the loss function or the classifier: each [:,:,x,y] represents the latent of a specific patch of the orginal image.

            if self.opt.train_mode =='var_hinge':
                scaling_factor = get_scaling_factor(cfg,self.encoder_num)
                self.scaled_patch = int(patch_size/scaling_factor)
                mu_input_dim = int(self.ch_out * (patch_size/scaling_factor)**2)
                self.latent_mu = LinearFA(mu_input_dim,
                                           200,self.opt, bias=False)
                self.latent_std = LinearFA(mu_input_dim,
                                            200, self.opt,bias=False)
                self.latent_decoder_vae = LinearFA( 200,mu_input_dim,self.opt, bias=False)

                self.optimizer_latent_mu = optim.Adam(self.latent_mu.parameters(), lr=self.opt.lr_var,
                                            weight_decay=self.opt.weight_decay,
                                            amsgrad=opt.optim == 'amsgrad')
                self.optimizer_latent_std = optim.Adam(self.latent_std.parameters(), lr=self.opt.lr_var,
                                            weight_decay=self.opt.weight_decay,
                                            amsgrad=opt.optim == 'amsgrad')
                self.optimizer_latent_decoder_vae = optim.Adam(self.latent_decoder_vae.parameters(), lr=self.opt.lr_var,
                                            weight_decay=self.opt.weight_decay,
                                            amsgrad=opt.optim == 'amsgrad')

                self.relu_vae = nn.LeakyReLU()
                modules = []
                hidden_dims = [self.ch_out,self.ch_out//2] #self.ch_out//4
                hidden_dim_input = self.ch_out
                if scaling_factor > 1:
                    nb_of_convs = int(np.log2(scaling_factor)) + 2
                    for i in range(nb_of_convs):
                        modules.append(
                            nn.Sequential(
                                nn.ConvTranspose2d(hidden_dim_input,
                                                   hidden_dim_input//2,
                                                   kernel_size=3,
                                                   stride=2,
                                                   padding=1,
                                                   output_padding=1),
                                nn.BatchNorm2d( hidden_dim_input//2),
                                nn.LeakyReLU())
                        )
                        hidden_dim_input = hidden_dim_input // 2

                modules.append(nn.Sequential(
                nn.Conv2d(hidden_dim_input, out_channels=self.opt.input_ch,
                          kernel_size=3, padding=1),
                nn.Tanh()))


                self.decoder_vae = nn.Sequential(*modules)


                self.optimizer_decoder_vae = optim.Adam(self.decoder_vae.parameters(), lr=self.opt.lr_var,
                                                      weight_decay=self.opt.weight_decay,
                                                      amsgrad=opt.optim == 'amsgrad')

        elif self.opt.train_mode == 'dorsal':
            self.patch_average_pool_out_dim = 1
            self.loss_criterion = torch.nn.CrossEntropyLoss()


            self.overlap = int(opt.overlap_factor)

            self.patch_size = int(opt.patch_size)

            self.prediction_steps = 3
            self.projection_layer = nn.Linear(in_features= self.prediction_steps * ch_out, out_features=128)
            self.relu = nn.ReLU()
            self.classification_layer = nn.Linear(in_features=128, out_features=8)

            #self.optimizer = optim.Adam(self.parameters(), lr=5e-5, weight_decay=self.opt.weight_decay,
            #                            amsgrad=opt.optim == 'amsgrad')

            if True:
                scaling_factor = get_scaling_factor(cfg, self.encoder_num)
                self.scaled_patch = int(patch_size / scaling_factor)
                mu_input_dim = int(self.prediction_steps * self.ch_out * (patch_size / scaling_factor) ** 2)
                latent_dec_input_dim = int( self.ch_out * (patch_size / scaling_factor) ** 2)

                self.latent_mu = LinearFA(mu_input_dim,
                                          200, self.opt, bias=False)
                self.latent_std = LinearFA(mu_input_dim,
                                           200, self.opt, bias=False)
                self.latent_decoder_vae = LinearFA(200, latent_dec_input_dim, self.opt, bias=False)

                #self.optimizer_latent_mu = optim.Adam(self.latent_mu.parameters(), lr=self.opt.lr_var,
                #                                      weight_decay=self.opt.weight_decay,
                #                                      amsgrad=opt.optim == 'amsgrad')
                #self.optimizer_latent_std = optim.Adam(self.latent_std.parameters(), lr=self.opt.lr_var,
                #                                       weight_decay=self.opt.weight_decay,
                #                                       amsgrad=opt.optim == 'amsgrad')
                #self.optimizer_latent_decoder_vae = optim.Adam(self.latent_decoder_vae.parameters(), lr=self.opt.lr_var,
                #                                               weight_decay=self.opt.weight_decay,
                #                                               amsgrad=opt.optim == 'amsgrad')

                self.relu_vae = nn.LeakyReLU()
                modules = []
                hidden_dim_input = self.ch_out
                nb_of_convs = int(np.log2(scaling_factor)) +2
                for i in range(nb_of_convs):
                    modules.append(
                        nn.Sequential(
                            nn.ConvTranspose2d(hidden_dim_input,
                                               hidden_dim_input // 2,
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dim_input // 2),
                            nn.LeakyReLU())
                    )
                    hidden_dim_input = hidden_dim_input // 2

                modules.append(nn.Sequential(
                    nn.Conv2d(hidden_dim_input, out_channels=self.opt.input_ch,
                              kernel_size=3, padding=1),
                    nn.Tanh()))

                self.decoder_vae = nn.Sequential(*modules)

                #self.optimizer_decoder_vae = optim.Adam(self.decoder_vae.parameters(), lr=self.opt.lr_var,
                #                                        weight_decay=self.opt.weight_decay,
                #                                        amsgrad=opt.optim == 'amsgrad')
                self.optimizer = optim.Adam(self.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay, #5e-6
                                            amsgrad=opt.optim == 'amsgrad')
        elif self.opt.train_mode == 'predSim':
            # Training with local Error Single and PredSim loss. Paper:https://arxiv.org/pdf/1901.06656.pdf
            ks_h, ks_w = 1, 1
            dim_out_h, dim_out_w = dim_out, dim_out
            dim_in_decoder = ch_out * dim_out_h * dim_out_w
            print(dim_in_decoder,ks_h)
            if ks_h > 1 or ks_w > 1:
                pad_h = (ks_h * (dim_out_h - dim_out // ks_h)) // 2
                pad_w = (ks_w * (dim_out_w - dim_out // ks_w)) // 2
                self.avg_pool = nn.AvgPool2d((ks_h, ks_w), padding=(pad_h, pad_w))
            else:
                self.avg_pool = None

            self.decoder_y = nn.Linear(dim_in_decoder, num_classes)

            self.decoder_y.weight.data.zero_()

            self.conv_loss = nn.Conv2d(ch_out, ch_out, 3, stride=1, padding=1, bias=False)

            self.optimizer = optim.Adam(self.main_branch.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay,
                                        amsgrad=opt.optim == 'amsgrad')
            self.optimizer_decoder_y = optim.Adam(self.decoder_y.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay,
                                        amsgrad=opt.optim == 'amsgrad')
            self.optimizer_conv_loss= optim.Adam(self.conv_loss.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay,
                                        amsgrad=opt.optim == 'amsgrad')
        else:
            raise NotImplementedError(f'{self.opt.train_mode} is not implemented.')

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
        if self.opt.train_mode=="hinge"  or self.opt.train_mode =='var_hinge':
            #x: eitherinput dims b, c, Y, X or (if coming from lower module which did unfolding, as variable z): b * n_patches_y * n_patches_x, c, y, x
            # Input preparation, i.e unfolding into patches. Usually only needed for first module. More complicated for experimental increasing_patch_size option.
            # print('ENCODER NUMBER',self.encoder_num)
            if self.encoder_num == 0:
                if self.opt.dataset != 'UCF101' and self.opt.dataset!='ER':
                    if self.opt.input_mode == "vanilla" or self.opt.input_mode == "lbp" or self.opt.input_mode == 'rgNorm':

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

                    elif self.opt.input_mode == "dtcwt":
                        # self.xfm = DTCWTForward(J=3, biort='near_sym_b',
                        #                   qshift='qshift_b')  # Accepts all wave types available to PyWavelets

                        x, Yh = self.xfm(x)
                        # print("dtcwt shapes",x.shape,Yl.shape)
                        x = (  # b, c, y, x
                            x.unfold(2, self.patch_size,
                                     self.patch_size // self.overlap, )  # b, c, n_patches_y, x, patch_size
                                .unfold(3, self.patch_size,
                                        self.patch_size // self.overlap)  # b, c, n_patches_y, n_patches_x, patch_size, patch_size
                                .permute(0, 2, 3, 1, 4, 5)  # b, n_patches_y, n_patches_x, c, patch_size, patch_size
                        )
                        # x gets reshaped to (batch_size,7,7,1,16,16) #where patch_size is 16 and overlap is 2
                        n_patches_y = x.shape[1]
                        n_patches_x = x.shape[2]
                        # TODO change this line
                        x = x.reshape(
                            x.shape[0] * x.shape[1] * x.shape[2], x.shape[3], x.shape[4], x.shape[5]
                        )  # b * n_patches_y * n_patches_x, c, patch_size, patch_size
                        # X = torch.randn(16, 3, 256, 256)  # 124


                        # x gets reshaped to (batch_Size* n_patches_y * n_patches_x, nb_channels,patch_size,patch_size) - (196,1,16,16)
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

                        # x gets reshaped to (batch_Size* n_patches_y * n_patches_x, nb_channels,patch_size,patch_size) - (196,1,16,16)

            # without recurrence split does not really matter, arbitrily choose 1
            split_ind = 1
            # 1. Apply encoding weights of model (e.g. conv2d layer)
            # print('Shape of inputs',x.shape,x.requires_grad)
            # cur_device = x_orig.get_device()
            # x_i = torch.rand(x_input.shape, requires_grad=True).to(cur_device)
            z = self.main_branch[:split_ind](x)  # .clone().detach() # b * n_patches_y * n_patches_x, c, y, x

            # 3. Apply nonlin and 'rest' of model (e.g. ReLU, MaxPool etc...)
            z = self.main_branch[split_ind:](z)  # b * n_patches_y * n_patches_x, c, y, x
            # Optional extra conv layer with downsampling (stride > 1) here to increase receptive field size ###
            dec = z
            if self.opt.train_mode =='var_hinge':
                    z_view = z.clone().detach()
                    z_view = z_view.view(z_view.size(0), -1)
                    #print('VIEW z',z_view.shape)
                    z_mean = self.latent_mu(z_view)
                    z_logvar = self.latent_std(z_view)
                    std = torch.exp(0.5 * z_logvar)
                    eps = std.data.new(std.size()).normal_()
                    z_sampled_compressed =  eps.mul(std).add(z_mean)
                    z_sampled = self.latent_decoder_vae(z_sampled_compressed)
                    #z_sampled = self.relu_vae(z_sampled)
                    z_sampled = z_sampled.view(-1, self.ch_out, self.scaled_patch, self.scaled_patch)
                    next_layer_input = z.clone().detach() + z_sampled.clone().detach()
            else:
                next_layer_input = z.clone().detach()

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
            output_dic ={'next_layer_input':next_layer_input,'rep':out,'n_patches_y':n_patches_y,'n_patches_x':n_patches_x,'z_for_hebb':z}
            return output_dic #out, z, n_patches_y, n_patches_x

        elif self.opt.train_mode == 'dorsal':
            if self.encoder_num == 0:
                #if not self.opt.test_mode:
                    x = (  # b, c, y, x
                        x.unfold(2, self.patch_size,
                                 self.patch_size // self.overlap)  # b, c, n_patches_y, x, patch_size
                            .unfold(3, self.patch_size,
                                    self.patch_size // self.overlap)  # b, c, n_patches_y, n_patches_x, patch_size, patch_size
                            .permute(0, 2, 3, 1, 4, 5)  # b, n_patches_y, n_patches_x, c, patch_size, patch_size
                    )
                    # x enters as (4,c,64,64) - (batch_size,channels,height,width)
                    n_patches_y = self.prediction_steps #x.shape[1]
                    n_patches_x = 1 #x.shape[2]
                    x,self.labels = prepare_labels(x,self.prediction_steps)
                #else:
                #    x = (  # b, c, y, x
                #        x.unfold(2, self.patch_size,
                #                 self.patch_size // self.overlap)  # b, c, n_patches_y, x, patch_size
                #            .unfold(3, self.patch_size,
                #                    self.patch_size // self.overlap)  # b, c, n_patches_y, n_patches_x, patch_size, patch_size
                #            .permute(0, 2, 3, 1, 4, 5)  # b, n_patches_y, n_patches_x, c, patch_size, patch_size
                #    )
                #    # x gets reshaped to (batch_size,7,7,1,16,16) #where patch_size is 16 and overlap is 2
                #    n_patches_y = x.shape[1]
                #    n_patches_x = x.shape[2]
                #    x = x.reshape(
                #        x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4], x.shape[5]
                #    ).permute(0,2,1,3,4)  # b * n_patches_y * n_patches_x, c, patch_size, patch_size

                # x gets reshaped to (batch_Size* n_patches_y * n_patches_x, nb_channels,patch_size,patch_size) - (196,1,16,16)

            #print("shape before input",x.shape) #torch.Size([16, 1, 3, 16, 16])
            z = self.main_branch(x)# .clone().detach() # b * n_patches_y * n_patches_x, c, y, x


            dec = z

            next_layer_input = z.clone().detach()

            # Pool over patch
            # in original CPC/GIM, pooling is done over whole patch, i.e. output shape 1 by 1
            out = F.adaptive_avg_pool2d(dec.permute(0,2,1,3,4).reshape(-1,z.shape[1],z.shape[3],z.shape[4]),
                                        self.patch_average_pool_out_dim)

            # Flatten over channel and pooled patch dimensions x_pooled, y_pooled:
            out = out.reshape(out.shape[0],
                              -1)
            n_p_x, n_p_y = n_patches_x, n_patches_y
            out = out.reshape(-1, n_p_y, n_p_x, z.shape[
                1])  # b, n_patches_y, n_patches_x, c * y_pooled * x_pooled OR  b * n_patches_y(_reduced) * n_patches_x(_reduced), n_extra_patches, n_extra_patches, c * y_pooled * x_pooled
            out = out.permute(0, 3, 1,
                              2).contiguous()  # b, c * y_pooled * x_pooled, n_patches_y, n_patches_x  OR  b * n_patches_y(_reduced) * n_patches_x(_reduced), c * y_pooled * x_pooled, n_extra_patches, n_extra_patches
            output_dic = {'next_layer_input': next_layer_input, 'rep': out, 'n_patches_y': n_patches_y,
                          'n_patches_x': n_patches_x}
            return output_dic  # out, z, n_patches_y, n_patches_x

        elif self.opt.train_mode == 'predSim':
            # Training with local Error Single and PredSim loss. Paper:https://arxiv.org/pdf/1901.06656.pdf


            # Pass through conv-bn-nonlinearity
            h = self.main_branch(x)

            # Save return value and add dropout
            h_return = h
            if self.dropout_p > 0:
                h_return = self.dropout(h_return)

            output_dic ={'next_layer_input':h_return,'rep':h,'n_patches_y':n_patches_y,'n_patches_x':n_patches_x}


            return output_dic #h_return, h

        else:
            raise NotImplementedError(f'{self.opt.train_mode} is not implemented.')

        # If pre-activation, apply batchnorm->nonlin->dropout

    def make_layers(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1):

        #if self.dropout_p > 0:
        #    self.dropout = torch.nn.Dropout2d(p=self.dropout_p, inplace=False)

        layers = []
        if self.opt.train_mode=='dorsal':
            conv2d = nn.Conv3d(ch_in, ch_out, kernel_size=kernel_size,stride=stride, padding=padding)
        else:
            conv2d = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size,stride=stride, padding=padding)

        if self.opt.nonlin == 'relu':
            nonlin = nn.ReLU(inplace=True)
        elif self.opt.nonlin == 'leakyrelu':
            nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        if not self.opt.no_batch_norm:
            if self.opt.train_mode == 'dorsal':
                bn = torch.nn.InstanceNorm3d(ch_out, eps=1e-05, momentum=0.1)
            else:
                bn = torch.nn.BatchNorm2d(ch_out)
                nn.init.constant_(bn.weight, 1)
                nn.init.constant_(bn.bias, 0)

            layers += [conv2d, bn, nonlin]
        else:
            layers += [conv2d, nonlin]
        return nn.Sequential(*layers)

    #crop feature map such that the loss always predicts/averages over same amount of patches (as the last one)
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

    def evaluate_loss(self, outs, y,cur_idx=None,y_onehot=None, gating=None):
        '''
        outs: list
        label: Tensor of shape (TBD,)
        '''
        # print('im in evaluate loss now',outs[cur_idx].shape ,label[cur_idx],cur_idx)

        if self.opt.train_mode=='hinge' or self.opt.train_mode =='var_hinge':
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
            output_dic ={'loss':loss,'loss_gated':loss_gated,'accuracy':accuracy,'gatting_out':gating_out}
            return output_dic #loss, loss_gated, accuracy, gating_out
        elif self.opt.train_mode=='contrastive':
            pass
        elif self.opt.train_mode =='predSim':
            h_loss = self.conv_loss(outs)
            Rh = similarity_matrix(h_loss)
            h = outs
            if self.avg_pool is not None:
                h = self.avg_pool(outs)
            y_hat_local = self.decoder_y(h.view(h.size(0), -1))
            Ry = similarity_matrix(y_onehot).detach()
            loss_pred = (1 - self.opt.beta) * F.cross_entropy(y_hat_local, y.detach())
            loss_sim = self.opt.beta * F.mse_loss(Rh, Ry)
            loss = loss_pred + loss_sim
            if not self.opt.no_print_stats:
                self.loss_pred += loss_pred.item() * h.size(0)
                self.loss_sim += loss_sim.item() * h.size(0)
                self.correct += y_hat_local.max(1)[1].eq(y).cpu().sum()
                self.examples += h.size(0)
            output_dic ={'loss':loss}

            return output_dic


    def train_step(self, x,y=None, y_onehot=None, n_patches_y=None, n_patches_x=None,orig_imgs=None,labels=None):
        '''
        A training step for the model
        x: represents the model input
        reps: the representation from the previous layers. Needed in case of reccurence (still work in progress..)
        t: only apply rec if iteration is not the first one (is not entered if recurrence is off since then t = 0)
        n_patches_y, n_patches_x: number of patches on x and y
        label: label of current image
        idx: idx of the current module in the hierarchy
        '''
        if self.opt.train_mode =='hinge' or self.opt.train_mode =='var_hinge':

            if self.opt.hebb_extension!='no_hebb':
                #Hebbian pass
                self.optimizer_hebb.zero_grad()

                output_dic = self.forward(
                    x, n_patches_y, n_patches_x
                )
                z_hebb, h_hebb, n_patches_y, n_patches_x = output_dic['z_for_hebb'],output_dic['rep'],output_dic['n_patches_y'],output_dic['n_patches_x']

                prelimy = z_hebb
                # We now compute the "real" output realy, with a k-WTA
                realy = prelimy.clone().detach()  # We don’t want to affect the graph
                tk = torch.topk(realy.data, 1, dim=1, largest=True)[0]
                realy.data[realy.data < tk.data[:, -1, :, :][:, None, :, :]] = 0
                realy.data = (realy.data > 0).float()  # Binarizing
                # We now compute the surrogate output y, used only to set up the
                # proper computational graph.
                #yforgrad = prelimy  # Plain Hebb, dw ~= xy
                # The following expressions implement the Instar rule (dw ~= y(x-w)) and Oja’s
                # rule (dw ~= y(x-wy)), respectively. Note the dimensional rearrangements and
                # broadcasting.
                # yforgrad = prelimy - 1/2 * torch.sum(w * w, dim=(1,2,3))[None,:, None, None]
                #yforgrad1 = prelimy
                yforgrad = prelimy - 1/2 * torch.sum(self.main_branch[0].weight * self.main_branch[0].weight, dim=(1,2,3))[None,:, None, None] * realy.data
                # We overwrite the values of yforgrad with the "real" y.
                yforgrad.data = realy.data
                # Compute the loss and perform the backward pass, which applies the
                # desired Hebbian updates.
                loss_hebb = torch.sum(-1 / 2 * yforgrad * yforgrad)
                #loss_hebb1 = torch.sum(-1 / 2 * yforgrad1 * yforgrad1)

                loss_hebb.backward()
                self.optimizer_hebb.step()
                #print(loss_hebb,loss_hebb1,z_hebb.shape)





            self.optimizer.zero_grad()

            #Nero-modulated pass
            output_dic= self.forward(
                x,  n_patches_y, n_patches_x
            )
            z, h, n_patches_y, n_patches_x = output_dic['next_layer_input'],output_dic['rep'],output_dic['n_patches_y'],output_dic['n_patches_x']
            # out: mean pooled per patch for each z. For example z shape is (198,128,16,16)  the mean pooling is (4,128,7,7)  where 198=4*7*7 = batch_size*img_patch_size*img_patch_size

            if self.asymmetric_W_pred:  # u = z*W_pred*c -> u = drop_grad(z)*W_pred1*c + z*W_pred2*drop_grad(c)
                if self.opt.train_mode != 'hinge' and self.opt.train_mode!='var_hinge':
                    raise ValueError("asymmetric_W_pred only implemented for hinge contrasting!")
                # the outs we are using are the mean pooled ones  so (4,nb_filters,7,7).
                loss, loss_gated, _ = self.loss(h, h.clone().detach())  # z, detach(c)

                loss_mirror, loss_mirror_gated, _ = self.loss_mirror(h.clone().detach(), h)  # detach(z), c
                #print("loss value mse and clapp",tt+tt_m,mse+mse_m)
                loss = loss + loss_mirror
                loss_gated = loss_gated + loss_mirror_gated
            else:
                loss, loss_gated, gating_out = self.loss(h, h)  # z, c  #outs is a list that contains tensors of shape (4,nb_channels,7,7)

            # print('Loss calculation done...',loss)

            # print('updating weights...')

            loss.backward()  # retain_graph=True

            self.optimizer.step()
            if self.opt.train_mode =='var_hinge':
                    z_view = z.clone().detach()
                    z_view = z_view.view(z_view.size(0), -1)
                    #print('VIEW z',z_view.shape)
                    z_mean = self.latent_mu(z_view)
                    z_logvar = self.latent_std(z_view)
                    std = torch.exp(0.5 * z_logvar)
                    eps = std.data.new(std.size()).normal_()
                    z_sampled_compressed =  eps.mul(std).add(z_mean)
                    z_sampled = self.latent_decoder_vae(z_sampled_compressed)
                    #z_sampled = self.relu_vae(z_sampled)
                    z_sampled = z_sampled.view(-1, self.ch_out, self.scaled_patch, self.scaled_patch)
                    img_rec = self.decoder_vae(z_sampled)
                    ld = 0.2
                    next_layer_input = z.clone().detach() + z_sampled.clone().detach()  # full module output # This sets requires_grad to False
                    reshaped_origs,_,_ = reshape_to_patches(orig_imgs,self.patch_size,self.overlap)
                    #print('ORIGS',torch.min(reshaped_origs).item(),torch.max(reshaped_origs).item())
                    #print('Reconstructed',torch.min(img_rec).item(),torch.max(img_rec).item())

                    recons_loss = F.mse_loss(img_rec, reshaped_origs)
                    ii = 1 + z_logvar - z_mean ** 2 - z_logvar.exp()
                    kld_loss = torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mean ** 2 - z_logvar.exp() , dim=1), dim=0) #We changed the sum to mean to avoid Nan Values. This has no bearings on the KL divergence loss
                    #kld_loss /= x.view(-1, input_size).data.shape[0] * input_size
                    #print(kld_loss.item(),torch.max(-0.5 * torch.sum(ii,dim=1)).item(),torch.min(ii).item(),torch.max(z_mean).item(),torch.max(z_logvar))

                    loss_vae = recons_loss + kld_loss
                    loss_vae.backward(retain_graph=False)

                    self.optimizer_latent_std.step()
                    self.optimizer_latent_mu.step()
                    self.optimizer_latent_decoder_vae.step()
                    self.optimizer_latent_decoder_vae.step()

                    self.optimizer_latent_std.zero_grad()
                    self.optimizer_latent_mu.zero_grad()
                    self.optimizer_latent_decoder_vae.zero_grad()
                    self.optimizer_latent_decoder_vae.zero_grad()
                    return {'next_layer_input':next_layer_input,'rep':h,'loss':loss,'n_patches_y':n_patches_y,'n_patches_x':n_patches_x,'rec_loss':recons_loss,'kld_loss':kld_loss,'loss_vae':loss_vae}

            else:
                next_layer_input = z.clone().detach()  # full module output # This sets requires_grad to False

                return {'next_layer_input':next_layer_input,'rep':h,'loss':loss,'n_patches_y':n_patches_y,'n_patches_x':n_patches_x}
            #next_layer_input, h, loss, n_patches_y, n_patches_x

        elif self.opt.train_mode =='dorsal':
            self.labels = labels
            self.optimizer.zero_grad()
            output_dic = self.forward(
                x, n_patches_y, n_patches_x
            )
            z, h, n_patches_y, n_patches_x = output_dic['next_layer_input'], output_dic['rep'], output_dic[
                'n_patches_y'], output_dic['n_patches_x']
            # out: mean pooled per patch for each z. For example z shape is (198,128,16,16)  the mean pooling is (4,128,7,7)  where 198=4*7*7 = batch_size*img_patch_size*img_patch_size

            # the outs we are using are the mean pooled ones  so (4,nb_filters,7,7).

            #print("Shape h",h.shape,h.reshape(self.opt.batch_size,-1).shape)
            out = self.projection_layer(h.reshape(z.shape[0],-1))
            out = self.relu(out)
            out = self.classification_layer(out)


            loss = self.loss_criterion(out,self.labels.to(torch.int64))

            #loss = self.loss(torch.squeeze(h,dim=3).permute(0,2,1),self.labels)  # z, detach(c)

            # print('Loss calculation done...',loss)

            # print('updating weights...')

            next_layer_input = z.clone().detach()  # full module output # This sets requires_grad to False

            if False:
                z_view = z.view(z.size(0), -1)
                z_mean = self.latent_mu(z_view)
                z_logvar = self.latent_std(z_view)
                std = torch.exp(0.5 * z_logvar)
                eps = std.data.new(std.size()).normal_()
                z_sampled_compressed = eps.mul(std).add(z_mean)
                z_sampled = self.latent_decoder_vae(z_sampled_compressed)
                z_sampled = z_sampled.view(-1, self.ch_out, self.scaled_patch, self.scaled_patch)
                img_rec = self.decoder_vae(z_sampled)
                ld = 0.2
                #print("z sampled ",z_sampled.shape)
                #reshaped_origs, _, _ = reshape_to_patches(orig_imgs, self.patch_size, self.overlap)
                # print('ORIGS',torch.min(reshaped_origs).item(),torch.max(reshaped_origs).item())
                # print('Reconstructed',torch.min(img_rec).item(),torch.max(img_rec).item())
                recons_loss = F.mse_loss(img_rec, orig_imgs)
                ii = 1 + z_logvar - z_mean ** 2 - z_logvar.exp()
                kld_loss = torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mean ** 2 - z_logvar.exp(), dim=1),
                                      dim=0)  # We changed the sum to mean to avoid Nan Values. This has no bearings on the KL divergence loss
                # kld_loss /= x.view(-1, input_size).data.shape[0] * input_size
                loss_vae = recons_loss + kld_loss
                total_loss = loss_vae + loss
                total_loss.backward(retain_graph=False)
                self.optimizer.step()

                #self.optimizer_latent_std.step()
                #self.optimizer_latent_mu.step()
                #self.optimizer_latent_decoder_vae.step()
                #self.optimizer_latent_decoder_vae.step()

                #self.optimizer_latent_std.zero_grad()
                #self.optimizer_latent_mu.zero_grad()
                #self.optimizer_latent_decoder_vae.zero_grad()
                #self.optimizer_latent_decoder_vae.zero_grad()
                return {'next_layer_input': next_layer_input, 'rep': h, 'loss': loss, 'n_patches_y': n_patches_y,
                        'n_patches_x': n_patches_x, 'rec_loss': recons_loss, 'kld_loss': kld_loss, 'loss_vae': loss_vae,'labels':self.labels}
            else:
                loss.backward()
                self.optimizer.step()

                next_layer_input = z.clone().detach()  # full module output # This sets requires_grad to False

                return {'next_layer_input': next_layer_input, 'rep': h, 'loss': loss, 'n_patches_y': n_patches_y,
                    'n_patches_x': n_patches_x,'labels':self.labels}

        elif self.opt.train_mode=='predSim':
            # Calculate local loss and update weights
            if (not self.opt.no_print_stats or self.training):

                output_dic= self.forward(x)
                h_return, z= output_dic['next_layer_input'], output_dic['rep']
                loss = self.evaluate_loss(z,y=y,y_onehot=y_onehot)['loss']

                # Single-step back-propagation
                loss.backward(retain_graph=False)

                # Update weights in this layer and detatch computational graph
                self.optimizer.step()
                self.optimizer_decoder_y.step()
                self.optimizer_conv_loss.step()
                self.optimizer.zero_grad()
                self.optimizer_decoder_y.zero_grad()
                self.optimizer_conv_loss.zero_grad()

                h_return.detach_()

                loss = loss.item()
            else:
                loss = 0.0
            return {'next_layer_input': h_return, 'rep': z, 'loss': loss, 'n_patches_y': n_patches_y,
             'n_patches_x': n_patches_x}




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch local error training')
    parser.add_argument('--model', default='vgg8b',
                        help='model, mlp, vgg13, vgg16, vgg19, vgg8b, vgg11b, resnet18, resnet34, wresnet28-10 and more (default: vgg8b)')
    parser.add_argument('--train_mode', default='hinge',
                        help='ll')
    parser.add_argument('--no_print_stats', default=False,
                        help='ll')
    parser.add_argument('--asymmetric_W_pred', default=True,
                        help='ll')
    parser.add_argument('--no_batch_norm', default=False,
                        help='ll')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout after each nonlinearity (default: 0.0)')
    parser.add_argument('--nonlin', default='relu',
                        help='nonlinearity, relu or leakyrelu (default: relu)')
    parser.add_argument('--optim', default='adam',
                        help='nonlinearity, relu or leakyrelu (default: relu)')
    parser.add_argument('--beta', default=0.1,
                        help='ll')
    parser.add_argument("--prediction_step", default=5,
        help="(Number of) Time steps to predict into future")
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='initial learning rate (default: 5e-4)')
    parser.add_argument(
        "--weight_decay",
        default=0.,
        help="weight decay or l2-penalty on weights (for ADAM optimiser, default = 0., i.e. no l2-penalty)"
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="",
    )
    ############ Next parameters are specific for contrastive loss
    parser.add_argument('--negative_samples', default=1,
                        help='ll')
    parser.add_argument('--contrast_mode', default="hinge",
                        help='ll')

    parser.add_argument('--gating_av_over_preds', default=False,
                        help='Boolean: average feedback gating (--feedback_gating) from higher layers over different prediction steps (k)')
    parser.add_argument('--detach_c', default=False,
                        help='"Boolean whether the gradient of the context c should be dropped (detached)')
    parser.add_argument('--current_rep_as_negative', default=False,
                        help='#Use the current feature vector (context at time t as opposed to predicted time step t+k) itself as/for sampling the negative sample')
    parser.add_argument('--sample_negs_locally', default=True,
                        help='Sample neg. samples from batch but within same location in image, i.e. no shuffling across locations')
    parser.add_argument('--sample_negs_locally_same_everywhere', default=True,
                        help='Extension of --sample_negs_locally_same_everywhere (must be True). No shuffling across locations and same sample (from batch) for all locations. I.e. negative sample is simply a new input without any scrambling')

    parser.add_argument('--either_pos_or_neg_update', default=False,
                        help='Randomly chose to do either pos or neg update in Hinge loss. --negative_samples should be 1. Only used with --current_rep_as_negative True')
    parser.add_argument(
        "--freeze_W_pred",
        default=False,
        help="Boolean whether the k prediction weights W_pred (W_k in ContrastiveLoss) are frozen (require_grad=False).",
    )
    parser.add_argument(
        "--unfreeze_last_W_pred",
        default=False,
        help="Boolean whether the k prediction weights W_pred of the last module should be unfrozen.",
    )

    parser.add_argument(
        "--weight_init",
        default=False,
        help="Boolean to decide whether to use special weight initialization (delta orthogonal)",
    )
    parser.add_argument(
        "--no_pred",
        default=False,
        help="Boolean whether Wpred * c is set to 1 (no prediction). i.e. fourth factor omitted in learning rule",
    )
    parser.add_argument(
        "--no_gamma",
        action="store_true",
        default=False,
        help="Boolean whether gamma (factor which sets the opposite sign of the update for pos and neg samples) is set to 1. i.e. third factor omitted in learning rule",
    )

    args = parser.parse_args()


    train_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomCrop(64, padding=2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


    local_conv = LocalLossBlockConv(ch_in=1, ch_out=64, kernel_size=3, stride=1, padding=1, num_classes=10, dim_out=64,opt=args,block_idx=0,
                 dropout=None, bias=None, pre_act=False, post_act=True,patch_size=16,   overlap_factor=2)

    dataset_train = datasets.MNIST('../data/MNIST', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=4, shuffle=False)

    for img_idx, (input_tensor,labels) in enumerate(train_loader):
        label_one_hot = to_one_hot(labels)
        next_layer_input, h, loss, n_patches_y, n_patches_x = local_conv.train_step(input_tensor,y=labels,y_onehot=label_one_hot)
        print("LOSS",h.shape,next_layer_input.shape)
        #print(h_return.shape,h.shape)