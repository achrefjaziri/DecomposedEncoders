import torch
import torch.nn as nn
import os,sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from models.local_blocks import LocalLossBlockConv
from models.CPC_model import CPC
from collections import OrderedDict
import torch.nn.functional as F
from torch.autograd import Variable
from utils.model_utils import init_model
from utils.logging_utils import CNNLayerVisualization
from torch.nn.parallel import DistributedDataParallel as DDP
from models.VGG8 import VGG_8

import argparse, yaml

class ClassificationModel(torch.nn.Module):
    def __init__(self, in_channels=256, num_classes=200, hidden_nodes=512,linear_classifier=True, p=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.avg_pool = nn.AvgPool2d((7, 7),stride=2,  padding=0) #,
        self.model = nn.Sequential()


        if not linear_classifier:
            self.model.add_module(
                "layer1", nn.Linear(self.in_channels, hidden_nodes, bias=True)
            )

            self.model.add_module("ReLU", nn.ReLU())
            self.model.add_module("Dropout", nn.Dropout(p=p))

            self.model.add_module(
                "layer 2", nn.Linear(hidden_nodes, num_classes, bias=True)
            )
        else:
            self.model.add_module(
                "layer1", nn.Linear(self.in_channels, hidden_nodes, bias=True)
            )
            self.model.add_module("Dropout", nn.Dropout(p=p))

            self.model.add_module(
                "layer2", nn.Linear(hidden_nodes, num_classes, bias=True)
            )

    def forward(self, x, *args):
        #x = self.avg_pool(x).squeeze()
        x = self.model(x)#.squeeze()
        return x


cfg = {
    'vgg6': [128, 256, 'M', 256, 512, 'M', 1024, 'M', 1024, 'M'],
    'vgg6a': [128, 'M', 256, 'M', 512, 'M', 512],
    'vgg6b': [128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'vgg8': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'vgg8a': [128, 256, 'M', 256, 512, 'M', 512, 'M', 512],
    'vgg8b': [128, 256, 'M', 256, 512, 'M', 512, 'M', 512, 'M'],
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg11b': [128, 128, 128, 256, 'M', 256, 512, 'M', 512, 512, 'M', 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def distribute_over_GPUs(opt,model,rank):
    model = model.to(rank)
    if opt.train_mode=='CPC':
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    else:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)


    # model = nn.parallel.DataParallel(model, device_ids=list(range(num_GPU)))
    """
    An easy way to find unused params is train your model on a single node without the DDP wrapper. after loss.backward() and before optimizer.step() call add the below lines

    for name, param in model.named_parameters():
        if param.grad is None:
            print(name)
    
    This will print any param which did not get used in loss calculation, their grad will be None.
    """

    return model

class FullModel(torch.nn.Module):

    '''
        VGG and VGG-like networks.
        The network can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.

        Args:
            vgg_name (str): The name of the network.
            input_dim (int): Feature map height/width for input.
            input_ch (int): Number of feature maps for input.
            num_classes (int): Number of classes (used in local prediction loss).
            feat_mult (float): Multiply number of feature maps with this number.

        We are focusing only on Self supervised Models
    possible losses include
    -Hinge Loss
    -
    -Multitarget Contrastive
    Our different models are:
    1- BP with a simple VGG or ResNet like Architecture (supervised and CLapp)
    2- Pred Sim with Local BP (supervised training)
    3- Pred sim with CLAPP like loss (unsupervised training)
    4-VGG trained with CLAPP Layer wise Training
    5-  Hebbian Fully unsupervised training
    6-CLAPP/Hebbian + Local VAE
    7- CLAPP/Hebbian + Local VAE with  FA
    8- Double model training

        '''
    def __init__(self, opt, rank=None):
        super().__init__()
        self.opt = opt
        self.device = self.opt.device

        self.cfg = cfg[opt.vgg_name]
        self.input_dim = int(opt.input_dim)
        self.input_ch = int(opt.input_ch)
        self.num_classes = int(opt.num_classes)
        self.encoder, self.output_dim = self._make_layers(self.input_dim, self.input_ch)
        print(self.encoder)

        if self.opt.backprop: #BP Mode

            if self.opt.train_mode == 'CPC':
                # Training model with hinge loss CPC. Paper:https://arxiv.org/abs/2010.08262
                self.encoder = CPC(self.encoder,ch_in=self.opt.input_ch, ch_out=self.cfg[-2],opt=self.opt,block_idx=0)
            elif self.opt.train_mode == 'contrastive':
                #SimCLR model
                self.criterion = nn.CrossEntropyLoss()  # DiceLoss(weight =class_weights)

            elif self.opt.train_mode == 'CE':
                # Training with cross entropy loss (supervised Training)
                self.decoder =ClassificationModel(in_channels=self.output_dim*self.output_dim*self.cfg[-2],num_classes=self.num_classes,linear_classifier=True)
                self.criterion = nn.CrossEntropyLoss()  # DiceLoss(weight =class_weights)
            elif self.opt.train_mode=='mvment':
                self.encoder = VGG_8(self.input_ch,True) #self.output_dim * self.output_dim *
                print('2* self.cfg[-2]',2* self.cfg[-2])
                self.decoder = ClassificationModel(in_channels=2048,
                                                   num_classes=self.num_classes, linear_classifier=True)
                self.criterion = nn.CrossEntropyLoss()  # DiceLoss(weight =class_weights)

            elif self.opt.train_mode=='so_mvment':
                self.encoder = VGG_8(self.input_ch, True)  # self.output_dim * self.output_dim *
                print('2* self.cfg[-2]', 2 * self.cfg[-2])
                self.decoder_motion_type = ClassificationModel(in_channels=2048,
                                                   num_classes=2, linear_classifier=True)

                self.decoder_velocity = ClassificationModel(in_channels=2048,
                                                               num_classes=12, linear_classifier=True)

                self.criterion = nn.CrossEntropyLoss()  # DiceLoss(weight =class_weights)

            else:
                    raise NotImplementedError(f'{self.opt.train_mode} is not implemented.')

            if self.opt.device!="cpu":  # and not configs['RESUME']
                if rank!=None: #If the rank is specified we use distributed Dataparallel and multiple GPUs
                    self.encoder = distribute_over_GPUs(opt,self.encoder,rank=rank)
                    #self.encoder._set_static_graph()
                    if self.opt.train_mode=='CE' or self.opt.train_mode=='mvment':
                        self.decoder = distribute_over_GPUs(opt, self.decoder, rank=rank)
                    elif self.opt.train_mode == 'so_mvment':
                        self.decoder_motion_type = distribute_over_GPUs(opt, self.decoder_motion_type, rank=rank)
                        self.decoder_velocity = distribute_over_GPUs(opt, self.decoder_velocity, rank=rank)



                else: #we use only a single gpu
                    self.encoder = torch.nn.DataParallel(self.encoder,
                                                         device_ids=list(range(torch.cuda.device_count())))
                    if self.opt.train_mode=='CE' or self.opt.train_mode=='mvment':
                        self.decoder = torch.nn.DataParallel(self.decoder,
                                                         device_ids=list(range(torch.cuda.device_count())))
                    elif self.opt.train_mode == 'so_mvment':
                        self.decoder_motion_type = torch.nn.DataParallel(self.decoder_motion_type,
                                                             device_ids=list(range(torch.cuda.device_count())))
                        self.decoder_velocity = torch.nn.DataParallel(self.decoder_velocity,
                                                             device_ids=list(range(torch.cuda.device_count())))

            if self.opt.weight_init:
                init_model(self.encoder, self.configs.init_model)
                if self.opt.train_mode == 'so_mvment':
                    init_model(self.decoder_motion_type, self.configs.init_model)
                    init_model(self.decoder_velocity, self.configs.init_model)

                else:
                    init_model(self.decoder, self.configs.init_model)

            if not self.opt.test_mode:
                # Optimizer
                self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters(), lr=self.opt.lr,
                                                          weight_decay=self.opt.weight_decay)

                # gamma = decaying factor
                self.scheduler_encoder = StepLR(self.optimizer_encoder, step_size=2,
                                                gamma=0.96)  # TODO check for the best scheduler

            if self.opt.train_mode == 'CE' or self.opt.train_mode == 'mvment':

                    self.optimizer_decoder = torch.optim.Adam(self.decoder.parameters(), lr=self.opt.lr,
                                                          weight_decay=self.opt.weight_decay)
                    self.scheduler_decoder = StepLR(self.optimizer_decoder, step_size=2,
                                                    gamma=0.96)  # TODO check for the best scheduler
            elif self.opt.train_mode=='so_mvment':
                self.optimizer_decoder_motion_type = torch.optim.Adam(self.decoder_motion_type.parameters(), lr=self.opt.lr,
                                                          weight_decay=self.opt.weight_decay)
                self.optimizer_decoder_velocity = torch.optim.Adam(self.decoder_velocity.parameters(), lr=self.opt.lr,
                                                          weight_decay=self.opt.weight_decay)

                self.scheduler_decoder = StepLR(self.optimizer_decoder_motion_type, step_size=2,
                                                gamma=0.96)  # TODO check for the best scheduler

                self.scheduler_decoder = StepLR(self.optimizer_decoder_velocity, step_size=2,
                                                gamma=0.96)  # TODO check for the best scheduler
        else:  # We don't  Backpropagate here
            if self.opt.train_mode=='hinge' or self.opt.train_mode=='var_hinge' or self.opt.train_mode=='dorsal':
                #Training each layer seperatly according to CLAPP loss. Paper:https://arxiv.org/abs/2010.08262

                pass
            elif self.opt.train_mode=='contrastive':
                pass
            elif self.opt.train_mode=='predSim':
                #Training with local Error Single and PredSim loss. Paper:https://arxiv.org/pdf/1901.06656.pdf
                pass
            else:
                raise NotImplementedError(f'{self.opt.train_mode} is not implemented.')

            if self.opt.device!="cpu" and torch.cuda.is_available():  # and not configs['RESUME']
                if rank!=None: #If the rank is specified we use distributed Dataparallel and multiple GPUs
                    self.encoder = distribute_over_GPUs(opt,self.encoder,rank=rank)
                else: #we use only a single gpu
                    self.encoder = torch.nn.DataParallel(self.encoder,
                                                         device_ids=list(range(torch.cuda.device_count())))
        self.img_output = None

    def _make_layers(self, input_dim,input_ch,mirror_model=False):
        """
        mirror_model: allows us to create a sequential model with normal conv layers regardless of the current train_mode
        This is needed for the visualization of the filters and the activations (see: the function filter_visualization )
        """
        layers = []
        scale_cum = 1
        idx_vgg =0
        for idx,x in enumerate(self.cfg):
            if x == 'M':
                if self.opt.train_mode=='dorsal':
                    layers += [nn.MaxPool3d(kernel_size=[1,2,2], stride=[1,2,2])]
                else:

                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                scale_cum *= 2
            elif x == 'M3':
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
                scale_cum *= 2
            elif x == 'M4':
                layers += [nn.MaxPool2d(kernel_size=4, stride=4)]
                scale_cum *= 4
            elif x == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
                scale_cum *= 2
            elif x == 'A3':
                layers += [nn.AvgPool2d(kernel_size=3, stride=2, padding=1)]
                scale_cum *= 2
            elif x == 'A4':
                layers += [nn.AvgPool2d(kernel_size=4, stride=4)]
                scale_cum *= 4
            else:
                #x = int(x * feat_mult)
                if idx==0 and int(input_dim) > 64:
                    scale_cum = 2
                    if self.opt.backprop or mirror_model:
                        if not self.opt.no_batch_norm:
                            bn = torch.nn.BatchNorm2d(x)
                            nn.init.constant_(bn.weight, 1)
                            nn.init.constant_(bn.bias, 0)
                            layers += [
                                nn.Conv2d(in_channels=input_ch, out_channels=x, kernel_size=3, stride=1, padding=1), bn,
                                nn.ReLU()]
                        else:
                            layers += [
                                nn.Conv2d(in_channels=input_ch, out_channels=x, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()]
                    else:
                        layers += [LocalLossBlockConv(ch_in=self.input_ch, ch_out=x, kernel_size=3, stride=1, padding=1, num_classes=self.num_classes, dim_out=input_dim // scale_cum,opt=self.opt,block_idx=idx_vgg,cfg=self.cfg)]
                else:
                    if self.opt.backprop or mirror_model:
                        if not self.opt.no_batch_norm:
                            bn = torch.nn.BatchNorm2d(x)
                            nn.init.constant_(bn.weight, 1)
                            nn.init.constant_(bn.bias, 0)
                            layers += [
                                nn.Conv2d(in_channels=input_ch, out_channels=x, kernel_size=3, stride=1, padding=1),bn,nn.ReLU()]
                        else:
                            layers += [
                                nn.Conv2d(in_channels=input_ch, out_channels=x, kernel_size=3, stride=1, padding=1),nn.ReLU()]

                    else:
                        layers += [LocalLossBlockConv(ch_in=input_ch, ch_out=x, kernel_size=3, stride=1, padding=1, num_classes=self.num_classes, dim_out=input_dim // scale_cum,opt=self.opt,block_idx=idx_vgg,cfg=self.cfg)]

                input_ch = x
                idx_vgg += 1

        self.model_splits = 1 if self.opt.backprop else idx_vgg

        return nn.Sequential(*layers), input_dim // scale_cum

    def get_classifier_input_dim(self):
        """returns the shape of input tensors for the classification model"""
        return self.output_dim * self.output_dim * self.cfg[-2]
    def get_model_splits(self):
        '''
        Returns the number of seperate modules trained locally
        '''
        return self.model_splits
    def forward(self,img_input):
        self.img_input =img_input
        if self.opt.backprop:
            if self.opt.train_mode =="CE" or self.opt.train_mode == 'mvment': # Using cross entropy as a loss means that we also have a decoder
                self.img_encoding = self.encoder(self.img_input)
                img_encoding_flattened= self.img_encoding.view(self.img_encoding.size(0), -1)
                self.img_output = self.decoder(img_encoding_flattened)
                output_dic = {'img_encoding': self.img_encoding, 'class_pred':self.img_output}
                #change it in the case of self mvment

                return output_dic
            elif self.opt.train_mode=='so_mvment':
                self.img_encoding = self.encoder(self.img_input)
                output_dic = {'img_encoding': self.img_encoding}
                # change it in the case of self mvment
                return output_dic

            elif self.opt.train_mode=='CPC':
                output_dic = self.encoder(self.img_input)
                self.img_encoding = output_dic['rep']

                next_layer_input, n_patches_y, n_patches_x = output_dic['next_layer_input'], output_dic['n_patches_y'], \
                                                             output_dic['n_patches_x']

                output_dic = {'img_encoding': self.img_encoding}

                return output_dic


            else: #in the case of self supervised training
                self.img_encoding = self.encoder(self.img_input)
                output_dic = {'img_encoding': self.img_encoding}

                return output_dic
        else: # No backprop training means our model is saved as list of layers and we have to iterate over them to get the image encoding
            next_layer_input =self.img_input
            n_patches_x, n_patches_y = None, None

            for idx, module in enumerate(self.encoder.module):
                if isinstance(self.cfg[idx],str):
                    next_layer_input = module(next_layer_input)
                else:
                    #Next layer input contains the z and the 'rep' contains
                    output_dic = module(next_layer_input, n_patches_y=n_patches_y,
                                                   n_patches_x=n_patches_x)
                    next_layer_input, n_patches_y, n_patches_x = output_dic['next_layer_input'], output_dic['n_patches_y'], output_dic['n_patches_x']


            self.img_encoding = output_dic['rep']
            output_dic = {'img_encoding':output_dic['rep'],'z':next_layer_input}
            return output_dic
    def get_outputs(self):
        return self.img_output

    def training_step(self,x,y,y_onehot=None,motion_type=None):
        self.img_input =x
        self.gt = y
        if self.opt.backprop:  # BP Mode
            if self.opt.train_mode == 'CPC':
                acts = []
                outs = []
                loss_values = []
                # Training each layer seperatly according to CLAPP loss. Paper:https://arxiv.org/abs/2010.08262
                #self.encoder._set_static_graph()
                output_dic = self.encoder.module.train_step(self.img_input)
                next_layer_input, h, loss, n_patches_y, n_patches_x = output_dic['next_layer_input'], output_dic['rep'], \
                                                                      output_dic['loss'], output_dic['n_patches_y'], \
                                                                      output_dic['n_patches_x']


                # h is the output after pooling so we have (4,128,7,7) and z is the encoding before pooling (198,128,16,16)  for the first layer. PS 198=4*7*7

                loss_values.append(loss)
                return {'main_loss': loss_values}

            elif self.opt.train_mode == 'contrastive':
                pass # DiceLoss(weight =class_weights)
            elif self.opt.train_mode == 'CE' or self.opt.train_mode == 'mvment':

                self.img_encoding = self.encoder(self.img_input)

                img_encoding_flattened = self.img_encoding.view(self.img_encoding.size(0), -1)
                self.img_output = self.decoder(img_encoding_flattened)
                loss = self.criterion(self.img_output, self.gt)

                # only backpropagate a loss that is nont nan
                loss.backward()
                self.optimizer_encoder.step()
                self.optimizer_encoder.zero_grad()

                self.optimizer_decoder.step()
                self.optimizer_decoder.zero_grad()
                return {'main_loss': [loss]}
            elif self.opt.train_mode=='so_mvment':
                self.img_encoding = self.encoder(self.img_input)

                img_encoding_flattened = self.img_encoding.view(self.img_encoding.size(0), -1)

                predicted_motion_type = self.decoder_motion_type(img_encoding_flattened)

                if motion_type==0:
                    motion_type_gt = torch.zeros((self.img_input.shape[0])).to(img_encoding_flattened.device).type(torch.uint8)
                else:
                    motion_type_gt = torch.ones((self.img_input.shape[0])).to(img_encoding_flattened.device).type(torch.uint8)
                loss_motion_type = self.criterion(predicted_motion_type, motion_type_gt)

                #this batch contains self motion examples
                predicted_velocity = self.decoder_velocity(img_encoding_flattened)

                loss_velocity = self.criterion(predicted_velocity, self.gt)



                loss = loss_motion_type +loss_velocity
                loss.backward()
                self.optimizer_encoder.step()
                self.optimizer_encoder.zero_grad()


                self.optimizer_decoder_motion_type.step()
                self.optimizer_decoder_motion_type.zero_grad()


                self.optimizer_decoder_velocity.step()
                self.optimizer_decoder_velocity.zero_grad()
                """
                                if motion_type==0:
                    self.optimizer_decoder_selfmotion_velocity.step()
                    self.optimizer_decoder_selfmotion_velocity.zero_grad()
                else:
                    self.optimizer_decoder_selfmotion_velocity.step()
                    self.optimizer_decoder_selfmotion_velocity.zero_grad()

                """

                return {'main_loss': [loss],'motion_type_loss':loss_motion_type,'velocity_loss':[loss_velocity]}

        else:
            #Training here depends on the encoder we are using

            next_layer_input = x  # Tensor of shape (batch,channels,img_width,img_height)
            acts = []
            outs =[]
            loss_values =[]
            n_patches_x, n_patches_y = None, None
            labels = None #labels created for the dorsal loss task
            loss_values_rec = []
            loss_values_kld = []
            loss_values_vae = []

            # forward loop through modules
            for idx, module in enumerate(self.encoder.module):
                # block gradient of h at some point -> should be blocked after one module since input was detached
                if isinstance(self.cfg[idx],str):
                    next_layer_input = module(next_layer_input)

                else:
                    output_dic = module.train_step(next_layer_input,y, y_onehot, n_patches_y=n_patches_y, n_patches_x=n_patches_x,orig_imgs=x,labels=labels)
                    next_layer_input, h, loss, n_patches_y, n_patches_x = output_dic['next_layer_input'],output_dic['rep'],output_dic['loss'],output_dic['n_patches_y'],output_dic['n_patches_x']

                    if self.opt.train_mode =='var_hinge' or self.opt.train_mode =='dorsal' : #
                        loss_values_rec.append(output_dic['rec_loss'])
                        loss_values_kld.append(output_dic['kld_loss'])
                        loss_values_vae.append(output_dic['loss_vae'])
                    if self.opt.train_mode =='dorsal':
                        labels = output_dic['labels'] # labels are created in encoder_num = 0 and then passed through the modules

                    # h is the output after pooling so we have (4,128,7,7) and z is the encoding before pooling (198,128,16,16)  for the first layer. PS 198=4*7*7

                    acts.append(next_layer_input)  # needed for optional recurrence
                    outs.append(h)  # out: mean pooled per patch for each z. For example z shape is (198,128,16,16)  the mean pooling is (4,128,7,7)  where 198=4*7*7 = batch_size*img_patch_size*img_patch_size
                    loss_values.append(loss)
                # We are basically getting a representation for each img patch independently.


            return {'main_loss':loss_values,'vae_loss':loss_values_vae,'kld_loss':loss_values_kld,'rec_loss':loss_values_rec}
    def eval_mode(self):
        self.encoder.eval()
        if self.opt.train_mode=="CE":
            self.decoder.eval()
    def get_lr(self):
        return self.scheduler_encoder.get_last_lr()
    def update_lr(self):
        self.scheduler_encoder.step()
        if self.opt.train_mode=="CE":
            self.scheduler_decoder.step()
    def save_model(self, current_saving_dir, epoch):
        #TODO save a simpler vgg model  also
        if not os.path.exists(current_saving_dir):
            os.makedirs(current_saving_dir)
        idx_convs =0
        if self.opt.train_mode=='CPC' or self.opt.train_mode == 'mvment' or self.opt.train_mode == 'CE' or self.opt.train_mode == 'so_mvment':
            torch.save(self.encoder.state_dict(),
                       os.path.join(
                           current_saving_dir,
                           "model_{}_{}.ckpt".format(0, epoch)))
        else:

            for idx, layer in enumerate(self.encoder.module):
                if True:#not isinstance(self.cfg[idx], str)
                    torch.save(layer.state_dict(),
                        os.path.join(
                            current_saving_dir,
                            "model_{}_{}.ckpt".format(idx_convs, epoch)))  # alternative: "/encoder_epoch_{0}.pth.tar".format(epoch)
                    idx_convs+=1
    def save_only_main(self,current_saving_dir, epoch):
        '''
        Saves main branch without local training modules
        '''
        if self.opt.backprop:
            if not os.path.exists(current_saving_dir):
                os.makedirs(current_saving_dir)
            torch.save(self.encoder.state_dict(),
                       os.path.join(
                           current_saving_dir,
                           "main_model_{}.ckpt".format(epoch)))
        else:
            mirror_model,_ = self._make_layers(self.input_dim, self.input_ch,mirror_model=True)
            idx_mirror_model = 0
            #we assign the weight of the main branch of the trained model to the mirror_model so we can perform layer visualization
            for idx, layer in enumerate(self.encoder.module):
                if not isinstance(self.cfg[idx], str):
                    if self.opt.no_batch_norm:
                        mirror_model[idx_mirror_model].weight.data = layer.main_branch[0].weight.data
                        mirror_model[idx_mirror_model].bias.data = layer.main_branch[0].bias.data
                        idx_mirror_model +=2
                    else:
                        mirror_model[idx_mirror_model].weight.data = layer.main_branch[0].weight.data
                        mirror_model[idx_mirror_model].bias.data = layer.main_branch[0].bias.data
                        # Assign running mean and running var to Batch norm layer
                        mirror_model[idx_mirror_model+1].running_mean.data = layer.main_branch[1].running_mean.data
                        mirror_model[idx_mirror_model+1].running_var.data = layer.main_branch[1].running_var.data
                        idx_mirror_model +=3
                else:
                    #skip the Max pooling layer
                    idx_mirror_model += 1
            if not os.path.exists(current_saving_dir):
                os.makedirs(current_saving_dir)
            torch.save(mirror_model.state_dict(),
                       os.path.join(
                           current_saving_dir,
                           "main_model_{}.ckpt".format(epoch)))
    def load_models(self,epoch):
        # get paths from configs
        # load models + optimizers
        if self.opt.train_mode=='CPC' or self.opt.train_mode == 'CE' or self.opt.train_mode == 'mvment' or self.opt.train_mode=='so_mvment':
            self.encoder.load_state_dict(
                torch.load(
                    os.path.join(
                        self.opt.model_path,
                        "model_{}_{}.ckpt".format(0, epoch),
                    )
                ,map_location='cuda')
            )
            print("=> loaded Encoder checkpoint '{}'"
                  .format(self.opt.model_path))
        else:
            idx_convs =0
            for idx, layer in enumerate(self.encoder.module):
                if True: #not isinstance(self.cfg[idx], str)
                    print("loading",idx)
                    layer.load_state_dict(
                        torch.load(
                            os.path.join(
                                self.opt.model_path,
                                "model_{}_{}.ckpt".format(idx, epoch),
                            )
                        ,map_location='cuda:0')
                    )

                    idx_convs+=1


            print("=> loaded Encoder checkpoint '{}'"
                  .format(self.opt.model_path))
    def load_models_incomplete(self,epoch,single_load):
        """
        This function can be used to laad the weights of the conv layers from CLAPP project
        """
        # get paths from configs
        # load models + optimizers
        idx_convs =0
        if single_load:
            current_block = torch.load(
                os.path.join(
                    self.opt.model_path,
                    "model_{}_{}.ckpt".format(0, epoch),
                )
            )
            print(current_block['model.0.weight'].shape) #0,2,5,7,10,13

        else:
            for idx, layer in enumerate(self.encoder):
                if not isinstance(self.cfg[idx], str):

                    current_block =torch.load(
                            os.path.join(
                                self.opt.model_path,
                                "model_{}_{}.ckpt".format(idx_convs, epoch),
                            )
                        )
                    print("==========================")
                    print(current_block['model.0.weight'].shape)
                    print(layer.main_branch[0].weight.data.shape)
                    #print(current_vgg_block)
                    #print(layer.main_branch.0.data)

                    idx_convs+=1
        print("=> loaded Encoder checkpoint '{}'"
                  .format(self.opt.model_path))
    def filter_visualization(self,writer,epoch,gpu,filter_nb=6):
        #create a main branch model to use with vis class
        if self.opt.backprop:
            filters = []
            conv_indices =[]
            mirror_model,_ = self._make_layers(self.input_dim, self.input_ch,mirror_model=True)
            if self.opt.train_mode=='CE' or self.opt.train_mode == 'mvment':
                for idx,layer in enumerate(self.encoder.module):
                    if isinstance(layer,nn.Conv2d):
                        mirror_model[idx].weight.data = self.encoder.module[idx].weight.data
                        mirror_model[idx].bias.data = self.encoder.module[idx].bias.data
                        conv_indices.append(idx)
                    if isinstance(layer,nn.BatchNorm2d):
                        mirror_model[idx].running_mean.data = self.encoder.module[idx].running_mean.data
                        mirror_model[idx].running_var.data = self.encoder.module[idx].running_var.data
            else:
                for idx,layer in enumerate(self.encoder.module.main_branch):
                    if isinstance(layer,nn.Conv2d):
                        mirror_model[idx].weight.data = self.encoder.module.main_branch[idx].weight.data
                        mirror_model[idx].bias.data = self.encoder.module.main_branch[idx].bias.data
                        conv_indices.append(idx)
                    if isinstance(layer,nn.BatchNorm2d):
                        mirror_model[idx].running_mean.data = self.encoder.module.main_branch[idx].running_mean.data
                        mirror_model[idx].running_var.data = self.encoder.module.main_branch[idx].running_var.data
            for conv_idx in conv_indices:
                layer_vis = CNNLayerVisualization(mirror_model, conv_idx, filter_nb)
                filter = layer_vis.visualise_layer_with_hooks(input_channels=self.opt.input_ch)
                filters.append(filter)

            filters = np.asarray(filters)
            writer.add_images(f'Learned Filters', filters,epoch, dataformats='NHWC')
        else:
            mirror_model,_ = self._make_layers(self.input_dim, self.input_ch,mirror_model=True)
            idx_mirror_model = 0
            conv_indices = [] #use them to access the cnn layers quickly during visualization
            #we assign the weight of the main branch of the trained model to the mirror_model so we can perform layer visualization
            for idx, layer in enumerate(self.encoder.module):
                if not isinstance(self.cfg[idx], str):
                    if self.opt.no_batch_norm:
                        mirror_model[idx_mirror_model].weight.data = layer.main_branch[0].weight.data
                        mirror_model[idx_mirror_model].bias.data = layer.main_branch[0].bias.data
                        conv_indices.append(idx_mirror_model)
                        idx_mirror_model +=2
                    else:
                        mirror_model[idx_mirror_model].weight.data = layer.main_branch[0].weight.data
                        mirror_model[idx_mirror_model].bias.data = layer.main_branch[0].bias.data
                        # Assign running mean and running var to Batch norm layer
                        mirror_model[idx_mirror_model+1].running_mean.data = layer.main_branch[1].running_mean.data
                        mirror_model[idx_mirror_model+1].running_var.data = layer.main_branch[1].running_var.data

                        conv_indices.append(idx_mirror_model)
                        idx_mirror_model +=3
                else:
                    #skip the Max pooling layer
                    idx_mirror_model += 1
            filters = []
            for conv_idx in conv_indices:
                layer_vis = CNNLayerVisualization(mirror_model, conv_idx, filter_nb)
                filter = layer_vis.visualise_layer_with_hooks(input_channels=self.opt.input_ch)
                filters.append(filter)

            filters = np.asarray(filters)
            writer.add_images(f'Learned Filters', filters,epoch, dataformats='NHWC')
    def get_main_branch_model(self):
        '''
        Returns a simple CNN model with the weights of the main branch. This is needed form the model stitching experiments
        '''
        if self.opt.backprop:
            conv_indices =[]
            mirror_model,_ = self._make_layers(self.input_dim, self.input_ch,mirror_model=True)

            if self.opt.train_mode == 'CE' :
                mirror_model = self.encoder.module
                for idx,layer in enumerate(self.encoder.module):
                    if isinstance(layer, nn.Conv2d):
                        conv_indices.append(idx)

            elif self.opt.train_mode=='so_mvment' or self.opt.train_mode=='mvment':
                mirror_model = self.encoder.module
            else:
                for idx,layer in enumerate(self.encoder.module.main_branch):
                    mirror_model = self.encoder.module.main_branch
                    if isinstance(layer,nn.Conv2d):
                        #mirror_model[idx].weight.data = self.encoder.module.main_branch[idx].weight.data
                        #mirror_model[idx].bias.data = self.encoder.module.main_branch[idx].bias.data
                        conv_indices.append(idx)
                    if isinstance(layer,nn.BatchNorm2d):
                        pass
                        #mirror_model[idx].running_mean.data = self.encoder.module.main_branch[idx].running_mean.data
                        #mirror_model[idx].running_var.data = self.encoder.module.main_branch[idx].running_var.data
        else:
            mirror_model,_ = self._make_layers(self.input_dim, self.input_ch,mirror_model=True)
            idx_mirror_model = 0
            conv_indices = [] #use them to access the cnn layers quickly during visualization
            #we assign the weight of the main branch of the trained model to the mirror_model so we can perform layer visualization
            for idx, layer in enumerate(self.encoder.module):
                if not isinstance(self.cfg[idx], str):
                    if self.opt.no_batch_norm:
                        mirror_model[idx_mirror_model].load_state_dict(layer.main_branch[0].state_dict())
                        #mirror_model[idx_mirror_model].weight.data = layer.main_branch[0].weight.data
                        #mirror_model[idx_mirror_model].bias.data = layer.main_branch[0].bias.data
                        conv_indices.append(idx_mirror_model)
                        idx_mirror_model +=2
                    else:
                        mirror_model[idx_mirror_model].load_state_dict(layer.main_branch[0].state_dict())
                        #mirror_model[idx_mirror_model].weight.data = layer.main_branch[0].weight.data
                        #mirror_model[idx_mirror_model].bias.data = layer.main_branch[0].bias.data
                        mirror_model[idx_mirror_model+1].load_state_dict(layer.main_branch[1].state_dict())
                        # Assign running mean and running var to Batch norm layer
                        #mirror_model[idx_mirror_model+1].running_mean.data = layer.main_branch[1].running_mean.data
                        #mirror_model[idx_mirror_model+1].running_var.data = layer.main_branch[1].running_var.data

                        conv_indices.append(idx_mirror_model)
                        idx_mirror_model +=3
                else:
                    #skip the Max pooling layer
                    idx_mirror_model += 1
        return mirror_model,conv_indices

def to_one_hot(y, n_dims=None):
    ''' Take integer tensor y with n dims and convert it to 1-hot representation with n+1 dims. '''
    y_tensor = y.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot


#TODO Classification evaluation script: Sample efficiency, noise robustness, incertainty measures,  switching datasets during evaluation ,continual learning
#TODO Representation Comparision script: linear Transformation, GradCam, Apple paper measure, Model stichting
#TODO Krotov Rule Training
#TODO build our own module

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch local error training')
    parser.add_argument('--vgg_name', default='vgg6',
                        help='model, mlp, vgg13, vgg16, vgg19, vgg8b, vgg11b, resnet18, resnet34, wresnet28-10 and more (default: vgg8b)')
    parser.add_argument('--train_mode', default='hinge',
                        help='ll')
    parser.add_argument(
        "--backprop",
        default=False,
        help="Boolean whether the k prediction weights W_pred of the last module should be unfrozen.",
    )
    #parser.add_argument('--contrast_mode', default="hinge",
    #                    help='ll')

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

    ################################################ General PArameters

    parser.add_argument(
        "--test_mode",
        default=False,
        help="Boolean to decide whether to use special weight initialization (delta orthogonal)",
    )
    parser.add_argument(
        "--input_dim",
        default=64,
        help="Boolean to decide whether to use special weight initialization (delta orthogonal)",
    )
    parser.add_argument(
        "--input_ch",
        default=1,
        help="Boolean to decide whether to use special weight initialization (delta orthogonal)",
    )
    parser.add_argument(
        "--num_classes",
        default=10,
        help="Boolean to decide whether to use special weight initialization (delta orthogonal)",
    )
    parser.add_argument(
        "--model_path",
        default='/home/ajaziri/Thesis_work/src/vision/logs/HingeLossCPC',
        help="Boolean to decide whether to use special weight initialization (delta orthogonal)",
    )
    args = parser.parse_args()

    train_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomCrop(64, padding=2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    vgg = FullModel(args)
    #vgg.save_model(epoch=3)
    #vgg.load_models(epoch=3)
    #vgg.load_models_incomplete(epoch=599,single_load=True)

    dataset_train = datasets.MNIST('../data/MNIST', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=4, shuffle=False)

    for img_idx, (input_tensor, labels) in enumerate(train_loader):
        label_one_hot = to_one_hot(labels)
        f= vgg(input_tensor,y=labels, y_onehot=label_one_hot)
        x = f['img_encoding']
        print('shape',x.shape)
        #f= vgg(input_tensor)

        #print("shape output",f.shape) #y=labels, y_onehot=label_one_hot
    #    print("LOSS", h.shape, next_layer_input.shape)
        # print(h_return.shape,h.shape)