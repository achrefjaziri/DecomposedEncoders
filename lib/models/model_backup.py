"""Wrapper for all our Deep learning models"""
import os,sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from models.unet_model import UNet_Encoder,UNet_Decoder, Projection_Head
from models.attn_unet_model import AttUNet_Encoder,AttUNet_Decoder
from models.axial_attention_nets import MedT,MedNet_Encoder,MedNet_Decoder
from models.axial_attention_blocks import AxialBlock,AxialBlock_dynamic,AxialBlock_wopos
from utils.losses import DiceLoss, GeneralizedDiceLoss, WeightedCrossEntropyLoss,LogNLLLoss
from models.deeplabv3 import DeepLabV3Plus_Encoder,DeepLabV3Plus_Decoder
from collections import OrderedDict
import torch.nn.functional as F
from torchsummary import summary
from torch.autograd import Variable
import argparse, yaml


def init_model(model, init_method='normal', gain=0.5):
    def init_func(m):
        # this will apply to each layer
        classname = m.__class__.__name__
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if init_method == 'normal':
                torch.nn.init.normal_(m.weight, 0.0, gain)
            elif init_method == 'xavier':
                torch.nn.init.xavier_normal_(m.weight, gain=gain)
            elif init_method == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')  # good for relu
            elif init_method == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_method)


        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, gain)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    model.apply(init_func)

class Model():
    def __init__(self, configs,test=False):
        self.configs = configs
        self.test_mode = test
        self.device = torch.device('cuda' if (torch.cuda.is_available() and not self.configs['CPU_MODE']) else 'cpu')

        if self.configs['ARCHI'] == 'Unet':
            self.encoder = UNet_Encoder(self.configs['INPUT_CHANNELS'])
            self.decoder = UNet_Decoder(self.configs['SEG_CLASSES'])
            if not self.configs['CPU_MODE'] : #and not configs['RESUME']
                self.encoder = torch.nn.DataParallel(self.encoder, device_ids=list(range(torch.cuda.device_count())))
                self.decoder = torch.nn.DataParallel(self.decoder, #
                                                 device_ids=list(range(torch.cuda.device_count())))
        elif self.configs['ARCHI'] == 'MedT':
            self.encoder = MedNet_Encoder(AxialBlock_dynamic, AxialBlock_wopos, [1, 2, 4, 1], s=0.125, img_size=self.configs['INPUT_SIZE'],
                                     imgchan=self.configs['INPUT_CHANNELS'])
            self.decoder = MedNet_Decoder(num_classes=self.configs['SEG_CLASSES'])
            if not self.configs['CPU_MODE'] : #and not configs['RESUME']
                self.encoder = torch.nn.DataParallel(self.encoder, device_ids=list(range(torch.cuda.device_count())))
                self.decoder = torch.nn.DataParallel(self.decoder, #
                                                 device_ids=list(range(torch.cuda.device_count())))
        elif self.configs['ARCHI'] == 'AttnUnet':
                self.encoder = AttUNet_Encoder(self.configs['INPUT_CHANNELS'])
                self.decoder = AttUNet_Decoder(self.configs['SEG_CLASSES'])
                if not self.configs['CPU_MODE']:  # and not configs['RESUME']
                    self.encoder = torch.nn.DataParallel(self.encoder,
                                                         device_ids=list(range(torch.cuda.device_count())))
                    self.decoder = torch.nn.DataParallel(self.decoder,  #
                                                         device_ids=list(range(torch.cuda.device_count())))
        elif self.configs['ARCHI'] =='Deeplab':
            self.encoder = DeepLabV3Plus_Encoder(
                in_ch=self.configs['INPUT_CHANNELS'],
                n_classes=self.configs['SEG_CLASSES'],
                n_blocks=[3, 4, 23, 3],
                atrous_rates=[6, 12, 18],
                multi_grids=[1, 2, 4],
                output_stride=16,
                )
            self.decoder = DeepLabV3Plus_Decoder(
                n_classes=self.configs['SEG_CLASSES'],
                size=[self.configs['INPUT_SIZE'],self.configs['INPUT_SIZE']]
                )
            if not self.configs['CPU_MODE']:  # and not configs['RESUME']
                self.encoder = torch.nn.DataParallel(self.encoder,
                                                     device_ids=list(range(torch.cuda.device_count())))
                self.decoder = torch.nn.DataParallel(self.decoder,  #
                                                     device_ids=list(range(torch.cuda.device_count())))
        if 'INIT_MODEL' in self.configs:
            init_model(self.encoder,configs['INIT_MODEL'])
            init_model(self.decoder,configs['INIT_MODEL'])

        # Loss function
        if configs['LOSS']=='CE':
            self.criterion =  nn.CrossEntropyLoss() # DiceLoss(weight =class_weights)
        elif configs['LOSS']=='DICE':
            self.criterion =    DiceLoss()
        elif configs['LOSS']=='GEN_DICE':
            self.criterion = GeneralizedDiceLoss()
        elif configs['LOSS']=='WCE':
            self.criterion = WeightedCrossEntropyLoss()
        elif configs['LOSS']=='LOGNLL':
            self.criterion = LogNLLLoss()

        if not self.test_mode:
            # Optimizer
            self.optimizer_encoder =  torch.optim.Adam(self.encoder.parameters(), lr=self.configs['LR'],
                             weight_decay=1e-5)

            self.optimizer_decoder = torch.optim.Adam(self.decoder.parameters(), lr=self.configs['LR'],
                             weight_decay=1e-5)

            # gamma = decaying factor
            self.scheduler_encoder = StepLR(self.optimizer_encoder, step_size=2, gamma=0.96)  # TODO check for the best scheduler
            self.scheduler_decoder = StepLR(self.optimizer_decoder, step_size=2, gamma=0.96)  # TODO check for the best scheduler

        self.img_output =None

    def set_inputs(self,img,gt = None):
        self.img_input = img #.to(f'cuda:{self.encoder.device_ids[0]}')
        #if self.configs['TRAIN']:
        self.gt = gt
    def forward(self):
        #TODO  ad hoc solution to normalize the input between 0 and 1. Check the literature for better solution
        self.input_img_copy = np.copy(self.img_input.cpu().detach().numpy())
        self.img_input = self.img_input.type(torch.float)
        current_batch = self.img_input.size(0)
        self.img_input = self.img_input.view(self.img_input.size(0), -1)
        self.img_input -= self.img_input.min(1, keepdim=True)[0]
        self.img_input /= self.img_input.max(1, keepdim=True)[0]
        if np.isnan(self.img_input.cpu().detach().numpy()).any():
            #print('Set nan values to 0')
            self.img_input[torch.isnan(self.img_input)] = 0

        self.img_input = self.img_input.view(current_batch,self.configs['INPUT_CHANNELS'],self.configs['INPUT_SIZE'],self.configs['INPUT_SIZE'])
        self.img_encoding = self.encoder(self.img_input)
        self.img_output = self.decoder(self.img_encoding)
        #print(self.img_output.shape)
    def get_outputs(self):
        return self.img_output

    def train(self):
        self.forward()
        if self.configs['LOSS'] !='DICE':
            self.gt = self.gt.squeeze(1).type(torch.long)

        #print('Shapes',self.img_output.shape,self.gt.shape)
        #print('unique Labels', torch.unique(self.gt))
        #zero = torch.zeros_like(self.gt)
        #self.gt = torch.where(self.gt > 5, zero,self.gt)

        #if 1 in torch.as_tensor((torch.unique(self.gt) - 5) > 0, dtype=torch.int32):
        #    print('Labels False!', torch.unique(self.gt))

        loss = self.criterion(self.img_output, self.gt)
        if np.isnan(loss.item()):
            print('output nans', np.isnan(self.img_output.cpu().detach().numpy()).any())
            print('input nans', np.isnan(self.img_input.cpu().detach().numpy()).any())
            print('before normalization',np.isnan(self.input_img_copy).any())
            print('gt nans', np.isnan(self.gt.cpu().detach().numpy()).any())
            #print('output',self.img_output)
            #print('inputs',self.img_input)
            sys.exit('FATAL error: Nan Values')
        else:
            #only backpropagate a loss that is nont nan
            loss.backward()
        return loss
    def update_weights(self):
        self.optimizer_encoder.step()
        self.optimizer_encoder.zero_grad()

        self.optimizer_decoder.step()
        self.optimizer_decoder.zero_grad()

    def eval_mode(self):
        self.encoder.eval()
        self.decoder.eval()
    def test(self,return_loss=False):
        self.forward()
        if return_loss:
            if self.configs['LOSS'] !='DICE':
                self.gt = self.gt.squeeze(1).type(torch.long)

            zero = torch.zeros_like(self.gt)
            self.gt = torch.where(self.gt > 5, zero, self.gt)
            #print('SHAPE GT',self.gt.shape,self.img_output.shape)
            loss = self.criterion(self.img_output, self.gt)
            return self.img_output,loss
        return self.img_output
    def get_lr(self):
        return self.scheduler_encoder.get_last_lr()
    def update_lr(self):
        self.scheduler_encoder.step()
        self.scheduler_decoder.step()

    def save_model(self, path, epoch, best_acc =None):
        checkpoint_enocder = {'epoch': epoch + 1,
                        'encoder': True,
                      'state_dict': self.encoder.state_dict(),
                      'optimizer': self.optimizer_encoder.state_dict(),
                       'best_acc':  best_acc    }

        checkpoint_decoder= {'epoch': epoch + 1,
                             'encoder': False,
                              'state_dict': self.decoder.state_dict(),
                              'optimizer': self.optimizer_decoder.state_dict(),
                              'best_acc': best_acc}

        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(checkpoint_enocder, path + "/encoder.pth.tar") # alternative: "/encoder_epoch_{0}.pth.tar".format(epoch)
        torch.save(checkpoint_decoder, path + "/decoder.pth.tar") # "/decoder_epoch_{0}.pth.tar".format(epoch)


    def load_models(self, load_decoder = True):
        #get paths from configs
        #load models + optimizers
        if os.path.isfile(self.configs['PATH_ENCODER']):
            print("=> loading checkpoint '{}'".format(self.configs['PATH_ENCODER']))
            checkpoint = torch.load(self.configs['PATH_ENCODER'])
            epoch_start = checkpoint['epoch'] -1
            #state_dict = checkpoint['state_dict']
            #new_state_dict = OrderedDict()
            #for k, v in state_dict.items():
            #    name = k[7:]  # remove 'module.' of dataparallel
            #    new_state_dict[name] = v

            #self.encoder.load_state_dict(new_state_dict)

            self.encoder.load_state_dict(checkpoint['state_dict'])
            if not self.test_mode:
                self.optimizer_encoder.load_state_dict(checkpoint['optimizer'])
            print("=> loaded Encoder checkpoint '{}' (epoch {})"
                  .format(self.configs['PATH_ENCODER'], checkpoint['epoch']))
        else:
        	print("MEEEHH", self.configs['PATH_ENCODER'])
        if load_decoder:
            print('*******************************')
            print(self.configs['PATH_DECODER'])
            if os.path.isfile(self.configs['PATH_DECODER']):
                print("=> loading checkpoint '{}'".format(self.configs['PATH_DECODER']))
                checkpoint = torch.load(self.configs['PATH_DECODER'])
                epoch_start = checkpoint['epoch'] -1
                #state_dict = checkpoint['state_dict']
                #new_state_dict = OrderedDict()
                #for k, v in state_dict.items():
                #    name = k[7:]  # remove 'module.' of dataparallel
                #    new_state_dict[name] = v

                #self.decoder.load_state_dict(new_state_dict)
                self.decoder.load_state_dict(checkpoint['state_dict'])
                if not self.test_mode:
                    self.optimizer_decoder.load_state_dict(checkpoint['optimizer'])
                print("=> loaded Decoder checkpoint '{}' (epoch {})"
                      .format(self.configs['PATH_DECODER'], checkpoint['epoch']))

        #self.encoder = torch.nn.DataParallel(self.encoder, device_ids=list(range(torch.cuda.device_count())))
        #self.decoder = torch.nn.DataParallel(self.decoder,  #
        #                                     device_ids=list(range(torch.cuda.device_count())))
        return epoch_start


if __name__ == "__main__":
    print('starting!!')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/unet_config.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    with open(opts.config, 'r') as f_in:
        configs = yaml.safe_load(f_in)
    print(configs)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(configs["GPU_IDS"][0])
    device = torch.device('cuda' if (torch.cuda.is_available() and not configs['CPU_MODE']) else 'cpu')
    m = Model(configs)
    im = torch.randn(1, 1, 256, 256).to(device)
    gt = torch.zeros(1,256,256).to(device)

    #im = Variable(im.cuda())
    m.set_inputs(im,gt)
    #m.eval_mode()
    out,l = m.test(True)
    #print(out.shape,l)


