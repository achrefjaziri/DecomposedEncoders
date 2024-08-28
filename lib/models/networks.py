class ResNet(nn.Module):
    '''
    Residual network.
    The network can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.
    '''

    def __init__(self, block, num_blocks, num_classes, input_ch, feature_multiplyer, input_dim):
        super(ResNet, self).__init__()
        self.in_planes = int(feature_multiplyer * 64)
        self.conv1 = LocalLossBlockConv(input_ch, int(feature_multiplyer * 64), 3, 1, 1, num_classes, input_dim,
                                        bias=False, post_act=not args.pre_act)
        self.layer1 = self._make_layer(block, int(feature_multiplyer * 64), num_blocks[0], 1, num_classes, input_dim)
        self.layer2 = self._make_layer(block, int(feature_multiplyer * 128), num_blocks[1], 2, num_classes, input_dim)
        self.layer3 = self._make_layer(block, int(feature_multiplyer * 256), num_blocks[2], 2, num_classes,
                                       input_dim // 2)
        self.layer4 = self._make_layer(block, int(feature_multiplyer * 512), num_blocks[3], 2, num_classes,
                                       input_dim // 4)
        self.linear = nn.Linear(int(feature_multiplyer * 512 * block.expansion), num_classes)
        if not args.backprop:
            self.linear.weight.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride, num_classes, input_dim):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        stride_cum = 1
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, num_classes, input_dim // stride_cum))
            stride_cum *= stride
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def parameters(self):
        if not args.backprop:
            return self.linear.parameters()
        else:
            return super(ResNet, self).parameters()

    def set_learning_rate(self, lr):
        self.conv1.set_learning_rate(lr)
        for layer in self.layer1:
            layer.set_learning_rate(lr)
        for layer in self.layer2:
            layer.set_learning_rate(lr)
        for layer in self.layer3:
            layer.set_learning_rate(lr)
        for layer in self.layer4:
            layer.set_learning_rate(lr)

    def optim_zero_grad(self):
        self.conv1.optim_zero_grad()
        for layer in self.layer1:
            layer.optim_zero_grad()
        for layer in self.layer2:
            layer.optim_zero_grad()
        for layer in self.layer3:
            layer.optim_zero_grad()
        for layer in self.layer4:
            layer.optim_zero_grad()

    def optim_step(self):
        self.conv1.optim_step()
        for layer in self.layer1:
            layer.optim_step()
        for layer in self.layer2:
            layer.optim_step()
        for layer in self.layer3:
            layer.optim_step()
        for layer in self.layer4:
            layer.optim_step()

    def forward(self, x, y, y_onehot):
        x, loss = self.conv1(x, y, y_onehot)
        x, _, _, loss = self.layer1((x, y, y_onehot, loss))
        x, _, _, loss = self.layer2((x, y, y_onehot, loss))
        x, _, _, loss = self.layer3((x, y, y_onehot, loss))
        x, _, _, loss = self.layer4((x, y, y_onehot, loss))
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x, loss


class wide_basic(nn.Module):
    ''' Used in WideResNet() '''

    def __init__(self, in_planes, planes, dropout_rate, stride, num_classes, input_dim, adapted):
        super(wide_basic, self).__init__()
        self.adapted = adapted
        self.conv1 = LocalLossBlockConv(in_planes, planes, 3, 1, 1, num_classes, input_dim * stride,
                                        dropout=None if self.adapted else 0, bias=True, pre_act=True, post_act=False)
        if not self.adapted:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = LocalLossBlockConv(planes, planes, 3, stride, 1, num_classes, input_dim,
                                        dropout=None if self.adapted else 0, bias=True, pre_act=True, post_act=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )
            if args.optim == 'sgd':
                self.optimizer = optim.SGD(self.shortcut.parameters(), lr=0, weight_decay=args.weight_decay,
                                           momentum=args.momentum)
            elif args.optim == 'adam' or args.optim == 'amsgrad':
                self.optimizer = optim.Adam(self.shortcut.parameters(), lr=0, weight_decay=args.weight_decay,
                                            amsgrad=args.optim == 'amsgrad')

    def set_learning_rate(self, lr):
        self.lr = lr
        self.conv1.set_learning_rate(lr)
        self.conv2.set_learning_rate(lr)
        if len(self.shortcut) > 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def optim_zero_grad(self):
        self.conv1.optim_zero_grad()
        self.conv2.optim_zero_grad()
        if len(self.shortcut) > 0:
            self.optimizer.zero_grad()

    def optim_step(self):
        self.conv1.optim_step()
        self.conv2.optim_step()
        if len(self.shortcut) > 0:
            self.optimizer.step()

    def forward(self, input):
        x, y, y_onehot, loss_total = input
        out, loss = self.conv1(x, y, y_onehot)
        loss_total += loss
        if not self.adapted:
            out = self.dropout(out)
        out, loss = self.conv2(out, y, y_onehot, self.shortcut(x))
        loss_total += loss
        if not args.no_detach:
            if len(self.shortcut) > 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        return (out, y, y_onehot, loss_total)


class Wide_ResNet(nn.Module):
    '''
    Wide residual network.
    The network can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.
    '''

    def __init__(self, depth, widen_factor, dropout_rate, num_classes, input_ch, input_dim, adapted=False):
        super(Wide_ResNet, self).__init__()
        self.adapted = adapted
        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor

        print('| Wide-Resnet %dx%d %s' % (depth, k, 'adapted' if adapted else ''))
        if self.adapted:
            nStages = [16 * k, 16 * k, 32 * k, 64 * k]
        else:
            nStages = [16, 16 * k, 32 * k, 64 * k]
        self.in_planes = nStages[0]

        self.conv1 = LocalLossBlockConv(input_ch, nStages[0], 3, 1, 1, num_classes, 32, dropout=0, bias=True,
                                        post_act=False)
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, 1, num_classes, input_dim, adapted)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, 2, num_classes, input_dim, adapted)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, 2, num_classes, input_dim // 2, adapted)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3] * (16 if self.adapted else 1), num_classes)
        if not args.backprop:
            self.linear.weight.data.zero_()

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, num_classes, input_dim, adapted):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        stride_cum = 1
        for stride in strides:
            stride_cum *= stride
            layers.append(
                block(self.in_planes, planes, dropout_rate, stride, num_classes, input_dim // stride_cum, adapted))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def parameters(self):
        if not args.backprop:
            return self.linear.parameters()
        else:
            return super(Wide_ResNet, self).parameters()

    def set_learning_rate(self, lr):
        self.conv1.set_learning_rate(lr)
        for layer in self.layer1:
            layer.set_learning_rate(lr)
        for layer in self.layer2:
            layer.set_learning_rate(lr)
        for layer in self.layer3:
            layer.set_learning_rate(lr)

    def optim_zero_grad(self):
        self.conv1.optim_zero_grad()
        for layer in self.layer1:
            layer.optim_zero_grad()
        for layer in self.layer2:
            layer.optim_zero_grad()
        for layer in self.layer3:
            layer.optim_zero_grad()

    def optim_step(self):
        self.conv1.optim_step()
        for layer in self.layer1:
            layer.optim_step()
        for layer in self.layer2:
            layer.optim_step()
        for layer in self.layer3:
            layer.optim_step()

    def forward(self, x, y, y_onehot):
        x, loss = self.conv1(x, y, y_onehot)
        x, _, _, loss = self.layer1((x, y, y_onehot, loss))
        x, _, _, loss = self.layer2((x, y, y_onehot, loss))
        x, _, _, loss = self.layer3((x, y, y_onehot, loss))
        x = F.relu(self.bn1(x))
        if self.adapted:
            x = F.max_pool2d(x, 2)
        else:
            x = F.avg_pool2d(x, 8)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x, loss


class Net(nn.Module):
    '''
    A fully connected network.
    The network can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.

    Args:
        num_layers (int): Number of hidden layers.
        num_hidden (int): Number of units in each hidden layer.
        input_dim (int): Feature map height/width for input.
        input_ch (int): Number of feature maps for input.
        num_classes (int): Number of classes (used in local prediction loss).
    '''

    def __init__(self, num_layers, num_hidden, input_dim, input_ch, num_classes):
        super(Net, self).__init__()

        self.num_hidden = num_hidden
        self.num_layers = num_layers
        reduce_factor = 1
        self.layers = nn.ModuleList(
            [LocalLossBlockLinear(input_dim * input_dim * input_ch, num_hidden, num_classes, first_layer=True)])
        self.layers.extend([LocalLossBlockLinear(int(num_hidden // (reduce_factor ** (i - 1))),
                                                 int(num_hidden // (reduce_factor ** i)), num_classes) for i in
                            range(1, num_layers)])
        self.layer_out = nn.Linear(int(num_hidden // (reduce_factor ** (num_layers - 1))), num_classes)
        if not args.backprop:
            self.layer_out.weight.data.zero_()

    def parameters(self):
        if not args.backprop:
            return self.layer_out.parameters()
        else:
            return super(Net, self).parameters()

    def set_learning_rate(self, lr):
        for i, layer in enumerate(self.layers):
            layer.set_learning_rate(lr)

    def optim_zero_grad(self):
        for i, layer in enumerate(self.layers):
            layer.optim_zero_grad()

    def optim_step(self):
        for i, layer in enumerate(self.layers):
            layer.optim_step()

    def forward(self, x, y, y_onehot):
        x = x.view(x.size(0), -1)
        total_loss = 0.0
        for i, layer in enumerate(self.layers):
            x, loss = layer(x, y, y_onehot)
            total_loss += loss
        x = self.layer_out(x)

        return x, total_loss


cfg = {
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

# Partly taken from (30 July 2020)
# https://pytorch.org/docs/stable/_modules/torchvision/models/vgg.html#vgg11
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os

from CLAPPVision.vision.models import ContrastiveLoss, Supervised_Loss
from CLAPPVision.utils import model_utils


class VGG_like_Encoder(nn.Module):
    def __init__(
            self,
            opt,
            block_idx,
            blocks,
            in_channels,
            patch_size=16,
            overlap_factor=2,
            calc_loss=False,
    ):
        super(VGG_like_Encoder, self).__init__()
        self.encoder_num = block_idx
        self.opt = opt

        self.save_vars = self.opt.save_vars_for_update_calc == block_idx + 1

        # Layer
        self.model = self.make_layers(blocks[block_idx], in_channels)

        # Params
        self.calc_loss = calc_loss

        self.overlap = overlap_factor
        self.increasing_patch_size = self.opt.increasing_patch_size
        if self.increasing_patch_size:  # This is experimental... take care, this must be synced with architecture, i.e. number and position of downsampling layers (stride 2, e.g. pooling)
            if self.overlap != 2:
                raise ValueError("if --increasing_patch_size is true, overlap(_factor) has to be equal 2")
            patch_sizes = [4, 4, 8, 8, 16, 16]
            self.patch_size_eff = patch_sizes[block_idx]
            self.max_patch_size = max(patch_sizes)
            high_level_patch_sizes = [4, 4, 4, 4, 4, 2]
            self.patch_size = high_level_patch_sizes[block_idx]
        else:
            self.patch_size = patch_size

        reduced_patch_pool_sizes = [4, 4, 3, 3, 2, 1]
        if opt.reduced_patch_pooling:
            self.patch_average_pool_out_dim = reduced_patch_pool_sizes[block_idx]
        else:
            self.patch_average_pool_out_dim = 1

        self.predict_module_num = self.opt.predict_module_num
        self.extra_conv = self.opt.extra_conv
        self.inpatch_prediction = self.opt.inpatch_prediction
        self.inpatch_prediction_limit = self.opt.inpatch_prediction_limit
        self.asymmetric_W_pred = self.opt.asymmetric_W_pred

        if opt.gradual_prediction_steps:
            prediction_steps = min(block_idx + 1, self.opt.prediction_step)
        else:
            prediction_steps = self.opt.prediction_step

        def get_last_index(block):
            if block[-1] == 'M':
                last_ind = -2
            else:
                last_ind = -1
            return last_ind

        last_ind = get_last_index(blocks[block_idx])
        self.in_planes = blocks[block_idx][last_ind]
        # in_channels_loss: z, out_channels: c
        if self.predict_module_num == '-1' or self.predict_module_num == 'both':
            if self.encoder_num == 0:  # exclude first module
                in_channels_loss = self.in_planes
                if opt.reduced_patch_pooling:
                    in_channels_loss *= reduced_patch_pool_sizes[block_idx] ** 2
            else:
                last_ind_block_below = get_last_index(blocks[block_idx - 1])
                in_channels_loss = blocks[block_idx - 1][last_ind_block_below]
                if opt.reduced_patch_pooling:
                    in_channels_loss *= reduced_patch_pool_sizes[block_idx - 1] ** 2
        else:
            in_channels_loss = self.in_planes
            if opt.reduced_patch_pooling:
                in_channels_loss *= reduced_patch_pool_sizes[block_idx] ** 2

        # Optional extra conv layer to increase rec. field size
        if self.extra_conv and self.encoder_num < 3:
            self.extra_conv_layer = nn.Conv2d(self.in_planes, self.in_planes, stride=3, kernel_size=3, padding=1)

        # in_channels_loss: z, out_channels: c
        if self.predict_module_num == '-1b':
            if self.encoder_num == len(blocks) - 1:  # exclude last module
                out_channels = self.in_planes
                if opt.reduced_patch_pooling:
                    out_channels *= reduced_patch_pool_sizes[block_idx] ** 2
            else:
                last_ind_block_above = get_last_index(blocks[block_idx + 1])
                out_channels = blocks[block_idx + 1][last_ind_block_above]
                if opt.reduced_patch_pooling:
                    out_channels *= reduced_patch_pool_sizes[block_idx + 1] ** 2
        else:
            out_channels = self.in_planes
            if opt.reduced_patch_pooling:
                out_channels *= reduced_patch_pool_sizes[block_idx] ** 2

        # Loss module; is always present, but only gets used when training CLAPPVision modules
        # in_channels_loss: z, out_channels: c
        if self.opt.loss == 0:
            self.loss = ContrastiveLoss.ContrastiveLoss(
                opt,
                in_channels=in_channels_loss,  # z
                out_channels=out_channels,  # c
                prediction_steps=prediction_steps,
                save_vars=self.save_vars
            )
            if self.predict_module_num == 'both':
                self.loss_same_module = ContrastiveLoss.ContrastiveLoss(
                    opt,
                    in_channels=in_channels_loss,
                    out_channels=in_channels_loss,  # on purpose, cause in_channels_loss is layer below
                    prediction_steps=prediction_steps
                )
            if self.asymmetric_W_pred:
                self.loss_mirror = ContrastiveLoss.ContrastiveLoss(
                    opt,
                    in_channels=in_channels_loss,
                    out_channels=out_channels,
                    prediction_steps=prediction_steps
                )
        elif self.opt.loss == 1:
            self.loss = Supervised_Loss.Supervised_Loss(opt, in_channels_loss, True)
        else:
            raise Exception("Invalid option")

        # Optional recurrent weights, Experimental!
        if self.opt.inference_recurrence == 1 or self.opt.inference_recurrence == 3:  # 1 - lateral recurrence within layer
            self.recurrent_weights = nn.Conv2d(self.in_planes, self.in_planes, 1, bias=False)
        if self.opt.inference_recurrence == 2 or self.opt.inference_recurrence == 3:  # 2 - feedback recurrence, 3 - both, lateral and feedback recurrence
            if self.encoder_num < len(blocks) - 1:  # exclude last module
                last_ind_block_above = get_last_index(blocks[block_idx + 1])
                rec_dim_block_above = blocks[block_idx + 1][last_ind_block_above]
                self.recurrent_weights_fb = nn.Conv2d(rec_dim_block_above, self.in_planes, 1, bias=False)

        if self.opt.weight_init:
            raise NotImplementedError("Weight init not implemented for vgg")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)

    def make_layers(self, block, in_channels, batch_norm=False, inplace=False):
        layers = []
        for v in block:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=inplace)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=inplace)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x, reps, t, n_patches_y, n_patches_x, label):
        # x: either input dims b, c, Y, X or (if coming from lower module which did unfolding, as variable z):  b * n_patches_y * n_patches_x, c, y, x
        # Input preparation, i.e unfolding into patches. Usually only needed for first module. More complicated for experimental increasing_patch_size option.
        # print('ENCODER NUMBER',self.encoder_num)
        if self.encoder_num in [0, 2, 4]:  # [0,2,4,5]
            # if increasing_patch_size is enabled, this has to be in sync with architecture and intended patch_size for respective module:
            # for every layer that increases the patch_size, the extra downsampling + unfolding has to be done!
            if self.encoder_num > 0 and self.increasing_patch_size:
                # undo unfolding of the previous module
                s1 = x.shape
                x = x.reshape(-1, n_patches_y, n_patches_x, s1[1], s1[2], s1[3])  # b, n_patches_y, n_patches_x, c, y, x
                # downsampling to get rid of the overlaps between paches of the previous module
                x = x[:, ::2, ::2, :, :, :]  # b, n_patches_x_red, n_patches_y_red, c, y, x.
                s = x.shape
                x = x.permute(0, 3, 2, 5, 1, 4).reshape(s[0], s[3], s[2], s[5], s[1] * s[4]).permute(0, 1, 4, 2,
                                                                                                     3).reshape(s[0],
                                                                                                                s[3],
                                                                                                                s[1] *
                                                                                                                s[4],
                                                                                                                s[2] *
                                                                                                                s[
                                                                                                                    5])  # b, c, Y, X

            if self.encoder_num == 0 or self.increasing_patch_size:
                # x enters as (4,1,64,64) - (batch_size,channels,height,width)
                x = (  # b, c, y, x
                    x.unfold(2, self.patch_size, self.patch_size // self.overlap)  # b, c, n_patches_y, x, patch_size
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

                # x gets reshaped to (batch_Size* n_patches_y * n_patches_x, nb_channels,patch_size,patch_size) - (196,1,16,16)

        # Main encoding step
        # forward through self.model is split into (conv)/(nonlin + pool) due to (optional) recurrence
        # assuming arch = [128, 256, 'M', 256, 512, 'M', 1024, 'M', 1024, 'M']
        if self.opt.inference_recurrence > 0:  # in case of recurrence
            if self.opt.model_splits == 6:
                split_ind = 1
            elif self.opt.model_splits == 3 or self.opt.model_splits == 1:
                split_ind = -2
            else:
                raise NotImplementedError("Recurrence is only implemented for model_splits = 1, 3 or 6")
        else:  # without recurrence split does not really matter, arbitrily choose 1
            split_ind = 1

        if self.save_vars:  # save input for (optional) manual update calculation
            torch.save(x, os.path.join(self.opt.model_path,
                                       'saved_input_layer_' + str(self.opt.save_vars_for_update_calc)))

        # 1. Apply encoding weights of model (e.g. conv2d layer)
        # print('Shape of inputs',x.shape,x.requires_grad)
        # cur_device = x_orig.get_device()
        # x_i = torch.rand(x_input.shape, requires_grad=True).to(cur_device)
        z = self.model[:split_ind](x)  # .clone().detach() # b * n_patches_y * n_patches_x, c, y, x

        # print('Current layer',self.model[:split_ind],x.shape)
        # z.requires_grad = True
        # z = torch.rand((1568,self.model[:split_ind][0].out_channels,x_input.shape[2],x_input.shape[2]), requires_grad=True).to(cur_device)
        # print('z shape',z.shape,self.model[:split_ind][0].out_channels)
        # Tensor of shape (196,128,16,16) for first layer. Then (196,156,8,8)

        # 2. Add (optional) recurrence if present
        # expand dimensionality if rec comes from layer after one or several (2x2 strided, i.e. downsampled) pooling layer(s). tensor.repeat_interleave() would do, but not available in pytorch 1.0.0
        def expand_2_by_2(rec):
            srec = rec.shape
            return (
                rec.unfold(2, 1, 1).repeat((1, 1, 1, 1, 2)).permute(0, 1, 2, 4, 3).reshape(srec[0], srec[1],
                                                                                           2 * srec[2], srec[3])
                    .unfold(3, 1, 1).repeat((1, 1, 1, 1, 2)).reshape(srec[0], srec[1], 2 * srec[2], 2 * srec[3])
            )

        if t > 0:  # only apply rec if iteration is not the first one (is not entered if recurrence is off since then t = 0)
            if self.opt.inference_recurrence == 1 or self.opt.inference_recurrence == 3:  # 1 - lateral recurrence within layer
                rec = self.recurrent_weights(
                    reps[self.encoder_num].clone().detach())  # Detach input to implement e-prop like BPTT
                while z.shape != rec.shape:  # if rec comes from strided pooling layer
                    rec = expand_2_by_2(rec)
                z += rec
            if self.opt.inference_recurrence == 2 or self.opt.inference_recurrence == 3:  # 2 - feedback recurrence, 3 - both, lateral and feedback recurrence
                if self.encoder_num < len(reps) - 1:  # exclude last module
                    rec_fb = self.recurrent_weights_fb(
                        reps[self.encoder_num + 1].clone().detach())  # Detach input to implement e-prop like BPTT
                    while z.shape != rec_fb.shape:  # if rec comes from strided pooling layer
                        rec_fb = expand_2_by_2(rec_fb)
                    z += rec_fb

        # 3. Apply nonlin and 'rest' of model (e.g. ReLU, MaxPool etc...)
        z = self.model[split_ind:](z)  # b * n_patches_y * n_patches_x, c, y, x

        # Optional extra conv layer with downsampling (stride > 1) here to increase receptive field size ###
        if self.extra_conv and self.encoder_num < 3:
            dec = self.extra_conv_layer(z)
            dec = F.relu(dec, inplace=False)
        else:
            dec = z

        # Optional in-patch prediction
        # if opt: change CPC task to smaller scale prediction (within patch -> smaller receptive field)
        # by extra unfolding + "cropping" (to avoid overweighing lower layers and memory overflow)
        if self.inpatch_prediction and self.encoder_num < self.inpatch_prediction_limit:
            extra_patch_size = [2 for _ in range(self.inpatch_prediction_limit)]
            extra_patch_steps = [1 for _ in range(self.inpatch_prediction_limit)]

            dec = dec.reshape(-1, n_patches_x, n_patches_y, dec.shape[1], dec.shape[2],
                              dec.shape[3])  # b, n_patches_y, n_patches_x, c, y, x
            # random "cropping"/selecting of patches that will be extra unfolded
            extra_crop_size = [n_patches_x // 2 for _ in range(self.inpatch_prediction_limit)]
            inds = np.random.randint(0, n_patches_x - extra_crop_size[self.encoder_num], 2)
            dec = dec[:, inds[0]:inds[0] + extra_crop_size[self.encoder_num],
                  inds[1]:inds[1] + extra_crop_size[self.encoder_num], :, :, :]

            # extra unfolding
            dec = (
                dec.unfold(4, extra_patch_size[self.encoder_num], extra_patch_steps[self.encoder_num])
                    .unfold(5, extra_patch_size[self.encoder_num], extra_patch_steps[
                    self.encoder_num])  # b, n_patches_y, n_patches_x, c, n_extra_patches, n_extra_patches, extra_patch_size, extra_patch_size
                    .permute(0, 1, 2, 4, 5, 3, 6, 7)
            # b, n_patches_y(_reduced), n_patches_x(_reduced), n_extra_patches, n_extra_patches, c, extra_patch_size, extra_patch_size
            )
            n_extra_patches = dec.shape[3]
            dec = dec.reshape(dec.shape[0] * dec.shape[1] * dec.shape[2] * dec.shape[3] * dec.shape[4], dec.shape[5],
                              dec.shape[6], dec.shape[7])
            # b * n_patches_y(_reduced) * n_patches_x(_reduced) * n_extra_patches * n_extra_patches, c, extra_patch_size, extra_patch_size

        # Pool over patch
        # in original CPC/GIM, pooling is done over whole patch, i.e. output shape 1 by 1
        out = F.adaptive_avg_pool2d(dec,
                                    self.patch_average_pool_out_dim)  # b * n_patches_y(_reduced) * n_patches_x(_reduced) (* n_extra_patches * n_extra_patches), c, x_pooled, y_pooled
        # for  self.patch_average_pool_out_dim=1: out has shape (198,128) we pooled each (16,16) to 1 for layer 0.

        # Flatten over channel and pooled patch dimensions x_pooled, y_pooled:
        out = out.reshape(out.shape[0],
                          -1)  # b * n_patches_y(_reduced) * n_patches_x(_reduced) (* n_extra_patches * n_extra_patches),  c * y_pooled * x_pooled
        # out has shape (4,7,7,128) we just seperated 198 to 3 dims

        if self.inpatch_prediction and self.encoder_num < self.inpatch_prediction_limit:
            n_p_x, n_p_y = n_extra_patches, n_extra_patches
        else:
            n_p_x, n_p_y = n_patches_x, n_patches_y

        out = out.reshape(-1, n_p_y, n_p_x, out.shape[
            1])  # b, n_patches_y, n_patches_x, c * y_pooled * x_pooled OR  b * n_patches_y(_reduced) * n_patches_x(_reduced), n_extra_patches, n_extra_patches, c * y_pooled * x_pooled
        out = out.permute(0, 3, 1,
                          2).contiguous()  # b, c * y_pooled * x_pooled, n_patches_y, n_patches_x  OR  b * n_patches_y(_reduced) * n_patches_x(_reduced), c * y_pooled * x_pooled, n_extra_patches, n_extra_patches

        return out, z, n_patches_y, n_patches_x

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

    def evaluate_loss(self, outs, cur_idx, label, gating=None):
        '''
        outs: list
        label: Tensor of shape (TBD,)
        '''
        # print('im in evaluate loss now',outs[cur_idx].shape ,label[cur_idx],cur_idx)
        accuracy = torch.zeros(1)
        gating_out = None
        if self.calc_loss and self.opt.loss == 0:
            # Special cases of predicting module below or same module and below ('both')
            if self.predict_module_num == '-1' or self.predict_module_num == 'both':  # gating not implemented here!
                if self.asymmetric_W_pred:
                    raise NotImplementedError("asymmetric W not implemented yet for predicting lower layers!")
                if self.encoder_num == 0:  # normal loss for first module
                    loss, loss_gated, gating_out = self.loss(outs[cur_idx], outs[cur_idx], gating=gating)  # z, c
                else:
                    loss, loss_gated, _ = self.loss(outs[cur_idx - 1], outs[cur_idx])  # z, c
                    if self.predict_module_num == 'both':
                        loss_intralayer, _, _ = self.loss_same_module(outs[cur_idx], outs[cur_idx])
                        loss = 0.5 * (loss + loss_intralayer)

            elif self.predict_module_num == '-1b':
                if self.asymmetric_W_pred:
                    raise NotImplementedError("asymmetric W not implemented yet for predicting lower layers!")
                if self.encoder_num == len(outs) - 1:  # normal loss for last module
                    loss, loss_gated, gating_out = self.loss(outs[cur_idx], outs[cur_idx], gating=gating)  # z, c
                else:
                    loss, loss_gated, _ = self.loss(outs[cur_idx], outs[cur_idx + 1])  # z, c
            # Normal case for prediction within same layer
            else:
                if self.asymmetric_W_pred:  # u = z*W_pred*c -> u = drop_grad(z)*W_pred1*c + z*W_pred2*drop_grad(c)
                    if self.opt.contrast_mode != 'hinge':
                        raise ValueError("asymmetric_W_pred only implemented for hinge contrasting!")

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

        elif self.calc_loss and self.opt.loss == 1:  # supervised loss
            loss, accuracy = self.loss(outs[cur_idx], label)
            loss_gated, gating_out = -1, -1
        else:  # only forward pass for downstream classification
            loss, loss_gated, accuracy, gating_out = None, None, None, None

        return loss, loss_gated, accuracy, gating_out

    def train_step(self, x, reps, t, n_patches_y, n_patches_x, label, idx):
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

        h, z, n_patches_y, n_patches_x = self.forward(
            x, reps, t, n_patches_y, n_patches_x, label
        )
        # out: mean pooled per patch for each z. For example z shape is (198,128,16,16)  the mean pooling is (4,128,7,7)  where 198=4*7*7 = batch_size*img_patch_size*img_patch_size
        if self.opt.feedback_gating:
            if idx == self.opt.model_splits - 1:  # no gating for highest layer
                gating = None
        else:
            gating = None

        if self.asymmetric_W_pred:  # u = z*W_pred*c -> u = drop_grad(z)*W_pred1*c + z*W_pred2*drop_grad(c)
            if self.opt.contrast_mode != 'hinge':
                raise ValueError("asymmetric_W_pred only implemented for hinge contrasting!")

            # the outs we are using are the mean pooled ones  so (4,nb_filters,7,7).
            loss, loss_gated, _ = self.loss(h, h.clone().detach(),
                                            gating=gating)  # z, detach(c)

            loss_mirror, loss_mirror_gated, _ = self.loss_mirror(h.clone().detach(), h,
                                                                 gating=gating)  # detach(z), c

            loss = loss + loss_mirror
            loss_gated = loss_gated + loss_mirror_gated
        else:
            loss, loss_gated, gating_out = self.loss(h, h,
                                                     gating=gating)  # z, c  #outs is a list that contains tensors of shape (4,nb_channels,7,7)

        # print('Loss calculation done...',loss)

        # print('updating weights...')

        loss.backward()  # retain_graph=True

        self.optimizer.step()

        next_layer_input = z.clone().detach()  # full module output # This sets requires_grad to False

        return next_layer_input, h, loss, n_patches_y, n_patches_x


class VGGn(nn.Module):
    '''
    VGG and VGG-like networks.
    The network can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.

    Args:
        vgg_name (str): The name of the network.
        input_dim (int): Feature map height/width for input.
        input_ch (int): Number of feature maps for input.
        num_classes (int): Number of classes (used in local prediction loss).
        feat_mult (float): Multiply number of feature maps with this number.
    '''

    def __init__(self, vgg_name, input_dim, input_ch, num_classes, feat_mult=1):
        super(VGGn, self).__init__()
        self.cfg = cfg[vgg_name]
        self.input_dim = input_dim
        self.input_ch = input_ch
        self.num_classes = num_classes
        self.features, output_dim = self._make_layers(self.cfg, input_ch, input_dim, feat_mult)
        for layer in self.cfg:
            if isinstance(layer, int):
                output_ch = layer


    def parameters(self):
        if not args.backprop:
            return self.classifier.parameters()
        else:
            return super(VGGn, self).parameters()

    def set_learning_rate(self, lr):
        for i, layer in enumerate(self.cfg):
            if isinstance(layer, int):
                self.features[i].set_learning_rate(lr)
        if args.num_layers > 0:
            self.classifier.set_learning_rate(lr)

    def optim_zero_grad(self):
        for i, layer in enumerate(self.cfg):
            if isinstance(layer, int):
                self.features[i].optim_zero_grad()
        if args.num_layers > 0:
            self.classifier.optim_zero_grad()

    def optim_step(self):
        for i, layer in enumerate(self.cfg):
            if isinstance(layer, int):
                self.features[i].optim_step()
        if args.num_layers > 0:
            self.classifier.optim_step()

    def forward(self, x, y, y_onehot):
        loss_total = 0
        for i, layer in enumerate(self.cfg):
            if isinstance(layer, int):
                x, loss = self.features[i](x, y, y_onehot)
                loss_total += loss
            else:
                x = self.features[i](x)

        if args.num_layers > 0:
            x, loss = self.classifier(x, y, y_onehot)
        else:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        loss_total += loss

        return x, loss_total

    def _make_layers(self, cfg, input_ch, input_dim, feat_mult):
        layers = []
        first_layer = True
        scale_cum = 1
        for x in cfg:
            if x == 'M':
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
                x = int(x * feat_mult)
                if first_layer and input_dim > 64:
                    scale_cum = 2
                    layers += [LocalLossBlockConv(input_ch, x, kernel_size=7, stride=2, padding=3,
                                                  num_classes=num_classes,
                                                  dim_out=input_dim // scale_cum,
                                                  first_layer=first_layer)]
                else:
                    layers += [LocalLossBlockConv(input_ch, x, kernel_size=3, stride=1, padding=1,
                                                  num_classes=num_classes,
                                                  dim_out=input_dim // scale_cum,
                                                  first_layer=first_layer)]
                input_ch = x
                first_layer = False

        return nn.Sequential(*layers), input_dim // scale_cum

    def train(self):
        pass
