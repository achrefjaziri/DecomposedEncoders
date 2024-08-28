
import torch
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
import numpy as np
from numpy.random import choice
from IPython import embed
import os



def MLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

# loss fn

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

def genOrthgonal(dim):
    a = torch.zeros((dim, dim)).normal_(0, 1)
    q, r = torch.qr(a)
    d = torch.diag(r, 0).sign()
    diag_size = d.size(0)
    d_exp = d.view(1, diag_size).expand(diag_size, diag_size)
    q.mul_(d_exp)
    return q


def makeDeltaOrthogonal(weights, gain):
    rows = weights.size(0)
    cols = weights.size(1)
    if rows > cols:
        print("In_filters should not be greater than out_filters.")
    weights.data.fill_(0)
    dim = max(rows, cols)
    q = genOrthgonal(dim)
    mid1 = weights.size(2) // 2
    mid2 = weights.size(3) // 2
    with torch.no_grad():
        weights[:, :, mid1, mid2] = q[: weights.size(0), : weights.size(1)]
        weights.mul_(gain)

class ContrastiveLoss(nn.Module):
    def __init__(self, opt, in_channels, out_channels, prediction_steps, save_vars=False): # in_channels: z, out_channels: c
        super().__init__()

        #--negative_sampleS  1
        #--sample_negs_locally
        #--sample_negs_locally_same_everywhere
        #--either_pos_or_neg_update
        self.opt = opt
        self.negative_samples = int(self.opt.negative_samples) #16 default
        self.k_predictions = prediction_steps #5
        self.contrast_mode = self.opt.train_mode #decides whether constrasting with neg. examples is done at once 'mutliclass' " "or one at a time with (and then averaged) with CE 'binary', BCE 'logistic' or 'hinge' loss
        self.average_feedback_gating = self.opt.gating_av_over_preds #Boolean: average feedback gating (--feedback_gating) from higher layers over different prediction steps ('k')
        self.detach_c = self.opt.detach_c #"Boolean whether the gradient of the context c should be dropped (detached)
        self.current_rep_as_negative = self.opt.current_rep_as_negative ##Use the current feature vector ('context' at time t as opposed to predicted time step t+k) itself as/for sampling the negative sample
        self.sample_negs_locally = self.opt.sample_negs_locally  #"Sample neg. samples from batch but within same location in image, i.e. no shuffling across locations"
        self.sample_negs_locally_same_everywhere = self.opt.sample_negs_locally_same_everywhere #Extension of --sample_negs_locally_same_everywhere (must be True). No shuffling across locations and same sample (from batch) for all locations. I.e. negative sample is simply a new input without any scrambling
        self.either_pos_or_neg_update = self.opt.either_pos_or_neg_update
        self.which_update = 'both'
        self.save_vars = save_vars
        
        if self.current_rep_as_negative:
            self.negative_samples = 1


        self.out_channels=out_channels
        self.W_k = nn.ModuleList(
            nn.Conv2d(in_channels, out_channels, 1, bias=False) # in_channels: z, out_channels: c
            for _ in range(self.k_predictions)
        )

        #self.projection_layer = MLP(out_channels,projection_size=128) #nn.Linear(in_features= out_channels, out_features=64)

        #self.TARGETRATE =float(1 / out_channels)   # The target winning rate of each cell *must* be K/N to allow for equilibrium

        #self.MUTHRES = 10.0  # Threshold adaptation rate
        #self.thres_hebb = 5.0# Thresholds (i.e. adaptive biases to ensure a roughly equal firing rate)
        if self.opt.freeze_W_pred: # freeze prediction weights W_k  #"Boolean whether the k prediction weights W_pred (W_k in ContrastiveLoss) are frozen (require_grad=False) Default:
            if self.opt.unfreeze_last_W_pred:
                print('im freezing last W_pred')
                params_to_freeze = self.W_k[:-1].parameters()
            else:
                print('im freezing all parameters')
                params_to_freeze = self.W_k.parameters()
            for p in params_to_freeze:
                p.requires_grad = False
        
        if self.contrast_mode == 'multiclass' or self.contrast_mode == 'binary':
            self.contrast_loss = ExpNLLLoss()
        elif self.contrast_mode == 'logistic':
            self.contrast_loss = MyBCEWithLogitsLoss()
        elif self.contrast_mode == 'hinge' or self.contrast_mode=='var_hinge'  or self.contrast_mode=='CPC':
            self.contrast_loss = CLAPPHingeLoss()

        if self.opt.weight_init:
            self.initialize()


    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                if m in self.W_k:
                    makeDeltaOrthogonal(
                        m.weight,
                        nn.init.calculate_gain(
                            "Sigmoid"
                        ),
                    )


    def forward(self, z, c, skip_step=1, gating=None):
        # gating should be either None or (nested) list of gating values for each prediction step (k_predictions)
        # z: b, channels, n_patches_y, n_patches_x
        if self.detach_c:
            c = c.clone().detach() # drop gradient of context
        batch_size = z.shape[0]

        # If self.either_pos_or_neg_update is True, select whether pos or neg update (or both) is done, independently for every sample in batch 
        # p = [0.5,0.5,0.] for equal sampling, p = [0.,0.,1.] implements normal HingeLossCPC
        if self.either_pos_or_neg_update:
            self.which_update = choice(['pos','neg','both'], size = batch_size, replace=True, p = [0.5,0.5,0.])
        
        total_loss, total_loss_gated = 0, 0
        gating_out = []
        if gating is not None and self.average_feedback_gating: # average gating over k predictions
            g_pos = sum([g[0] for g in gating]) / self.k_predictions
            g_neg = sum([g[1] for g in gating]) / self.k_predictions
            gating = [g_pos, g_neg] # 2 x b each

        if self.opt.device != "cpu":
            cur_device = z.get_device()
        else:
            cur_device = self.opt.device
        

        # Loop over different prediction intervals, For each element in c, contrast with elements below
        for k in range(1, self.k_predictions + 1):    
            ### compute log f(c_t, x_{t+k}) = z^T_{t+k} W_k c_t
            # compute z^T_{t+k} W_k:
            #print('Relevant Infos in forward of Contrastive Loss W_k:',self.W_k[k - 1])
            #print('Relevant Infos in forward of Contrastive Loss Z:',z[:, :, (k + skip_step) :, :].shape)


            #Skip steps is fixed but we are going to the next patch so we have (4,7,batch,channels) instead of (5,7,batch,channels)
            #print(self.W_k[k-1])
            #print("valllll",k+skip_step)
            #print("shapessssssssssssssssss",z[:, :, (k + skip_step) :, :].shape)

            ztwk = (
                self.W_k[k - 1]
                .forward(z[:, :, (k + skip_step) :, :])# Bx, C , H , W
                #.clone().detach()
                .permute(2, 3, 0, 1)# H, W, Bx, C
                .contiguous())  # y, x, b, c example (5,7,Batch_size,channels)
            #ztwk = torch.rand((7-(k+skip_step),7,32,z.shape[1]), requires_grad=True).to(cur_device)
            #print('ztwk',ztwk.shape,ztwk_new.shape)
            #sape (5,7,64,128) #


            # Creation of neg. examples
            ztwk_shuf, rand_index = self.create_negative_samples(k, skip_step, z, ztwk, cur_device) # y, x, b, c, n
            
            #### Compute  x_W1 . c_t:
            # context: b, c, H, W = x
            context = (
                c[:, :, : -(k + skip_step), :].permute(2, 3, 0, 1).unsqueeze(-2)
            )  # y (reduced H), x, b, 1, c
            #ztwk_copy = ztwk.clone()
            log_fk_main = torch.matmul(context, ztwk.unsqueeze(-1)).squeeze(
                -2
            )  # y, x, b, 1

            log_fk_shuf = torch.matmul(context, ztwk_shuf).squeeze(-2)  # y, x, b, n

            #Use this  as mean to get variance of activity between examples
            #reshape_for_var =z.reshape(z.shape[0],z.shape[1],-1).permute(0,2,1)
            #mean = torch.mean(reshape_for_var,dim=1)
            #mse= -torch.log(torch.var(mean))

            #context and z
            #instead of matmul do MSE,
            #print(c[:, :, : -(k + skip_step), :].shape,z[:, :, (k + skip_step) :, :].shape)
            #print(context.reshape(-1,context.shape[-1]).shape,z.shape) #.reshape(-1,z.shape[-1])
            #mse = torch.nn.functional.mse_loss(c[:, :, : -(k + skip_step), :], z[:, :, (k + skip_step) :, :])

            c_for_linear = c[:, :, : -(k + skip_step), :].permute(0,2,3,1).reshape(-1,self.out_channels)
            z_for_linear = z[:, :, (k + skip_step) :, :].permute(0,2,3,1).reshape(-1,self.out_channels)

            #print("New shapes",c_for_linear.shape,z_for_linear.shape)
            #projections_c = self.projection_layer(c_for_linear)
            #projections_z =self.projection_layer(z_for_linear)
            #print("projections",projections_c.shape,projections_z.shape)
            #mse = torch.nn.functional.mse_loss(projections_c,projections_z)
            #loss_fn_val = torch.mean(loss_fn(projections_z,projections_c))
            #print(loss_fn_val)
            #print('contrast mode',self.contrast_mode)
            if self.contrast_mode=='multiclass':
                log_fk, target = self.multiclass_contrasting(log_fk_main, log_fk_shuf,
                                                            batch_size, cur_device) # b, 1+n, y, x; b, y, x
            elif self.contrast_mode=='binary':
                log_fk, target = self.binary_contrasting(log_fk_main, log_fk_shuf,
                                                            batch_size, cur_device) # b, 1+1, n, y, x; b, n, y, x
            elif self.contrast_mode=='logistic' or self.contrast_mode=='hinge' or  self.contrast_mode=='var_hinge' or self.contrast_mode=='CPC':
                if self.opt.no_pred: #  Wpred * c is set to 1 (no prediction). i.e. fourth factor omitted in learning rule. In this case, the score function is equal to the sum of activations
                    log_fk_main = ztwk.sum(dim=-1).unsqueeze(-1) + context.sum(dim=-1) # y, x, b, 1

                    log_fk_shuf = ztwk_shuf.sum(dim=-2) + context.sum(dim=-1).repeat(1, 1, 1, self.negative_samples) # y, x, b, n



                log_fk, target = self.logistic_contrasting(log_fk_main, log_fk_shuf,
                                                            batch_size, cur_device) # b, 1+1, n, y, x (both)

            if gating is None:
                gate = None
            else:
                if self.average_feedback_gating:
                    gate = gating # already average over k
                else:
                    gate = gating[k-1] # k-1 because k is in range(1,k_predictions+1)
            #print('log fk',log_fk.shape)

            #Loss k gates is the loss just multiplied by gate. gating_out_k is a list with 4 elements each *2 (pos and neg examples) .
            loss_k, loss_k_gated, gating_out_k = self.contrast_loss(self.opt, k, input=log_fk, target=target, gating=gate, which_update=self.which_update, save_vars=self.save_vars)
            total_loss += loss_k

            if loss_k_gated is not None:
                total_loss_gated += loss_k_gated

            gating_out.append(gating_out_k)

            if self.save_vars:
                z_s = z[:, :, (k + skip_step) :, :].permute(2, 3, 0, 1).clone() # y (red.), x, b, c
                torch.save((context.clone(), z_s.clone(), z_s[:, :, rand_index, :].clone(), rand_index), os.path.join(self.opt.model_path, 'saved_c_and_z_layer_'+str(self.opt.save_vars_for_update_calc)+'_k_'+str(k)))

        if self.save_vars:
            if type(self.which_update) == str:
                which_update_save = self.which_update 
            else:
                which_update_save = self.which_update.copy()
            torch.save(which_update_save, os.path.join(self.opt.model_path, 'saved_which_update_layer_'+str(self.opt.save_vars_for_update_calc)))

        total_loss /= self.k_predictions
        total_loss_gated /= self.k_predictions

        #thres.append(torch.zeros((1, N[numl], 1, 1), requires_grad=False).to(
        #    device))  # Thresholds (i.e. adaptive biases to ensure a roughly equal firing rate)
        """
                with torch.no_grad():
            realy = (z - self.thres_hebb)
            tk = torch.topk(realy.data, 1, dim=1, largest=True)[0]
            realy.data[realy.data < tk.data[:, -1, :, :][:, None, :, :]] = 0
            realy.data = (realy.data > 0).float()




        yforgrad = z - 1/2 * torch.sum(self.W_k[k - 1].weight * self.W_k[k - 1].weight, dim=(1,2,3))[None,:, None, None] # Oja's rule is missing the realy.data part , dw ~= y(x-yw)
        #print(yforgrad.shape) #Tensor of shape (64,128,7,7)
        with torch.no_grad():
            self.thres_hebb += self.MUTHRES * (
                    torch.mean((realy.data > 0).float(), dim=(0, 2, 3))[None, :, None, None] - self.TARGETRATE)

        yforgrad.data = realy.data  # We force the value of yforgrad to be the "correct" y
        loss_hebb = torch.sum(-1 / 2 * yforgrad * yforgrad)
        
        """
        #print("hebb loss",loss_hebb)
        tt = total_loss  #+ mse#+ var_mean + loss_fn_val
        return tt, total_loss_gated, gating_out#,mse,total_loss


    def multiclass_contrasting(self, log_fk_main, log_fk_shuf,
                                batch_size, cur_device):
        """ contrasting all the negative examples at the same time via multi-class classification"""
        log_fk = torch.cat((log_fk_main, log_fk_shuf), 3)  # y, x, b, 1+n
        log_fk = log_fk.permute(2, 3, 0, 1)  # b, 1+n, y, x  This is the shape expected by nll_loss

        log_fk = torch.softmax(log_fk, dim=1)

        target = torch.zeros(
            (batch_size, log_fk.shape[-2], log_fk.shape[-1]),
            dtype=torch.long,
            device=cur_device,
        )  # b, y, x
        return log_fk, target # b, 1+n, y, x; b, y, x

    def _get_log_fk(self, log_fk_main, log_fk_shuf):

        log_fk_main = log_fk_main.repeat(1, 1, 1, self.negative_samples) # y, x, b, n
        log_fk = torch.cat((log_fk_main.unsqueeze(-1), log_fk_shuf.unsqueeze(-1)), 4) # y, x, b, n, 1+1
        log_fk = log_fk.permute(2, 4, 3, 0, 1) # b, 1+1, n, y, x  This is the shape expected by nll_loss
        return log_fk

    def binary_contrasting(self, log_fk_main, log_fk_shuf,
                            batch_size, cur_device):
        """ contrasting all the negative examples independently and later average over losses"""
        log_fk = self._get_log_fk(log_fk_main, log_fk_shuf) # b, 1+1, n, y, x

        log_fk = torch.softmax(log_fk, dim=1)
        target = torch.zeros(
            (batch_size, self.negative_samples, log_fk.shape[-2], log_fk.shape[-1]),
            dtype=torch.long,
            device=cur_device,
        )  # b, n, y, x
        return log_fk, target # b, 1+1, n, y, x; b, n, y, x

    def logistic_contrasting(self, log_fk_main, log_fk_shuf,
                            batch_size, cur_device):
        """ contrasting by doing binary logistic regression on pos. and neg. ex. separately and later average over losses"""
        log_fk = self._get_log_fk(log_fk_main, log_fk_shuf) # b, 1+1, n, y, x

        zeros = torch.zeros(
            (batch_size, self.negative_samples, log_fk.shape[-2], log_fk.shape[-1]),
            dtype=torch.float32,
            device=cur_device,
        )  # b, n, y, x
        ones = torch.ones(
            (batch_size, self.negative_samples, log_fk.shape[-2], log_fk.shape[-1]),
            dtype=torch.float32,
            device=cur_device,
        )  # b, n, y, x
        target = torch.cat((ones.unsqueeze(1), zeros.unsqueeze(1)), 1) # b, 1+1, n, y, x
        return log_fk, target # b, 1+1, n, y, x (both)

    def sample_negative_samples(self, ztwk, cur_device):
        ztwk_shuf = ztwk.view(
            ztwk.shape[0] * ztwk.shape[1] * ztwk.shape[2], ztwk.shape[3]
        )  # y * x * b, c
        rand_index = torch.randint(
            ztwk_shuf.shape[0],  # upper limit: y * x * b
            (ztwk_shuf.shape[0] * self.negative_samples, 1), # shape: y * x * b * n, 1
            dtype=torch.long,
            device=cur_device,
        )
        # replicate neg. sample indices for all channels
        rand_index = rand_index.repeat(1, ztwk_shuf.shape[1]) # y * x * b * n, c

        ztwk_shuf = torch.gather(
            ztwk_shuf, dim=0, index=rand_index, out=None
        )  # y * x * b * n, c

        ztwk_shuf = ztwk_shuf.view(
            ztwk.shape[0],
            ztwk.shape[1],
            ztwk.shape[2],
            self.negative_samples,
            ztwk.shape[3],
        ).permute(
            0, 1, 2, 4, 3
        )  # y, x, b, c, n

        return ztwk_shuf, rand_index
    
    def sample_negative_samples_locally(self, ztwk, cur_device, same_sampling_everywhere=False):
        # ztwk: y, x, b, c
        # same sampling (same sample from batch) for all locations
        if same_sampling_everywhere:
            rand_index = torch.randint(ztwk.shape[2], # upper limit: b
                (ztwk.shape[2],), # shape: b, assumes n=1 neg. samples, 
                dtype=torch.long,
                device=cur_device,
            ) 
            ztwk_shuf = ztwk[:, :, rand_index, :] # y, x, b, c
        # or different sampling (different sample from batch) for different locations:
        else:
            rand_index = torch.randint(ztwk.shape[2], # upper limit: b
                (ztwk.shape[0], ztwk.shape[1], ztwk.shape[2]), # shape: y, x, b, assumes n=1 neg. samples, 
                dtype=torch.long,
                device=cur_device,
            )
            # replicate neg. sample indices for all channels
            rand_index = rand_index.repeat(ztwk.shape[-1], 1, 1, 1).permute(1, 2, 3, 0) # y, x, b, c
            ztwk_shuf = torch.gather(
                ztwk, dim=2, index=rand_index, out=None
            )  # y, x, b, c    

        ztwk_shuf = ztwk_shuf.unsqueeze(-1) # y, x, b, c, n=1
        return ztwk_shuf, rand_index

    # Creation of neg. examples
    def create_negative_samples(self, k, skip_step, z, ztwk, cur_device):   
        if self.current_rep_as_negative: # (unsuccesful) idea of using averaged activity over batch ("memory trace")
            ztwk_context = (
                self.W_k[k - 1]
                .forward(z[:, :, : -(k + skip_step), :])  # Bx, C , H , W
                .permute(2, 3, 0, 1)  # H, W, Bx, C
                .contiguous()
            )  # y, x, b, c (number of negative examples is set to n=1 in that case)
            ztwk_shuf, rand_index = self.sample_negative_samples_locally(ztwk_context, cur_device, same_sampling_everywhere=self.sample_negs_locally_same_everywhere)
        else:
            if self.sample_negs_locally:
                ztwk_shuf, rand_index = self.sample_negative_samples_locally(ztwk, cur_device, same_sampling_everywhere=self.sample_negs_locally_same_everywhere)
            else:
                ztwk_shuf, rand_index = self.sample_negative_samples(ztwk, cur_device)

        return ztwk_shuf, rand_index

##############################################################################################################
# Functions that implement actual contrasting, CPC (ExpNLLLoss), Logistic (MyBCEWithLogitsLoss) or CLAPP (CLAPPHingeLoss)

class ExpNLLLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(ExpNLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, opt, k, input, target, gating=None, which_update='both', save_vars=False):
        if which_update != 'both':
            raise ValueError("which_update must be both for ExpNLLLoss")
        x = torch.log(input + 1e-11)
        loss = F.nll_loss(x, target, weight=self.weight, ignore_index=self.ignore_index,
                          reduction=self.reduction)
        return loss, None, None


class MyBCEWithLogitsLoss(_WeightedLoss):
    def __init__(self):
        super(MyBCEWithLogitsLoss, self).__init__()
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, opt, k, input, target, gating=None, which_update='both', save_vars=False):
        if which_update != 'both':
            raise ValueError("which_update must be both for BCEWithLogitsLoss")
        loss = self.loss_func(input, target)
        return loss, None, None

class CLAPPHingeLoss(_WeightedLoss):
    def __init__(self):
        super(CLAPPHingeLoss, self).__init__()

    def forward(self, opt, k, input, target, gating=None, which_update='both', save_vars=False): #  b, 1+1, n, y, x (both)
        # Take care: pos sample appears N times for N neg. examples
        # gating should be 2 1-dim vectors (for pos and neg samples) with length b containing weights/gating values for each image in batch

        z = input
        def _normalise_gating(g):
            g_mean = g.mean()
            g = g - g_mean
            g = g / g_mean
            g = torch.sigmoid(3 * g)
            g /= g.shape[0] # this normalisation makes the sum in matmul in the gated loss an (unnormed) average
            return g

        def _add_losses(loss_pos, loss_neg, which_update, cur_device):
            l = 0  
            if type(which_update) == str:
                if which_update == 'both': # default case
                    l = loss_pos.mean() + loss_neg.mean()
            else:
                for loss, c in zip([loss_pos, loss_neg], ['pos', 'neg',]):
                    ind = (which_update == c) | (which_update == 'both')
                    if sum(ind) > 0: # exclude empty sets which lead no NaN in loss
                        l += torch.masked_select(loss, torch.tensor(ind.tolist()).to(cur_device)).mean()
            return l

        if opt.device != "cpu":
            cur_device = z.get_device()
        else:
            cur_device = opt.device

        scores_pos = input[:,0,:,:,:] # b, n, y, x
        scores_neg = input[:,1,:,:,:] # b, n, y, x

        ones = torch.ones(size=scores_pos.shape, dtype=torch.float32, device=cur_device)
        zeros = torch.zeros(size=scores_neg.shape, dtype=torch.float32, device=cur_device)

        if opt.no_gamma: # gamma (factor which sets the opposite sign of the update for pos and neg samples and sets gating) is set to 1. i.e. third factor omitted in learning rule
            # loss per pair : shape  for example (4,1,5,7) where 4 is the number of batches. 5*7 is the number of patches to contrasts
            losses_pos = ones - scores_pos
            losses_neg = ones - scores_neg
        else:
            losses_pos = torch.where(scores_pos < ones, ones - scores_pos, zeros) # b, n, y, x
            losses_neg = torch.where(scores_neg > - ones, ones + scores_neg, zeros) # b, n, y, x    
            
        if save_vars:
            torch.save((losses_pos, losses_neg), os.path.join(opt.model_path, 'saved_losses_layer_'+str(opt.save_vars_for_update_calc)+'_k_'+str(k)))
        
        # if gating values are given, take weighted sum before averaging over remaining dimensions
        if gating == None:
            loss_gated = None
        else:
            losses_pos_gated = torch.matmul(losses_pos.permute(1,2,3,0), gating[0])
            losses_neg_gated = torch.matmul(losses_neg.permute(1,2,3,0), gating[1])
            loss_gated = _add_losses(losses_pos_gated, losses_neg_gated, which_update, cur_device)

        losses_pos_per_sample = losses_pos.mean(dim=(-1,-2,-3))
        losses_neg_per_sample = losses_neg.mean(dim=(-1,-2,-3))

        # detach gating such that no gradient of original loss is back-propagated!,
        # clone is important so that later normalisation of gating does not influence loss
        gating_pos = losses_pos_per_sample.clone().detach() 
        gating_neg = losses_neg_per_sample.clone().detach()

        gating_pos = _normalise_gating(gating_pos)
        gating_neg = _normalise_gating(gating_neg)

        gating_out = [gating_pos, gating_neg] 

        loss = _add_losses(losses_pos_per_sample, losses_neg_per_sample, which_update, cur_device) # average over remaining batch dimension

        return loss, loss_gated, gating_out
