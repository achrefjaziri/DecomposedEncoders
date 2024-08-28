import torch
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
import numpy as np
from numpy.random import choice
from IPython import embed
import os


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

#TODO understand how the patches are generated in a notebook
#TODO write down the code steps
#TODO decide whether to add GRU layer as a final layer before loss calcualtion and change how labels are generated.
class DorsalLoss(nn.Module):
    def __init__(self, opt=None, in_channels=128, prediction_steps=3,
                 save_vars=False):  # in_channels: z, out_channels: c
        super().__init__()

        # --negative_sampleS  1
        # --sample_negs_locally
        # --sample_negs_locally_same_everywhere
        # --either_pos_or_neg_update
        self.opt = opt
        self.k_predictions = prediction_steps  # 5

        #in_channels, out_channels, 1, bias=False
        self.num_layers=6
        self.h_out = int(in_channels//2)


        self.rnn = nn.RNN(input_size=in_channels, hidden_size=self.h_out, num_layers=self.num_layers,batch_first=True)

        self.loss_criterion = torch.nn.CrossEntropyLoss()

        self.projection_layer = nn.Linear(in_features=self.k_predictions*self.h_out, out_features=128)
        self.activation = nn.ReLU()
        self.classification_layer = nn.Linear(in_features=128,out_features=8)


        #if self.opt.weight_init:
        #    self.initialize()

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

    def forward(self, reps, labels):
        # gating should be either None or (nested) list of gating values for each prediction step (k_predictions)
        # z: b, channels, n_patches_y, n_patches_x
        batch_size = reps.shape[0]


        total_loss, total_loss_gated = 0, 0


        #h0 = torch.zeros(self.num_layers, batch_size, self.h_out).to(reps.get_device())
        output, hn = self.rnn(reps)
        hn = hn.permute(1,0,2)
        hn = hn.reshape(hn.size(0), -1)
        output = output.reshape(output.size(0), -1)
        #print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",output.shape,self.k_predictions,self.h_out)
        predictions = self.projection_layer(output).float()
        predictions = self.activation(predictions)
        predictions = self.classification_layer(predictions)

        loss = self.loss_criterion(predictions, labels.to(torch.int64)) #.type(torch.LongTensor)



        return loss





if __name__ == "__main__":
    inputs = torch.randn(4, 5, 128)
    labels = torch.empty(4, dtype=torch.long).random_(8)
    loss_function = DorsalLoss()
    loss = loss_function(inputs,labels)
    print(loss)



