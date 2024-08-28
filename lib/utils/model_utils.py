import torch
import torch.nn as nn
import numpy as np
import random
from skimage import feature

def count_parameters(model):
    ''' Count number of parameters in model influenced by global loss. '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_one_hot(y, n_dims=None):
    ''' Take integer tensor y with n dims and convert it to 1-hot representation with n+1 dims. '''
    y_tensor = y.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot


def similarity_matrix(x,args):
    ''' Calculate adjusted cosine similarity matrix of size x.size(0) x x.size(0). '''
    if x.dim() == 4:
        if not args.no_similarity_std and x.size(1) > 3 and x.size(2) > 1:
            z = x.view(x.size(0), x.size(1), -1)
            x = z.std(dim=2)
        else:
            x = x.view(x.size(0), -1)
    xc = x - x.mean(dim=1).unsqueeze(1)
    xn = xc / (1e-8 + torch.sqrt(torch.sum(xc ** 2, dim=1))).unsqueeze(1)
    R = xn.matmul(xn.transpose(1, 0)).clamp(-1, 1)
    return R


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

def reshape_to_patches(x,patch_size,overlap_factor):
    # x enters as  (batch_size,channels,height,width) for example (4,1,64,64) -
    # x gets reshaped to (batch_Size* n_patches_y * n_patches_x, nb_channels,patch_size,patch_size) - for example (196,1,16,16) for patch_size=16 and overlap_factor=5
    x = (  # b, c, y, x
        x.unfold(2, patch_size,
                 patch_size // overlap_factor)  # b, c, n_patches_y, x, patch_size
            .unfold(3, patch_size,
                    patch_size // overlap_factor)  # b, c, n_patches_y, n_patches_x, patch_size, patch_size
            .permute(0, 2, 3, 1, 4, 5)  # b, n_patches_y, n_patches_x, c, patch_size, patch_size
    )
    # x gets reshaped to (batch_size,7,7,1,16,16) #where patch_size is 16 and overlap is 2
    n_patches_y = x.shape[1]
    n_patches_x = x.shape[2]
    x = x.reshape(
        x.shape[0] * x.shape[1] * x.shape[2], x.shape[3], x.shape[4], x.shape[5]
    )  # b * n_patches_y * n_patches_x, c, patch_size, patch_size
    return x,n_patches_y,n_patches_x

def prepare_labels(x,seq_length):
    training_batch = []
    training_labels = []
    dir_x = 0
    dir_y =0
    cur_device = x.get_device()
    for i in range(x.shape[0]):
        starting_pos_x = random.randint(seq_length, x.shape[1] - seq_length)
        starting_pos_y = random.randint(seq_length, x.shape[2] - seq_length)
        while dir_x == 0 and dir_y == 0:
            dir_x = random.randint(-1, 1)
            dir_y = random.randint(-1, 1)

        if dir_y == -1 and dir_x == 0:
            # the permute function is used to ensure consistent outputs with the other conditions that use the np.diagonal
            imgs = x[:, starting_pos_x, starting_pos_y + dir_y * seq_length:starting_pos_y, :, :, :].permute(0, 2, 3, 4,
                                                                                                             1).flip(
                dims=[4]).detach().cpu().numpy()
            label = 0

        elif dir_y == -1 and dir_x == 1:
            placeholder = x[:, starting_pos_x:starting_pos_x + dir_x * seq_length,
                          starting_pos_y + dir_y * seq_length:starting_pos_y, :, :, :]
            placeholder = np.fliplr(placeholder.detach().cpu().numpy()).copy()
            imgs = np.diagonal(placeholder, axis1=1, axis2=2)
            imgs = torch.Tensor(imgs.copy()).flip(dims=[4]).detach().cpu().numpy()
            label = 1


        elif dir_y == 0 and dir_x == 1:
            imgs = x[:, starting_pos_x:starting_pos_x + dir_x * seq_length, starting_pos_y, :, :, :].permute(0, 2, 3, 4,
                                                                                                             1).detach().cpu().numpy()
            label = 2

        elif dir_y == 1 and dir_x == 1:
            placeholder = x[:, starting_pos_x:starting_pos_x + dir_x * seq_length,
                          starting_pos_y:starting_pos_y + dir_y * seq_length, :, :, :]
            imgs = np.diagonal(placeholder.detach().cpu().numpy(), axis1=1, axis2=2)
            # imgs =torch.Tensor(imgs)
            label = 3

        elif dir_y == 1 and dir_x == 0:
            imgs = x[:, starting_pos_x, starting_pos_y:starting_pos_y + dir_y * seq_length, :, :, :]
            imgs = imgs.permute(0, 2, 3, 4, 1).detach().cpu().numpy()
            label = 4


        elif dir_y == 1 and dir_x == -1:
            placeholder = x[:, starting_pos_x + dir_x * seq_length:starting_pos_x,
                          starting_pos_y:starting_pos_y + dir_y * seq_length, :, :, :]
            placeholder = np.fliplr(placeholder.detach().cpu().numpy()).copy()
            imgs = np.diagonal(placeholder, axis1=1, axis2=2)
            # imgs =torch.Tensor(imgs)
            label = 5

        elif dir_y == 0 and dir_x == -1:
            imgs = x[:, starting_pos_x + dir_x * seq_length:starting_pos_x, starting_pos_y, :, :, :].permute(0, 2, 3, 4,
                                                                                                             1).flip(
                dims=[4]).detach().cpu().numpy()
            label = 6

        elif dir_y == -1 and dir_x == -1:
            placeholder = x[:, starting_pos_x + dir_x * seq_length:starting_pos_x,
                          starting_pos_y + dir_y * seq_length:starting_pos_y, :, :, :]
            imgs = np.diagonal(placeholder.detach().cpu().numpy(), axis1=1, axis2=2)
            imgs = torch.Tensor(imgs.copy()).flip(dims=[4]).detach().cpu().numpy()
            label = 7
        else:
            print("not implemented", dir_x, dir_y)
        training_batch.append(imgs[i])
        training_labels.append(label)

    training_batch = torch.Tensor(np.asarray(training_batch).transpose(0,1,4, 2, 3))
    training_batch = training_batch.reshape(training_batch.shape[0] , training_batch.shape[1], training_batch.shape[2],
                                            training_batch.shape[3], training_batch.shape[4])
    training_batch = torch.Tensor(training_batch).to(cur_device)
    training_labels = np.asarray(training_labels)
    training_labels = torch.Tensor(training_labels).to(cur_device)
    return training_batch,training_labels

def LBP_estimation(image,num_points,radius):
    lbp = feature.local_binary_pattern(image, num_points,radius, method="uniform")
    return lbp


class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius
	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")

