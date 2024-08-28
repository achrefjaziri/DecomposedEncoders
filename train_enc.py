'''
Main script to train our encoders. This script allows the use of multiple GPUs to train our encoders
'''
import random,os,logging
from datetime import datetime
import numpy as np
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from lib.utils.logging_utils import tsne_visualization
from lib.dataloaders.motion_datagenerator import MoSIGenerator
from lib.dataloaders.get_dataloader import get_dataloader
from lib.models.full_model import FullModel
from lib.arg_parser import arg_parser


def to_one_hot(y, n_dims=None):
    ''' Take integer tensor y with n dims and convert it to 1-hot representation with n+1 dims. '''
    y_tensor = y.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot


def train(args, model, train_loader, gpu, test_loader, writer, current_dir):
    """
    Main function to train the encoders
    :param args: parsed arguments
    :param model: the encoder as nn.Module object
    :param train_loader: training loader
    :param gpu: the index of the current GPU
    :param test_loader: validation data
    :param writer: the tensorboard writer
    :param current_dir: the current working directory.
    :return:
    """


    with torch.autograd.set_detect_anomaly(True):

        for epoch in range(args.start_epoch, args.num_epochs + args.start_epoch):
            train_loader.sampler.set_epoch(epoch)

            loss_epoch = [0 for i in range(model.get_model_splits())]
            loss_updates = [1 for i in range(model.get_model_splits())]
            loss_epoch_vae = [0 for i in range(model.get_model_splits())]
            loss_epoch_rec = [0 for i in range(model.get_model_splits())]
            loss_epoch_kld = [0 for i in range(model.get_model_splits())]
            motion_type = None  # Variable needed for training the dual mouvement module

            with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{args.num_epochs + args.start_epoch}',
                      unit='batch') as pbar:
                for step, (img, label) in enumerate(train_loader):

                    model_input = img.to(gpu)  # shape (4,1,64,64)
                    label = label.to(gpu)
                    if args.train_mode == 'mvment':
                        # mosi_gen = MoSIGenerator(args)
                        model_input = model_input.permute(0, 2, 3, 1)
                        # model_input, label = mosi_gen(model_input)
                        model_input = model_input.permute(0, 4, 1, 2, 3)
                        label = label['move_joint']
                        label = label.to(gpu)
                    if args.train_mode == 'so_mvment':
                        # mvment type is either mvment of the camera(label=0) or mvment in the scene (label=1)
                        motion_type = random.choice([0, 1])
                        if motion_type == 0:
                            # mvment of the camera is created using a single frame from each video (self-motion)
                            mosi_gen = MoSIGenerator(args)
                            # get random from each video
                            random_frame_idx = random.randint(0, 15)
                            model_input = model_input[:, :, random_frame_idx, :, :]

                            model_input = model_input.permute(0, 2, 3, 1)
                            model_input, label = mosi_gen(model_input)
                            model_input = model_input.permute(0, 4, 1, 2, 3)

                            label = label[
                                        'move_joint'] + 3  # velocity labels for self motion start with index 3. /index 0 to 2 are reserved for scene motion velocity
                            label = label.to(gpu)
                            label_one_hot = to_one_hot(label)
                            label_one_hot = label_one_hot.to(gpu)
                        else:
                            # mvment in the scene is from the orginal video, different speeds correspond to different labels
                            # TODO resize input to input_dim
                            label = label.to(gpu)
                            label_one_hot = to_one_hot(label)
                            label_one_hot = label_one_hot.to(gpu)

                    #In the case of supervised training with cross entropy. one hot labels are needed.
                    if args.train_mode == 'CE' or args.train_mode == 'mvment':
                        label_one_hot = to_one_hot(label)
                        label_one_hot = label_one_hot.to(gpu)
                    else:
                        label_one_hot = None

                    loss_dic = model.training_step(model_input, y=label, y_onehot=label_one_hot,
                                                   motion_type=motion_type)
                    loss_values = loss_dic['main_loss']
                    for idx in range(len(loss_values)):
                        if args.train_mode == 'hinge':
                            if args.asymmetric_W_pred:
                                loss_epoch[idx] += (0.5 * loss_values[
                                    idx].item())  # loss is double in that case but gradients are still the same -> print the corresponding values
                            else:
                                loss_epoch[idx] += loss_values[idx].item()
                        else:
                            loss_epoch[idx] += loss_values[idx].item()
                        loss_updates[idx] += 1

                    pbar.set_postfix(
                        **{'loss (batch)': np.mean([x / loss_updates[idx] for idx, x in enumerate(loss_epoch)])})

                    pbar.update(1)

            train_loss = [x / loss_updates[idx] for idx, x in enumerate(loss_epoch)]

            print(
                f'Epoch {epoch + 1}, Train loss mean: {"%.4f" % np.mean(train_loss)},Loss per layer: {train_loss}')
            # print('Logging for epoch', epoch,train_loss)
            logging.info(
                f'Epoch {epoch + 1}, Train loss mean: {"%.4f" % np.mean(train_loss)},Loss per layer: {train_loss}')
            for idx, val in enumerate(train_loss):
                writer.add_scalar(f'train/Loss_{idx}', val, epoch)
            # Log and save img graphs

            if (epoch + 1) % 25 == 0:  # we save the model every 25 epochs
                model.save_model(current_saving_dir=current_dir, epoch=epoch)

            if args.train_mode != 'mvment' and args.train_mode != 'so_mvment' and args.dataset != 'UCF101':
                #due to a unfixed bug the validation step is done only on a selected datasets (GTSRB and STL10)
                if (epoch + 1) % 5 == 0:
                    # do some validation :)
                    features = []
                    corresponding_labels = []
                    for step, (img, label) in enumerate(test_loader):
                        model_input = img.to(gpu)  # shape (4,1,64,64)
                        label = label.to(gpu)

                        # print('CURRENT DEVICES',model_input.get_device(),label.get_device())
                        output_dic = model(model_input)
                        img_encoding = output_dic['img_encoding']
                        current_outputs = img_encoding.detach().cpu().numpy()
                        # features = np.concatenate((features, current_outputs))
                        features.append(current_outputs)
                        corresponding_labels.append(label.cpu().numpy())

                    # Reshape patches before stiching them up
                    features = np.asarray(features[:-1])
                    features = features.mean(axis=(3, 4))
                    features = features.reshape(-1, features.shape[-1])
                    corresponding_labels = np.ravel(np.asarray(corresponding_labels[:-1]))
                    if args.dataset != 'CODEBRIM' and args.dataset != 'UCF101':
                        # Cluster visualization TSNE
                        tsne_visualization(writer, features, corresponding_labels, epoch, 6)

    dist.destroy_process_group()
    writer.close()


def main(gpu, args, current_dir):
    # setup the process groups
    print('Setup...')
    ############################################################
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )
    torch.cuda.set_device(gpu)

    ############################################################
    # load model
    print('Loading Model...')
    model = FullModel(args, rank=gpu)
    if args.resume:
        print("resuming training...")
        model.load_models(epoch=args.res_epoch)
        args.start_epoch = args.res_epoch

    print('Creating Logger...')

    logging.basicConfig(handlers=[logging.FileHandler(filename=os.path.join(current_dir, f'run_history_{gpu}.log'),
                                                      # args.model_path
                                                      encoding='utf-8', mode='a+')],
                        format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%F %A %T",
                        level=logging.INFO)
    logging.info(f''' Training parameters:
                    Epochs:          {args.num_epochs}
                    Batch size:      {args.batch_size}
                    Learning rate:   {args.lr}
                    Weight decay:   {args.weight_decay}
                    Dataset:   {args.dataset}
                    Loss:  {args.train_mode}
                    Number of negative samples: {args.negative_samples}
                    Sample negs locally: {args.sample_negs_locally_same_everywhere}
                    Either pos or neg update: {args.either_pos_or_neg_update}                  
                    Asymmetric W_pred: {args.asymmetric_W_pred}
                    Hinge Mode: {args.input_mode}                  
                    Hebbian Extension: {args.hebb_extension}
                    Resize Input: {args.resize_input}
                    Input size: {args.input_dim}
                    Resize size: {args.resize_size}
                ''')
    if args.resume:
        print("Resuming...")
        logging.info(f'Model Loaded from {args.model_path} Epoch: {args.res_epoch}')

    print('creating Tensorboard')
    model_id = os.path.basename(current_dir)
    tensorboard_dir = os.path.join(args.save_dir, 'tensorboard',
                                   'runs', args.dataset, args.train_mode,
                                   args.vgg_name + '-' + model_id)  # datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    writer = SummaryWriter(log_dir=tensorboard_dir,
                           comment=f'Model Id {model_id},Architecture {args.vgg_name}, Training Set {args.dataset}, Train mode {args.train_mode}')

    print('Loading Data...')
    train_loader, _, test_loader, _ = get_dataloader(
        args,
        gpu, args.world_size)

    print('Start training...')
    try:
        # Train the model
        train(args, model, train_loader, gpu, test_loader, writer, current_dir)

    except KeyboardInterrupt:
        print("Training got interrupted, saving log-files now.")


if __name__ == "__main__":

    #os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = arg_parser.parse_args()

    #########################################################
    if args.device != 'cpu':
        args.world_size = args.gpus * args.nodes
        print(args.world_size)

    os.environ['MASTER_ADDR'] = '127.0.0.1'  #
    os.environ['MASTER_PORT'] = args.port  # '6009'

    if args.resume:
        model_id = os.path.basename(args.model_path)
    else:
        model_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    current_dir = os.path.join(args.save_dir, args.vgg_name, args.dataset, model_id)
    if not os.path.isdir(current_dir):
        os.makedirs(current_dir)

    mp.spawn(
        main,
        args=(args, current_dir,),
        nprocs=args.gpus
    )  #########################################################

