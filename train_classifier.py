"""
    Script to train and evaluate a linear classifier for a single encoder model.
"""
import os, sys, time
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from lib.utils.misc_functions import to_one_hot, accuracy_classification, save_eval_history
from lib.dataloaders.endless_runner_dataloader import transform_batch
from lib.dataloaders.get_dataloader import get_dataloader, lbp_lambda, rg_lambda
from lib.dataloaders.gtsrb_script import GTSRB
from lib.models.full_model import FullModel, ClassificationModel
from lib.arg_parser import arg_parser



a_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print('path', a_path)
sys.path.insert(0, a_path)


def train_classifier(args, context_model, classification_model, train_loader, criterion, optimizer, test_loader,
                     val_loader):
    total_step = len(train_loader)
    print('Total steps', total_step)
    classification_model.train()

    print('Training Epochs', args.num_epochs)
    best_val_acc = 0

    for epoch in range(args.num_epochs):
        epoch_acc1 = 0
        epoch_acc5 = 0
        epoch_class_acc = np.zeros(args.num_classes)

        loss_epoch = 0
        context_model.eval()
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{args.num_epochs + args.start_epoch}',
                  unit='batch') as pbar:

            for step, (img, label) in enumerate(train_loader):

                classification_model.zero_grad()

                model_input = img.to(args.device)  # shape (4,1,64,64)

                label = label.to(args.device)

                with torch.no_grad():
                    output_dic = context_model(model_input)

                img_encodings = output_dic['img_encoding'].detach()  # Shape [128, 1024, 7, 7]
                img_encodings = img_encodings.view(img_encodings.size(0), -1)
                prediction = classification_model(img_encodings)
                label = label.to(args.device)
                loss = criterion(prediction, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                class_threshold = 0.49  # This is need in the case of multi-target classification (CODEBRIM dataset)
                if args.dataset == 'CODEBRIM':
                    outputs = torch.sigmoid(prediction.data)
                    outputs = outputs > class_threshold
                    comp = outputs.float() == label.float()
                    acc1 = (torch.prod(comp, dim=1)).sum().cpu().item() / outputs.shape[0]  # all classes have to be
                    # correctly predicted
                    correct_per_class = torch.sum(comp, dim=0) / outputs.shape[0]
                    epoch_acc1 += acc1
                    correct_per_class = correct_per_class.detach().cpu().numpy()
                    epoch_class_acc = correct_per_class + epoch_class_acc

                    epoch_acc5 += 0
                    acc5 = 0

                else:  # calculate accuracy
                    acc1, acc5 = accuracy_classification(prediction.data, label, topk=(1, 5))
                    epoch_acc1 += acc1
                    epoch_acc5 += acc5

                sample_loss = loss.item()
                loss_epoch += sample_loss
                if args.dataset == 'CODEBRIM' or args.dataset == 'ER':

                    pbar.set_postfix(
                        **{'loss (batch)': sample_loss, 'Top 1 Acc': acc1, 'Class Acc': correct_per_class})
                else:
                    pbar.set_postfix(
                        **{'loss': sample_loss, 'Top 1(batch)': acc1, 'Top 5': acc5})

                pbar.update(1)

        print("Overall training accuracy for this epoch: ", epoch_acc1 / total_step)
        logging.info(f"Overall training accuracy for this epoch: {epoch_acc1 / total_step} ")
        if args.dataset == 'CODEBRIM' or args.dataset == 'ER':
            print("Overall training accuracy for this epoch: ", epoch_class_acc / total_step)

            logging.info(f"Overall training accuracy for this epoch: {epoch_class_acc / total_step} ")

        if (epoch + 1) % 2 == 0:
            # validate the model - in this case, test_loader loads validation data
            val_acc1, val_acc5, val_loss = test_classifier(
                args, context_model, classification_model, val_loader, criterion
            )
            if val_acc1 > best_val_acc:
                best_val_acc = val_acc1
                torch.save(classification_model.state_dict(), os.path.join(args.model_path, f'best_classifier.pth.tar'))

    classification_model.load_state_dict(torch.load(os.path.join(args.model_path, f'best_classifier.pth.tar')))

    val_acc1, val_acc5, val_loss = test_classifier(
        args, context_model, classification_model, test_loader, criterion
    )
    return val_acc1, val_acc5, val_loss


def test_classifier(args, context_model, classification_model, test_loader, criterion):
    total_step = len(test_loader)
    classification_model.eval()

    starttime = time.time()

    loss_epoch = 0
    epoch_acc1 = 0
    epoch_acc5 = 0

    epoch_class_acc = np.zeros(args.num_classes)

    for step, (img, label) in enumerate(test_loader):

        if args.dataset == 'ER':
            img, label = transform_batch(img, label)
        model_input = img.to(args.device)  # shape (4,1,64,64)
        label = label.to(args.device)
        if args.train_mode == 'predSim' or args.train_mode == 'CE':
            label_one_hot = to_one_hot(label)
            label_one_hot = label_one_hot.to(args.device)

        else:
            label_one_hot = None

        with torch.no_grad():
            output_dic = context_model(model_input)
        img_encodings = output_dic['img_encoding'].detach()

        img_encodings = img_encodings.view(img_encodings.size(0), -1)

        prediction = classification_model(img_encodings)

        loss = criterion(prediction, label)

        # calculate accuracy
        class_threshold = 0.49
        if args.dataset == 'CODEBRIM' or args.dataset == 'ER':
            outputs = torch.sigmoid(prediction.data)
            outputs = outputs > class_threshold
            comp = outputs.float() == label.float()
            acc1 = (torch.prod(comp, dim=1)).sum().cpu().item() / outputs.shape[
                0]  # all clases have to be predicted correct
            correct_per_class = torch.sum(comp, dim=0) / outputs.shape[0]
            correct_per_class = correct_per_class.detach().cpu().numpy()
            epoch_class_acc = correct_per_class + epoch_class_acc

            epoch_acc1 += acc1
            epoch_acc5 += 0
            acc5 = 0
        else:
            acc1, acc5 = accuracy_classification(prediction.data, label, topk=(1, 5))
            epoch_acc1 += acc1
            epoch_acc5 += acc5

        sample_loss = loss.item()
        loss_epoch += sample_loss

    print("Testing Accuracy: ", epoch_acc1 / total_step)
    logging.info(f"Testing Accuracy: {epoch_acc1 / total_step} ")

    if args.dataset == 'CODEBRIM' or args.dataset == 'ER':
        print("Testing Accuracy Class wise: ", epoch_class_acc / total_step)

        logging.info(f"Testing Accuracy Class wise: {epoch_class_acc / total_step} ")
        return epoch_acc1 / total_step, epoch_class_acc / total_step, loss_epoch / total_step


    else:
        return epoch_acc1 / total_step, epoch_acc5 / total_step, loss_epoch / total_step


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

    args = arg_parser.parse_args()

    model = FullModel(args)
    model.load_models(epoch=args.res_epoch)
    # get the nb of the input nodes for the classifier
    if args.dataset == 'UCF101' or args.dataset == 'ER':
        random_input = torch.rand((1, args.input_ch, 16, args.input_dim, args.input_dim))

    else:
        random_input = torch.rand((1, args.input_ch, args.input_dim, args.input_dim))
    random_input = random_input.to(args.device)
    out = model(random_input)['img_encoding']
    input_features = out.view(out.size(0), -1).shape[1]
    print('Creating Logger...')
    logging.basicConfig(
        handlers=[
            logging.FileHandler(filename=os.path.join(args.model_path, f'run_history_classfier.log'),  # args.model_path
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
                        ''')

    print('Loading Data...')
    if args.dataset == "GTSRB":
        trans = [
            # transforms.RandomAutocontrast(),
            transforms.Resize([args.input_dim, args.input_dim]),
            # transforms.RandomCrop([args.input_dim, args.input_dim]),
            transforms.ToTensor()
        ]
        if args.input_mode == 'lbp':
            trans.append(transforms.Lambda(lbp_lambda))
        if args.input_mode == 'rgNorm':
            trans.append(transforms.Lambda(rg_lambda))

        data_transforms = transforms.Compose(trans)

        path_csv = '/data/aj_data/data/GTSRB/trainingset/training.csv' #TODO change this
        csv_data = pd.read_csv(path_csv)

        train_csv, val_csv = train_test_split(csv_data, test_size=0.05)

        print(f"Number of training samples = {len(train_csv)}")
        print(f"Number of validation samples = {len(val_csv)}")

        # Create Datasets
        trainset = GTSRB(
            root_dir=args.data_input_dir, train=True, transform=data_transforms,
            csv_data=train_csv)

        valset = GTSRB(
            root_dir=args.data_input_dir, train=True, transform=data_transforms,
            csv_data=val_csv)
        testset = GTSRB(
            root_dir=args.data_input_dir, train=False, transform=data_transforms)

        if args.nb_samples < 1.0:
            # if sample represents the percentage of the dataset used for training 1.0 means the whole dataset
            valid_size = args.nb_samples
            num_train = len(trainset)
            indices = list(range(num_train))
            np.random.shuffle(indices)
            split = int(np.floor(valid_size * num_train))
            train_idx, throwaway_idx = indices[:split], indices[split:]
            train_sampler = SubsetRandomSampler(train_idx)
            # default dataset loaders
            train_loader = torch.utils.data.DataLoader(
                trainset, batch_size=args.batch_size, sampler=train_sampler, num_workers=16
            )  # shuffle=True,
            print('finished sampling from train', len(train_loader))
        else:
            # Load Datasets
            train_loader = torch.utils.data.DataLoader(
                trainset, batch_size=args.batch_size, shuffle=True, num_workers=16)

        # Load Datasets
        val_loader = torch.utils.data.DataLoader(
            valset, batch_size=args.batch_size, shuffle=True, num_workers=16)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=16)
        val_loader = test_loader

    else:
        train_loader, _, test_loader, _ = get_dataloader(args)
        val_loader = test_loader
    classifier = ClassificationModel(in_channels=input_features, num_classes=args.num_classes, #hidden_nodes=2048, TODO check this one
                                     linear_classifier=args.linear_classifier,
                                     p=0.5)  # args.linear_classifier

    if args.device != "cpu" and torch.cuda.is_available():  # and not configs['RESUME']
        classifier = torch.nn.DataParallel(classifier,
                                           device_ids=list(range(torch.cuda.device_count())))

    if args.dataset == 'CODEBRIM' or args.dataset == 'ER':
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    else:
        criterion = torch.nn.CrossEntropyLoss()  # DiceLoss(weight =class_weights)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr,  # args.lr, #args.lr 0.0005 lr 0.00005
                                 weight_decay=args.weight_decay)  # weight_decay=args.weight_decay TODO

    val_acc1, val_acc5, val_loss = train_classifier(args, context_model=model, classification_model=classifier,
                                                    train_loader=train_loader, criterion=criterion, optimizer=optimizer,
                                                    test_loader=test_loader, val_loader=val_loader)

    # Creating Header
    d = {'Model': [], 'Train_mode': [], 'Hinge_mode': [], 'Archi': [], 'Encoder Training Dataset': [], 'Dataset': [],
         'Augmented Training dataset': [], 'Augmented Evaluation Dataset': [],
         'Linear_classifier': [], 'lr': [], 'Batch': [],
         'Sampling_Percentage': [], 'Acc1': [], 'Acc5': [], 'Loss': []}

    df = pd.DataFrame(data=d)

    encoder_dataset_name = os.path.basename(os.path.abspath(os.path.join(args.model_path, "..")))
    archi = os.path.basename(os.path.abspath(os.path.join(args.model_path, "../..")))

    new_row = {'Model': os.path.basename(args.model_path), 'Train_mode': args.train_mode, 'Hinge_mode': args.input_mode,
               'Archi': archi, 'Encoder Training Dataset': encoder_dataset_name, 'Dataset': args.dataset,
               'Augmented Training dataset': args.aug_train, 'Augmented Evaluation Dataset': args.aug_eval,
               'Linear_classifier': args.linear_classifier, 'lr': args.lr, 'Batch': args.batch_size,
               'Sampling_Percentage': args.nb_samples, 'Acc1': val_acc1, 'Acc5': val_acc5, 'Loss': val_loss}

    save_eval_history(d, new_row,
                      os.path.join(args.save_dir, 'classification_results', args.dataset, args.csv_file))
