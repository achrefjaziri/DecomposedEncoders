"""
    In this script, we want to test the impact of common data perturbations on the performance on CODEBRIM.
    This script specifically tests the performance of single encoder models.
"""

import time, os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import kornia
from kornia.augmentation.container.image import ImageSequential, ParamItem
import torch
from lib.arg_parser import arg_parser
from lib.dataloaders.get_dataloader import get_dataloader
from lib.models.full_model import FullModel, ClassificationModel
from lib.utils.misc_functions import to_one_hot, accuracy_classification, save_eval_history


def train_classifier(args, context_model, classification_model, train_loader, criterion, optimizer, test_loader):
    total_step = len(train_loader)
    print('Total steps', total_step)
    classification_model.train()

    print('Training Epochs', args.num_epochs)

    best_val_acc = 0

    for epoch in range(args.num_epochs):
        epoch_acc1 = 0
        epoch_acc5 = 0
        epoch_class_acc = np.zeros(args.num_classes)

        # epoch_class_acc = np.array([0,0,0,0,0,0])
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
                # print(prediction.data.shape,target.shape)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                class_threshold = 0.49
                if args.dataset == 'CODEBRIM':
                    outputs = torch.sigmoid(prediction.data)
                    outputs = outputs > class_threshold
                    comp = outputs.float() == label.float()
                    acc1 = (torch.prod(comp, dim=1)).sum().cpu().item() / outputs.shape[
                        0]  # all clases have to be predicted correct
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
                if args.dataset == 'CODEBRIM':

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

        if (epoch + 1) % 5 == 0:
            # validate the model - in this case, test_loader loads validation data
            val_acc1, val_acc5, val_loss = test_classifier(
                args, context_model, classification_model, test_loader, criterion
            )
            if val_acc1 > best_val_acc:
                best_val_acc = val_acc1
                best_model = classification_model

    val_acc1, val_acc5, val_loss = test_classifier(
        args, context_model, classification_model, test_loader, criterion
    )

    return val_acc1, val_acc5, val_loss, best_model


def test_classifier(args, context_model, classification_model, test_loader, criterion, aug_list=None):
    total_step = len(test_loader)
    classification_model.eval()


    loss_epoch = 0
    epoch_acc1 = 0
    epoch_acc5 = 0

    epoch_class_acc = np.zeros(args.num_classes)

    for step, (img, label) in enumerate(test_loader):

        model_input = img.to(args.device)  # shape (4,1,64,64)
        label = label.to(args.device)

        if aug_list is not None:
            model_input = aug_list(model_input)

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
    os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

    args = arg_parser.parse_args()

    model = FullModel(args)
    model.load_models(epoch=args.res_epoch)

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
    train_loader, _, test_loader, _ = get_dataloader(args)

    classifier = ClassificationModel(in_channels=input_features, num_classes=args.num_classes, hidden_nodes=1024,
                                     linear_classifier=True)  # args.linear_classifier

    if args.device != "cpu" and torch.cuda.is_available():  # and not configs['RESUME']
        classifier = torch.nn.DataParallel(classifier,
                                           device_ids=list(range(torch.cuda.device_count())))

    if args.dataset == 'CODEBRIM' or args.dataset == 'ER':
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    else:
        criterion = torch.nn.CrossEntropyLoss()  # DiceLoss(weight =class_weights)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-5,  # args.lr
                                 weight_decay=0.00001)  # weight_decay=args.weight_decay
    val_acc1, val_acc5, val_loss, classifier = train_classifier(args, context_model=model,
                                                                classification_model=classifier,
                                                                train_loader=train_loader, criterion=criterion,
                                                                optimizer=optimizer, test_loader=test_loader)

    # Creating Header
    d = {'Model': [], 'Train_mode': [], 'Hinge_mode': [], 'Perturbation': [], 'Perturbation_Param': [],
         'Linear_classifier': [],
         'Sampling_Percentage': [], 'Acc1': [], 'Acc5': [], 'Loss': []}

    df = pd.DataFrame(data=d)

    new_row = {'Model': os.path.basename(args.model_path), 'Train_mode': args.train_mode, 'Hinge_mode': args.input_mode,
               'Perturbation': 'None', 'Perturbation_Param': 0,
               'Linear_classifier': args.linear_classifier,
               'Sampling_Percentage': args.nb_samples, 'Acc1': val_acc1, 'Acc5': val_acc5, 'Loss': val_loss}

    save_eval_history(d, new_row, os.path.join(
        args.save_dir, 'classification_results', args.dataset,
        'robustness_eval_linear', args.csv_file))

    for pert in ['Noise', 'Motion', 'Shadows', 'Brightness', 'Jitter']:
        if pert == 'Jitter':
            aug_list = ImageSequential(
                kornia.augmentation.RandomPlanckianJitter(p=1.0, select_from=[23, 24, 1, 2]),
            )

            new_row = {'Model': os.path.basename(args.model_path), 'Train_mode': args.train_mode,
                       'Hinge_mode': args.input_mode,
                       'Perturbation': pert, 'Perturbation_Param': 0,
                       'Linear_classifier': args.linear_classifier,
                       'Sampling_Percentage': args.nb_samples, 'Acc1': val_acc1, 'Acc5': val_acc5, 'Loss': val_loss}

            save_eval_history(d, new_row, os.path.join(
                args.save_dir, 'classification_results', args.dataset,
                'robustness_eval_linear', args.csv_file))


        else:
            for index, param in enumerate([0.05, 0.1, 0.15, 0.2, 0.3]):
                if pert == 'Noise':
                    aug_list = ImageSequential(
                        kornia.augmentation.RandomGaussianNoise(mean=param, std=0.01, same_on_batch=False, p=1,
                                                                keepdim=False,
                                                                return_transform=None)  # 0.05,0.1,0.15,0.2
                    )
                elif pert == 'Motion':
                    filters = [3, 5, 7, 9, 11]

                    filter_size = filters[index]
                    print('filter_size', filter_size)
                    aug_list = ImageSequential(

                        kornia.filters.MotionBlur(filter_size, 35., 0.0),  # intensity 0.05,0.1,0.15,0.2,0.3,0.5
                    )
                elif pert == 'Shadows':
                    aug_list = ImageSequential(
                        kornia.augmentation.RandomPlasmaShadow(roughness=(0.1, 0.7), shade_intensity=(- param, -0.01),
                                                               shade_quantity=(0.05, 0.4), p=1.)
                    )
                elif pert == 'Brightness':
                    aug_list = ImageSequential(
                        kornia.augmentation.RandomPlasmaBrightness(roughness=(0.1, 0.7), intensity=(0.0, param),
                                                                   same_on_batch=False,
                                                                   p=1.0, keepdim=False, return_transform=None)
                        # intensity 0.05,0.1,0.15,0.2,0.3,0.5
                    )

                val_acc1, val_acc5, val_loss = test_classifier(
                    args, model, classifier, test_loader, criterion, aug_list=aug_list
                )

                if pert != 'Motion':
                    out_param = param
                else:
                    out_param = filter_size
                new_row = {'Model': os.path.basename(args.model_path), 'Train_mode': args.train_mode,
                           'Hinge_mode': args.input_mode,
                           'Perturbation': pert, 'Perturbation_Param': out_param,
                           'Linear_classifier': args.linear_classifier,
                           'Sampling_Percentage': args.nb_samples, 'Acc1': val_acc1, 'Acc5': val_acc5, 'Loss': val_loss}

                save_eval_history(d, new_row, os.path.join(
                    args.save_dir, 'classification_results', args.dataset,
                    'robustness_eval_linear', args.csv_file))
