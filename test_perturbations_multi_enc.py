"""
    In this script, we want to test the impact of common data perturbations on the performance on CODEBRIM.
    This script specifically tests the framework with multiple-encoders.
"""

import time, os, sys
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import kornia
from kornia.augmentation.container.image import ImageSequential, ParamItem
import torch
import torchvision.transforms as transforms
from lib.arg_parser import arg_parser
from lib.dataloaders.get_dataloader import get_multi_modal_stl10_dataloader
from lib.dataloaders.codebrim_dataloader import get_multi_modal_codebrim_dataloader
from lib.dataloaders.lbp_rg_transfo import LBP, NormalizedRG  # dataloaders
from lib.models.full_model import FullModel, ClassificationModel
from lib.utils.misc_functions import to_one_hot, accuracy_classification, save_eval_history

a_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print('path', a_path)
sys.path.insert(0, a_path)


def train_classifier(args, context_model1, classification_model, train_loader, criterion, optimizer, test_loader,
                     context_model2=None, context_model3=None, context_model4=None):
    total_step = len(train_loader)
    print('Total steps', total_step)
    classification_model.train()

    print('Training Epochs', args.num_epochs)
    best_val_acc = 0

    for epoch in range(args.num_epochs):
        epoch_acc1 = 0
        epoch_acc5 = 0
        epoch_class_acc = np.array([0, 0, 0, 0, 0, 0])

        loss_epoch = 0
        context_model1.eval()
        context_model2.eval()
        context_model3.eval()
        if context_model4 != None:
            context_model4.eval()

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{args.num_epochs + args.start_epoch}',
                  unit='batch') as pbar:

            for step, (vanilla_img, rg_img, lbp_img, wavelet_img, label) in enumerate(train_loader):

                classification_model.zero_grad()

                vanilla_img = vanilla_img.to(args.device)  # shape (4,1,64,64)
                rg_img = rg_img.to(args.device)  # shape (4,1,64,64)
                lbp_img = lbp_img.to(args.device)  # shape (4,1,64,64)
                wavelet_img = wavelet_img.to(args.device)  # shape (4,1,64,64)


                label = label.to(args.device)

                with torch.no_grad():
                    output_dic2 = context_model2(rg_img)
                    output_dic3 = context_model3(lbp_img)
                    output_dic = context_model1(wavelet_img)
                    if context_model4 is not None:
                        output_dic4 = context_model4(vanilla_img)

                img_encodings = output_dic['img_encoding'].detach()  # Shape [128, 1024, 7, 7]

                img_encodings2 = output_dic2['img_encoding'].detach()  # Shape [128, 1024, 7, 7]
                img_encodings3 = output_dic3['img_encoding'].detach()  # Shape [128, 1024, 7, 7]
                if context_model4 is not None:
                    img_encodings4 = output_dic4['img_encoding'].detach()  # Shape [128, 1024, 7, 7]

                img_encodings = img_encodings.view(img_encodings.size(0), -1)
                img_encodings2 = img_encodings2.view(img_encodings2.size(0), -1)
                img_encodings3 = img_encodings3.view(img_encodings3.size(0), -1)

                if context_model4 is None:

                    imgs_encodings_all = torch.cat([img_encodings, img_encodings2, img_encodings3], dim=1)
                else:
                    img_encodings4 = img_encodings4.view(img_encodings4.size(0), -1)
                    imgs_encodings_all = torch.cat([img_encodings, img_encodings2, img_encodings3, img_encodings4],
                                                   dim=1)

                prediction = classification_model(imgs_encodings_all)

                label = label.to(args.device)
                loss = criterion(prediction, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                class_threshold = 0.49
                outputs = torch.sigmoid(prediction.data)
                outputs = outputs > class_threshold
                comp = outputs.float() == label.float()
                acc1 = (torch.prod(comp, dim=1)).sum().cpu().item() / outputs.shape[
                    0]
                correct_per_class = torch.sum(comp, dim=0) / outputs.shape[0]
                epoch_acc1 += acc1
                correct_per_class = correct_per_class.detach().cpu().numpy()
                epoch_class_acc = correct_per_class + epoch_class_acc

                epoch_acc5 += 0
                acc5 = 0
                sample_loss = loss.item()
                loss_epoch += sample_loss

                pbar.set_postfix(
                    **{'loss (batch)': sample_loss, 'Top 1 Acc (batch)': acc1,
                       'Class Acc (batch)': correct_per_class})

                pbar.update(1)

        if (epoch + 1) % 5 == 0:
            # validate the model - in this case, test_loader loads validation data
            val_acc1, val_acc5, val_loss = test_classifier(args, context_model1, classification_model,
                                                           test_loader, criterion, context_model2,
                                                           context_model3, context_model4)

            if val_acc1 > best_val_acc:
                best_val_acc = val_acc1
                torch.save(classification_model.state_dict(), os.path.join(args.model_path, f'best_classifier.pth.tar'))

        print("Overall training accuracy for this epoch: ", epoch_acc1 / total_step)
        logging.info(f"Overall training accuracy for this epoch: {epoch_acc1 / total_step} ")
        if args.dataset == 'CODEBRIM':
            print("Overall training accuracy for this epoch: ", epoch_class_acc / total_step)

            logging.info(f"Overall training accuracy for this epoch: {epoch_class_acc / total_step} ")

    classification_model.load_state_dict(
        torch.load(os.path.join(args.model_path, f'best_classifier.pth.tar')))

    val_acc1, val_acc5, val_loss = test_classifier(args, context_model1, classification_model, test_loader,
                                                   criterion, context_model2, context_model3, context_model4)

    return val_acc1, val_acc5, val_loss, classification_model


def test_classifier(args, context_model1, classification_model, test_loader, criterion, context_model2=None,
                    context_model3=None, context_model4=None):
    total_step = len(test_loader)
    classification_model.eval()
    loss_epoch = 0
    epoch_acc1 = 0
    epoch_acc5 = 0
    epoch_class_acc = np.array([0, 0, 0, 0, 0, 0])

    for step, (vanilla_img, rg_img, lbp_img, wavelet_img, label) in enumerate(test_loader):
        vanilla_img = vanilla_img.to(args.device)  # shape (4,1,64,64)
        rg_img = rg_img.to(args.device)  # shape (4,1,64,64)
        lbp_img = lbp_img.to(args.device)  # shape (4,1,64,64)
        wavelet_img = wavelet_img.to(args.device)  # shape (4,1,64,64)

        label = label.to(args.device)

        with torch.no_grad():
            output_dic2 = context_model2(rg_img)
            output_dic3 = context_model3(lbp_img)
            output_dic = context_model1(wavelet_img)
            if context_model4 is not None:
                output_dic4 = context_model4(vanilla_img)

        img_encodings = output_dic['img_encoding'].detach()  # Shape [128, 1024, 7, 7]
        img_encodings2 = output_dic2['img_encoding'].detach()  # Shape [128, 1024, 7, 7]
        img_encodings3 = output_dic3['img_encoding'].detach()  # Shape [128, 1024, 7, 7]
        if context_model4 is not None:
            img_encodings4 = output_dic4['img_encoding'].detach()  # Shape [128, 1024, 7, 7]

        img_encodings = img_encodings.view(img_encodings.size(0), -1)
        img_encodings2 = img_encodings2.view(img_encodings2.size(0), -1)
        img_encodings3 = img_encodings3.view(img_encodings3.size(0), -1)
        if context_model4 is None:
            imgs_encodings_all = torch.cat([img_encodings, img_encodings2, img_encodings3], dim=1)
        else:
            img_encodings4 = img_encodings4.view(img_encodings4.size(0), -1)
            imgs_encodings_all = torch.cat([img_encodings, img_encodings2, img_encodings3, img_encodings4], dim=1)

        prediction = classification_model(imgs_encodings_all)

        loss = criterion(prediction, label)

        class_threshold = 0.49
        if args.dataset == 'CODEBRIM':
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
    if args.dataset == 'CODEBRIM':
        print("Testing Accuracy Class wise: ", epoch_class_acc / total_step)

        logging.info(f"Testing Accuracy Class wise: {epoch_class_acc / total_step} ")
        return epoch_acc1 / total_step, epoch_class_acc / total_step, loss_epoch / total_step


    else:
        return epoch_acc1 / total_step, epoch_acc5 / total_step, loss_epoch / total_step


def lbp_lambda(x):
    lbp_transform = LBP(radius=3, points=24)
    # print('shape in lbp_lambda',x.shape)
    # img_out = torch.Tensor(lbp_transform(x[0].detach().cpu().numpy()))
    all_imgs = []
    transform = transforms.Grayscale()
    for i in range(x.shape[0]):
        current_img_gray = transform(x[i])
        # print("grayscaled shape",current_img_gray.shape)
        current_img = current_img_gray[0].detach().cpu().numpy()
        # print("current image",current_img.shape)
        current_img = np.expand_dims(lbp_transform(current_img), axis=0)
        # print("after lbp transfo",current_img.shape)
        all_imgs.append(current_img)
    all_imgs = np.asarray(all_imgs)
    # print("shape all images", all_imgs.shape)
    img_out = torch.Tensor(all_imgs)

    # img_out=torch.unsqueeze(img_out, 0)
    return img_out


def rg_lambda(x):
    rg_norm = NormalizedRG(conf=False)
    # print('shape i rg_lambda',x.shape)
    all_imgs = []
    for i in range(x.shape[0]):
        current_img = x[i].permute(1, 2, 0).detach().cpu().numpy()
        all_imgs.append(rg_norm(current_img))
    all_imgs = np.asarray(all_imgs)
    # print("shape all images",all_imgs.shape)
    img_out = torch.Tensor(all_imgs).permute(0, 3, 1, 2)
    # print("shape all images final",all_imgs.shape)

    return img_out


def test_classifier_with_pert(args, context_model1, classification_model, test_loader, criterion, context_model2=None,
                              context_model3=None, context_model4=None):
    total_step = len(test_loader)
    classification_model.eval()

    loss_epoch = 0
    epoch_acc1 = 0
    epoch_acc5 = 0
    epoch_class_acc = np.array([0, 0, 0, 0, 0, 0])

    for step, (vanilla_img, rg_img, lbp_img, wavelet_img, label) in enumerate(test_loader):

        if aug_list is not None:
            vanilla_img = aug_list(vanilla_img)

            wavelet_img = aug_list(wavelet_img)

            # Create LBP and RG images from the vanilla images after applying the augmentations/perturbations

            transform_rg = transforms.Compose(
                [transforms.Lambda(rg_lambda)])

            transform_lbp = transforms.Compose(
                [transforms.Lambda(lbp_lambda)])

            rg_img = transform_rg(vanilla_img)
            lbp_img = transform_lbp(vanilla_img)

        vanilla_img = vanilla_img.to(args.device)  # shape (4,1,64,64)
        rg_img = rg_img.to(args.device)  # shape (4,1,64,64)
        lbp_img = lbp_img.to(args.device)  # shape (4,1,64,64)
        wavelet_img = wavelet_img.to(args.device)  # shape (4,1,64,64)

        label = label.to(args.device)

        with torch.no_grad():
            output_dic2 = context_model2(rg_img)
            output_dic3 = context_model3(lbp_img)
            output_dic = context_model1(wavelet_img)
            if context_model4 is not None:
                output_dic4 = context_model4(vanilla_img)

        img_encodings = output_dic['img_encoding'].detach()  # Shape [128, 1024, 7, 7]
        img_encodings2 = output_dic2['img_encoding'].detach()  # Shape [128, 1024, 7, 7]
        img_encodings3 = output_dic3['img_encoding'].detach()  # Shape [128, 1024, 7, 7]
        if context_model4 is not None:
            img_encodings4 = output_dic4['img_encoding'].detach()  # Shape [128, 1024, 7, 7]

        img_encodings = img_encodings.view(img_encodings.size(0), -1)
        img_encodings2 = img_encodings2.view(img_encodings2.size(0), -1)
        img_encodings3 = img_encodings3.view(img_encodings3.size(0), -1)
        if context_model4 is None:
            imgs_encodings_all = torch.cat([img_encodings, img_encodings2, img_encodings3], dim=1)
        else:
            img_encodings4 = img_encodings4.view(img_encodings4.size(0), -1)
            imgs_encodings_all = torch.cat([img_encodings, img_encodings2, img_encodings3, img_encodings4], dim=1)

        prediction = classification_model(imgs_encodings_all)

        loss = criterion(prediction, label)

        class_threshold = 0.49
        if args.dataset == 'CODEBRIM':
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
    if args.dataset == 'CODEBRIM':
        print("Testing Accuracy Class wise: ", epoch_class_acc / total_step)

        logging.info(f"Testing Accuracy Class wise: {epoch_class_acc / total_step} ")
        return epoch_acc1 / total_step, epoch_class_acc / total_step, loss_epoch / total_step


    else:
        return epoch_acc1 / total_step, epoch_acc5 / total_step, loss_epoch / total_step


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'

    args1 = arg_parser.parse_args()
    args2 = arg_parser.parse_args()
    args3 = arg_parser.parse_args()
    args4 = arg_parser.parse_args()

    args3.model_path = args3.model_path_lbp
    args3.res_epoch = args3.res_epoch_lbp
    args3.input_mode = 'lbp'  # 'lbp'
    args3.input_ch = 1

    args2.model_path = args2.model_path_rg
    # args2.train_mode = 'CPC'
    # args2.backprop = True
    args2.res_epoch = args2.res_epoch_rg
    args2.input_mode = 'rgNorm'
    args2.input_ch = 2

    args1.model_path = args1.model_path_dtcwt
    # args1.train_mode = 'CPC'
    # args1.backprop = True
    args1.res_epoch = args1.res_epoch_dtcwt  # 199
    args1.input_mode = 'dtcwt'  # 'dtcwt'
    args1.input_ch = 3
    args1.input_dim = 256
    args1.resize_input = True

    args4.model_path = args4.model_path_van
    args4.res_epoch = args4.res_epoch_van
    args4.input_mode = 'vanilla'
    args4.input_ch = 3

    print("Loading wavelet model")

    wavelet_model = FullModel(args1)
    wavelet_model.load_models(epoch=args1.res_epoch)
    main_model1, _ = wavelet_model.get_main_branch_model()

    print("Loading rg model")

    rg_model = FullModel(args2)
    rg_model.load_models(epoch=args2.res_epoch)
    main_model2, _ = rg_model.get_main_branch_model()

    print("Loading lbp model")

    lbp_model = FullModel(args3)
    lbp_model.load_models(epoch=args3.res_epoch)
    main_model3, _ = lbp_model.get_main_branch_model()

    print("Loading vanilla model")

    vanilla_model = FullModel(args4)
    vanilla_model.load_models(epoch=args4.res_epoch)
    main_model4, _ = vanilla_model.get_main_branch_model()
    print('Creating Logger...')

    random_input = torch.rand((1, args4.input_ch, args4.input_dim, args4.input_dim))
    # random_input = torch.rand((1, args4.input_ch, args4.input_dim, args4.input_dim))
    random_input = random_input.to(args4.device)
    print("random input shape", random_input.shape, args4.device)
    out = vanilla_model(random_input)['img_encoding']
    print("shape out", out.shape)
    input_features = out.view(out.size(0), -1).shape[1] * 3
    print(input_features)
    print('Creating Logger...')
    logging.basicConfig(
        handlers=[
            logging.FileHandler(
                filename=os.path.join(args1.save_dir, 'classification_results',
                                      args1.dataset, f'multi_encoder_classifcation_perturbations.log'),
                # args.model_path
                encoding='utf-8', mode='a+')],
        format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
        datefmt="%F %A %T",
        level=logging.INFO)
    logging.info(f''' Model Paramters:
                                Path dtcwt:          {args1.model_path}
                                Path Rg:      {args2.model_path}
                                Path lbp:   {args3.model_path}
                                Path vanilla:   {args4.model_path}
                                Train mode:   {args4.train_mode}
                                Dataset:   {args1.dataset}
                            ''')

    print('Loading Data...')
    train_loader, _, test_loader, _ = get_multi_modal_codebrim_dataloader(args4, args2, args3, args1)

    classifier = ClassificationModel(in_channels=input_features, num_classes=args1.num_classes,
                                     linear_classifier=True)

    if args1.device != "cpu" and torch.cuda.is_available():  # and not configs['RESUME']
        classifier = torch.nn.DataParallel(classifier,
                                           device_ids=list(range(torch.cuda.device_count())))
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-5,  # args.lr
                                 weight_decay=0.00001)  # weight_decay=args.weight_decay

    val_acc1, val_acc5, val_loss, best_classifier = train_classifier(args1, context_model1=wavelet_model,
                                                                     classification_model=classifier,
                                                                     train_loader=train_loader, criterion=criterion,
                                                                     optimizer=optimizer,
                                                                     test_loader=test_loader, context_model2=rg_model,
                                                                     context_model3=lbp_model, context_model4=None)
    logging.info(f''' Classification Results:
                                    Accuracy:          {val_acc1}
                                    Loss:      {val_acc5}
                                    Acc (classwise or top5):   {val_acc5}
                                ''')
    # Creating Header
    d = {'Model': [], 'Train_mode': [], 'Hinge_mode': [], 'Perturbation': [], 'Perturbation_Param': [],
         'Linear_classifier': [],
         'Sampling_Percentage': [], 'Acc1': [], 'Acc5': [], 'Loss': []}

    df = pd.DataFrame(data=d)

    new_row = {'Model': os.path.basename(args1.model_path), 'Train_mode': args1.train_mode, 'Hinge_mode': 'all',
               'Perturbation': 'None', 'Perturbation_Param': 0,
               'Linear_classifier': args1.linear_classifier,
               'Sampling_Percentage': args1.nb_samples, 'Acc1': val_acc1, 'Acc5': val_acc5, 'Loss': val_loss}

    save_eval_history(d, new_row, os.path.join(
        args1.save_dir, 'classification_results', args1.dataset,
        'robustness_eval_linear',args1.csv_file))

    for pert in ['Noise', 'Motion', 'Shadows', 'Brightness', 'Jitter']:  # 'Noise', 'Motion', 'Shadows', 'Brightness', 'Jitter'
        if pert == 'Jitter':
            aug_list = ImageSequential(
                kornia.augmentation.RandomPlanckianJitter(p=1.0, select_from=[23, 24, 1, 2]),
            )

            new_row = {'Model': os.path.basename(args1.model_path), 'Train_mode': args1.train_mode,
                       'Hinge_mode': 'all',
                       'Perturbation': pert, 'Perturbation_Param': 0,
                       'Linear_classifier': args1.linear_classifier,
                       'Sampling_Percentage': args1.nb_samples, 'Acc1': val_acc1, 'Acc5': val_acc5, 'Loss': val_loss}

            save_eval_history(d, new_row, os.path.join(
                args1.save_dir, 'classification_results', args1.dataset,
                'robustness_eval_linear', args1.csv_file))


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
                    # print('filter_size', filter_size)
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

                elif pert == 'Defor':
                    defor = [1.0, 5.0, 10.0, 20.0, 40.0]

                    defor = defor[index]
                    aug_list = ImageSequential(
                        kornia.augmentation.RandomElasticTransform(kernel_size=(63, 63), sigma=(defor, defor),
                                                                   alpha=(1.0, 1.0),
                                                                   align_corners=False, mode='bilinear',
                                                                   padding_mode='zeros',
                                                                   same_on_batch=False, p=1.0, keepdim=False,
                                                                   return_transform=None),
                        # intensity 0.05,0.1,0.15,0.2,0.3,0.5
                    )

                val_acc1, val_acc5, val_loss = test_classifier_with_pert(args1, context_model1=wavelet_model,
                                                                         classification_model=best_classifier,
                                                                         test_loader=test_loader, criterion=criterion,
                                                                         context_model2=rg_model,
                                                                         context_model3=lbp_model, context_model4=None)

                if pert != 'Motion' and pert != 'Defor':
                    out_param = param
                elif pert == 'Defor':
                    out_param = defor
                else:
                    out_param = filter_size
                new_row = {'Model': os.path.basename(args1.model_path), 'Train_mode': args1.train_mode,
                           'Hinge_mode': 'all',
                           'Perturbation': pert, 'Perturbation_Param': out_param,
                           'Linear_classifier': args1.linear_classifier,
                           'Sampling_Percentage': args1.nb_samples, 'Acc1': val_acc1, 'Acc5': val_acc5,
                           'Loss': val_loss}

                save_eval_history(d, new_row, os.path.join(
                    args1.save_dir, 'classification_results', args1.dataset,
                    'robustness_eval_linear', args1.csv_file))
