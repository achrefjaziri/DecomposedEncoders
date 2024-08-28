"""
    Script to train and evaluate a linear classifier for a multi-encoder model.
"""
import time, os, sys
import numpy as np
import torch
import torchvision.transforms as transforms
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
from lib.arg_parser import arg_parser
from lib.dataloaders.get_dataloader import lbp_lambda, rg_lambda
from lib.models.full_model import FullModel, ClassificationModel
from lib.dataloaders.gtsrb_script import MultiGTSRB
from lib.dataloaders.get_dataloader import get_multi_modal_stl10_dataloader
from lib.dataloaders.codebrim_dataloader import get_multi_modal_codebrim_dataloader
from lib.dataloaders.endless_runner_dataloader import transform_batch, get_multi_modal_endlessrunner_dataloader
from lib.dataloaders.ucf101_dataloader import get_multi_modal_ucf101_dataloader
from lib.utils.misc_functions import accuracy_classification, save_eval_history

a_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print('path', a_path)
sys.path.insert(0, a_path)


def train_classifier(args, context_model1, classification_model, train_loader, criterion, optimizer, validation_loader,
                     test_loader,
                     context_model2=None, context_model3=None, context_model4=None):
    """
    :param args:
    :param context_model1: the encoder with dtcwt operator
    :param classification_model: the classifer network that we want to train
    :param train_loader: the training data loader
    :param criterion: loss function
    :param optimizer: ADAM optimizer
    :param validation_loader: validation data
    :param test_loader: test data
    :param context_model2: the encoder with rg normalization operator
    :param context_model3: the encoder with lbp operator
    :param context_model4: the encoder without any operator (this is optional)
    :return: top1, top5 accuracy scores and loss score on the test set after 30 training epochs.
    """
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
        context_model1.eval()
        context_model2.eval()
        context_model3.eval()
        if context_model4 != None:
            context_model4.eval()

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{args.num_epochs + args.start_epoch}',
                  unit='batch') as pbar:

            for step, (vanilla_img, rg_img, lbp_img, wavelet_img, label) in enumerate(train_loader):
                # print('shape input',img.shape)

                classification_model.zero_grad()

                if args.dataset == 'ER':
                    vanilla_img, label_new = transform_batch(vanilla_img, label, patch_size=120)
                    lbp_img, _ = transform_batch(lbp_img, label, patch_size=120)
                    rg_img, _ = transform_batch(rg_img, label, patch_size=120)
                    wavelet_img, _ = transform_batch(wavelet_img, label, patch_size=120)
                    label = label_new

                vanilla_img = vanilla_img.to(args.device)  # shape (4,1,64,64)
                rg_img = rg_img.to(args.device)  # shape (4,1,64,64)
                lbp_img = lbp_img.to(args.device)  # shape (4,1,64,64)
                wavelet_img = wavelet_img.to(args.device)  # shape (4,1,64,64)

                label = label.to(args.device)
                with torch.no_grad():
                    output_dic2 = context_model2(rg_img.detach().contiguous())
                    output_dic3 = context_model3(lbp_img.detach().contiguous())
                    output_dic = context_model1(wavelet_img.detach().contiguous())
                    if context_model4 is not None:
                        output_dic4 = context_model4(vanilla_img.detach().contiguous())

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
                    imgs_encodings_all = torch.cat([img_encodings4, img_encodings4, img_encodings4],
                                                   dim=1)

                prediction = classification_model(imgs_encodings_all)

                label = label.to(args.device)
                loss = criterion(prediction, label)
                # print(prediction.data.shape,target.shape)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                class_threshold = 0.49
                if args.dataset == 'CODEBRIM' or args.dataset == 'ER':
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
                if args.dataset == 'CODEBRIM' or args.dataset == 'ER':

                    pbar.set_postfix(
                        **{'loss (batch)': sample_loss, 'Top 1 Acc (batch)': acc1,
                           'Class Acc (batch)': correct_per_class})
                else:
                    pbar.set_postfix(
                        **{'loss (batch)': sample_loss, 'Top 1 Acc (batch)': acc1, 'Top 5 Acc (batch)': acc5})

                pbar.update(1)

        if (epoch + 1) % 2 == 0:
            # validate the model - in this case, test_loader loads validation data
            val_acc1, val_acc5, val_loss = test_classifier(args, context_model1, classification_model,
                                                           validation_loader,
                                                           criterion, context_model2, context_model3, context_model4)

            if val_acc1 > best_val_acc:
                best_val_acc = val_acc1
                torch.save(classification_model.state_dict(),
                           os.path.join(args.model_path, f'best_classifier_multi.pth.tar'))

        print("Overall training accuracy for this epoch: ", epoch_acc1 / total_step)
        logging.info(f"Overall training accuracy for this epoch: {epoch_acc1 / total_step} ")
        if args.dataset == 'CODEBRIM':
            print("Overall training accuracy for this epoch: ", epoch_class_acc / total_step)

            logging.info(f"Overall training accuracy for this epoch: {epoch_class_acc / total_step} ")

    classification_model.load_state_dict(torch.load(os.path.join(args.model_path, f'best_classifier_multi.pth.tar')))

    val_acc1, val_acc5, val_loss = test_classifier(
        args, context_model1, classification_model, test_loader, criterion, context_model2, context_model3,
        context_model4
    )

    return val_acc1, val_acc5, val_loss


def test_classifier(args, context_model1, classification_model, test_loader, criterion, context_model2=None,
                    context_model3=None, context_model4=None):
    """
    :param args:
    :param context_model1: the encoder with dtcwt operator
    :param classification_model: the classifer network that we want to train
    :param criterion: loss function
    :param test_loader: test data
    :param context_model2: the encoder with rg normalization operator
    :param context_model3: the encoder with lbp operator
    :param context_model4: the encoder without any operator (this is optional)
    :return: top1, top5 accuracy scores and loss score on the test set after 30 training epochs.
    """
    total_step = len(test_loader)
    classification_model.eval()

    starttime = time.time()

    loss_epoch = 0
    epoch_acc1 = 0
    epoch_acc5 = 0
    epoch_class_acc = np.zeros(args.num_classes)

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
            imgs_encodings_all = torch.cat([img_encodings4, img_encodings4, img_encodings4],
                                           dim=1)
            # print(img_encodings.shape,img_encodings2.sh

        prediction = classification_model(imgs_encodings_all)
        loss = criterion(prediction, label)

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
        """
                if step % 10 == 0:
            print(
                "Step [{}/{}], Time (s): {:.1f}, Acc1: {:.4f}, Acc5: {:.4f}, Loss: {:.4f}".format(
                    step, total_step, time.time() - starttime, acc1, acc5, sample_loss
                )
            )
            starttime = time.time()

        """

    print("Testing Accuracy: ", epoch_acc1 / total_step)
    logging.info(f"Testing Accuracy: {epoch_acc1 / total_step} ")
    if args.dataset == 'CODEBRIM' or args.dataset == 'ER':
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


    args2.model_path = args2.model_path_rg  # '/home/ajaziri/Thesis_work/src/vision/main/workspace/logs_rgNorm/vgg6/GTSRB/2022-07-22_20-55-09'
    # args2.train_mode = 'CPC'
    # args2.backprop = True
    args2.res_epoch = args2.res_epoch_rg
    args2.input_mode = 'rgNorm'
    args2.input_ch = 2


    args1.model_path = args1.model_path_dtcwt  # '/home/ajaziri/Thesis_work/src/vision/main/workspace/logs_dtcwt/vgg6/GTSRB/2022-07-22_20-48-59'
    # args1.train_mode = 'CPC'
    # args1.backprop = True
    args1.res_epoch = args1.res_epoch_dtcwt  # 199
    args1.input_mode = 'dtcwt'  # 'dtcwt'
    args1.input_ch = 3
    args1.input_dim = 256
    args1.resize_input = True


    args4.model_path = args4.model_path_van
    # args4.train_mode = 'CPC'
    # args4.backprop = True
    args4.res_epoch = args4.res_epoch_van
    args4.input_mode = 'vanilla'
    args4.input_ch = 3

    '''
    args4.model_path = '/home/ajaziri/Thesis_work/src/vision/main/workspace/logs_motion_disc/vgg6/UCF101/2022-05-23_16-22-34'
    args4.train_mode = 'so_mvment'
    args4.backprop = True
    args4.res_epoch = 399
    args4.input_mode = 'vanilla'
    args4.input_ch = 3
    args4.input_dim = 120
    args4.resize_input = True
    args4.resize_size= 512
    '''

    print("Loading wavelet model")

    wavelet_model = FullModel(args1)
    wavelet_model.load_models(epoch=args1.res_epoch)
    main_model1, _ = wavelet_model.get_main_branch_model()
    ##get the nb of the input nodes for the classifier

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

    ##get the nb of the input nodes for the classifier

    if args1.dataset == 'UCF101' or args1.dataset == 'ER':
        random_input = torch.rand((1, args4.input_ch, 16, args4.input_dim, args4.input_dim))

    else:
        random_input = torch.rand((1, args4.input_ch, args4.input_dim, args4.input_dim))
    # random_input = torch.rand((1, args4.input_ch, args4.input_dim, args4.input_dim))

    # This code snippet makes sure that our classifier has a correct width/number of features.
    random_input = random_input.to(args4.device)
    print("random input shape", random_input.shape, args4.device)
    out = vanilla_model(random_input)['img_encoding']
    print("shape out", out.shape)

    # if args1.dataset=='CODEBRIM':
    #    input_features =451584  #out.view(out.size(0), -1).shape[1] * 3  # 51200
    # else:
    input_features = out.view(out.size(0), -1).shape[1] * 3  # 51200
    print(input_features)
    print('Creating Logger...')
    logging.basicConfig(
        handlers=[
            logging.FileHandler(
                filename=os.path.join(args1.save_dir, 'classification_results',
                                      args1.dataset, f'multi_encoder_classifcation.log'),  # args.model_path
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

    if args1.dataset == 'STL10':
        train_loader, _, test_loader, _ = get_multi_modal_stl10_dataloader(args4, args2, args3, args1)
    elif args1.dataset == 'ER':
        train_loader, _, test_loader, _ = get_multi_modal_endlessrunner_dataloader(args4)
    elif args1.dataset == 'UCF101':
        train_loader, _, test_loader, _ = get_multi_modal_ucf101_dataloader(args4)

    elif args1.dataset == 'GTSRB':
        trans_wavelet = [
            # transforms.RandomAutocontrast(),
            transforms.Resize([args1.input_dim, args1.input_dim]),
            # transforms.RandomCrop([args.input_dim, args.input_dim]),
            transforms.ToTensor()
        ]

        data_transforms_vanilla = transforms.Compose([
            # transforms.RandomAutocontrast(),
            transforms.Resize([args4.input_dim, args4.input_dim]),
            # transforms.RandomCrop([args.input_dim, args.input_dim]),
            transforms.ToTensor()
        ])

        data_transforms_vanilla = transforms.Compose([
            # transforms.RandomAutocontrast(),
            transforms.Resize([args4.input_dim, args4.input_dim]),
            # transforms.RandomCrop([args.input_dim, args.input_dim]),
            transforms.ToTensor()
        ])

        data_transforms_lbp = transforms.Compose([
            # transforms.RandomAutocontrast(),
            transforms.Resize([args4.input_dim, args4.input_dim]),
            transforms.ToTensor(),
            transforms.Lambda(lbp_lambda)
        ])

        data_transforms_rg = transforms.Compose([
            # transforms.RandomAutocontrast(),
            transforms.Resize([args4.input_dim, args4.input_dim]),
            transforms.ToTensor(),
            transforms.Lambda(rg_lambda)

        ])

        data_transforms_wavelet = transforms.Compose([
            # transforms.RandomAutocontrast(),
            transforms.Resize([args1.input_dim, args1.input_dim]),
            # transforms.RandomCrop([args.input_dim, args.input_dim]),
            transforms.ToTensor()
        ])

        path_csv = '/data/aj_data/data/GTSRB/trainingset/training.csv'  # TODO change this
        csv_data = pd.read_csv(path_csv)

        train_csv, val_csv = train_test_split(csv_data, test_size=0.05)
        print(f"Number of training samples = {len(train_csv)}")
        print(f"Number of validation samples = {len(val_csv)}")

        # Create Datasets
        trainset = MultiGTSRB(
            root_dir=args1.data_input_dir, train=True, transform=data_transforms_vanilla,
            transform2=data_transforms_rg, transform3=data_transforms_lbp, transform4=data_transforms_wavelet,
            csv_data=train_csv)

        valset = MultiGTSRB(
            root_dir=args1.data_input_dir, train=True, transform=data_transforms_vanilla,
            transform2=data_transforms_rg, transform3=data_transforms_lbp, transform4=data_transforms_wavelet,
            csv_data=val_csv)
        testset = MultiGTSRB(
            root_dir=args1.data_input_dir, train=False, transform=data_transforms_vanilla,
            transform2=data_transforms_rg, transform3=data_transforms_lbp, transform4=data_transforms_wavelet)
        # Load Datasets
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args1.batch_size, shuffle=True, num_workers=16)

        # Load Datasets
        val_loader = torch.utils.data.DataLoader(
            valset, batch_size=args1.batch_size, shuffle=True, num_workers=16)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args1.batch_size, shuffle=False, num_workers=16)
        val_loader = test_loader
    else:
        train_loader, _, test_loader, _ = get_multi_modal_codebrim_dataloader(args4, args2, args3, args1)

    classifier = ClassificationModel(in_channels=input_features, num_classes=args1.num_classes, hidden_nodes=512,
                                     linear_classifier=True)

    if args1.device != "cpu" and torch.cuda.is_available():  # and not configs['RESUME']
        classifier = torch.nn.DataParallel(classifier,
                                           device_ids=list(range(torch.cuda.device_count())))

    if args1.dataset == 'CODEBRIM':
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    else:
        criterion = torch.nn.CrossEntropyLoss()  # DiceLoss(weight =class_weights)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args1.lr,  # args.lr 5e-5
                                 weight_decay=args1.weight_decay)  # weight_decay=args.weight_decay 0.00001
    val_acc1, val_acc5, val_loss = train_classifier(args4, context_model1=wavelet_model,
                                                    classification_model=classifier,
                                                    train_loader=train_loader, criterion=criterion, optimizer=optimizer,validation_loader=val_loader,
                                                    test_loader=test_loader, context_model2=rg_model,
                                                    context_model3=lbp_model, context_model4=None)

    logging.info(f''' Classification Results:
                                Accuracy:          {val_acc1}
                                Loss:      {val_acc5}
                                Acc (classwise or top5):   {val_acc5}
                            ''')

    # Creating Header
    d = {'Model': [], 'Train_mode': [], 'Hinge_mode': [], 'Archi': [], 'Encoder Training Dataset': [], 'Dataset': [],
         'Augmented Training dataset': [], 'Augmented Evaluation Dataset': [],
         'Linear_classifier': [],
         'Sampling_Percentage': [], 'Acc1': [], 'Acc5': [], 'Loss': []}

    df = pd.DataFrame(data=d)

    encoder_dataset_name = os.path.basename(os.path.abspath(os.path.join(args1.model_path, "..")))
    archi = os.path.basename(os.path.abspath(os.path.join(args1.model_path, "../..")))

    new_row = {'Model': os.path.basename(args1.model_path), 'Train_mode': args1.train_mode,
               'Hinge_mode': f'{args1.input_mode}_{args2.input_mode}_{args3.input_mode}',
               'Archi': archi, 'Encoder Training Dataset': encoder_dataset_name, 'Dataset': args1.dataset,
               'Augmented Training dataset': args1.aug_train, 'Augmented Evaluation Dataset': args1.aug_eval,
               'Linear_classifier': args1.linear_classifier,
               'Sampling_Percentage': args1.nb_samples, 'Acc1': val_acc1, 'Acc5': val_acc5, 'Loss': val_loss}

    save_eval_history(d, new_row,
                      os.path.join(args1.save_dir, 'classification_results', args1.dataset, args1.csv_file))
