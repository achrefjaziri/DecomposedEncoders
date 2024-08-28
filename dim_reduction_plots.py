"""
    Script to generate TSNE and PCA plots for encoder features on GTSRB and STL10 datasets.
"""
import os,sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torchvision.transforms as transforms
from lib.dataloaders.get_dataloader import get_multi_modal_stl10_dataloader
from lib.arg_parser import arg_parser
from lib.models.full_model import FullModel, ClassificationModel
from lib.dataloaders.get_dataloader import get_dataloader,lbp_lambda,rg_lambda
from lib.gtsrb_script import MultiGTSRB
from lib.dataloaders.get_dataloader import get_dataloader


a_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print('path', a_path)
sys.path.insert(0, a_path)


def pca_visualization(features, labels, idx,name):
    pca_2 = PCA(n_components=2, random_state=200)
    pca_2.fit(features)
    features_pca = pca_2.fit_transform(features)

    # TSNE(n_components=2,learning_rate='auto',init='pca').fit_transform(features)
    # scale and move the coordinates so they fit [0; 1] range
    def scale_to_01_range(x):
        # compute the distribution range
        value_range = (np.max(x) - np.min(x))

        # move the distribution so that it starts from zero
        # by extracting the minimal value from all its values
        starts_from_zero = x - np.min(x)

        # make the distribution fit [0; 1] by dividing by its range
        return starts_from_zero / value_range

    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = features_pca[:, 0]
    ty = features_pca[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    df = pd.DataFrame()
    df["y"] = labels
    df["comp-1"] = tx
    df["comp-2"] = ty



    matplotlib.rcParams["text.usetex"] = False
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.size"] = "18"

    matplotlib.rcParams['pdf.fonttype'] = 42

    sns.color_palette("hls", 10)
    sns.set_style("darkgrid")

    plt.figure()

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), palette="deep",
                    data=df)
    # plt.imshow(np.transpose(img_batch[0], [1, 2, 0]))
    plt.legend([], [], frameon=False)
    plt.savefig(f"./view_examples/tsne_examples_stl10/gtsrb_pca_example{name}.pdf", bbox_inches="tight")


def tsne_visualization(features, labels, idx,name):
    tsne = TSNE(n_components=2,learning_rate='auto',init='pca').fit_transform(features)

    # TSNE(n_components=2,learning_rate='auto',init='pca').fit_transform(features)
    # scale and move the coordinates so they fit [0; 1] range
    def scale_to_01_range(x):
        # compute the distribution range
        value_range = (np.max(x) - np.min(x))

        # move the distribution so that it starts from zero
        # by extracting the minimal value from all its values
        starts_from_zero = x - np.min(x)

        # make the distribution fit [0; 1] by dividing by its range
        return starts_from_zero / value_range


    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    df = pd.DataFrame()
    df["y"] = labels
    df["comp-1"] = tx
    df["comp-2"] = ty

    matplotlib.rcParams["text.usetex"] = False
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.size"] = "18"

    matplotlib.rcParams['pdf.fonttype'] = 42
    sns.color_palette("hls", 10)
    sns.set_style("darkgrid")


    plt.figure()

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), palette="deep",
                    data=df)
    # plt.imshow(np.transpose(img_batch[0], [1, 2, 0]))
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend([], [], frameon=False)

    plt.savefig(f"./view_examples/tsne_examples_stl10/gtsrb_tsne_example{name}.pdf", bbox_inches="tight")


def create_pcas(args, test_loader, context_model1,
                context_model2=None, context_model3=None, context_model4=None):
    total_step = len(test_loader)
    print('Total steps', total_step)

    for epoch in range(1):
        context_model1.eval()
        context_model2.eval()
        context_model3.eval()
        context_model4.eval()

        features_wavelet = []
        features_rg = []
        features_lbp = []
        features_vanilla = []
        features_all = []

        corresponding_labels = []

        with tqdm(total=len(test_loader), desc=f'Epoch {epoch + 1}/{args.num_epochs + args.start_epoch}',
                  unit='batch') as pbar:

            for step, (vanilla_img, rg_img, lbp_img, wavelet_img, label) in enumerate(test_loader):
                # print('shape input',img.shape)

                vanilla_img = vanilla_img.to(args.device)  # shape (4,1,64,64)
                rg_img = rg_img.to(args.device)  # shape (4,1,64,64)
                lbp_img = lbp_img.to(args.device)  # shape (4,1,64,64)
                wavelet_img = wavelet_img.to(args.device)  # shape (4,1,64,64)

                label = label.to(args.device)
                with torch.no_grad():
                    output_dic2 = context_model2(rg_img.detach().contiguous())
                    output_dic3 = context_model3(lbp_img.detach().contiguous())
                    output_dic = context_model1(wavelet_img.detach().contiguous())
                    output_dic4 = context_model4(vanilla_img.detach().contiguous())

                img_encodings = output_dic['img_encoding'].detach()  # Shape [128, 1024, 7, 7]

                img_encodings2 = output_dic2['img_encoding'].detach()  # Shape [128, 1024, 7, 7]
                img_encodings3 = output_dic3['img_encoding'].detach()  # Shape [128, 1024, 7, 7]
                img_encodings4 = output_dic4['img_encoding'].detach()  # Shape [128, 1024, 7, 7]
                # print("img encodings before reshape xxxxxxxxxxxxxxxxxxxxxx",img_encodings.shape,img_encodings2.shape,img_encodings3.shape)

                # img_encodings = img_encodings.view(img_encodings.size(0), -1)
                # img_encodings2 = img_encodings2.view(img_encodings2.size(0), -1)
                # img_encodings3 = img_encodings3.view(img_encodings3.size(0), -1)
                # img_encodings4 = img_encodings4.view(img_encodings4.size(0), -1)

                imgs_encodings_all = torch.cat(
                    [img_encodings.view(img_encodings.size(0), -1), img_encodings2.view(img_encodings2.size(0), -1),
                     img_encodings3.view(img_encodings3.size(0), -1)], dim=1)

                current_outputs_wavelet = img_encodings.detach().cpu().numpy()
                current_outputs_rg = img_encodings2.detach().cpu().numpy()
                current_outputs_lbp = img_encodings3.detach().cpu().numpy()
                current_outputs_vanilla = img_encodings4.detach().cpu().numpy()
                current_outputs_all = imgs_encodings_all.detach().cpu().numpy()

                # features = np.concatenate((features, current_outputs))
                features_wavelet.append(current_outputs_wavelet)
                features_lbp.append(current_outputs_lbp)
                features_rg.append(current_outputs_rg)
                features_vanilla.append(current_outputs_vanilla)
                features_all.append(current_outputs_all)



                corresponding_labels.append(label.cpu().numpy())




            # Reshape patches before stiching them up
            features_wavelet = np.asarray(features_wavelet[:-1])
            print("features wavelet",features_wavelet.shape)

            features_wavelet = features_wavelet.mean(axis=(3, 4))
            print("features wavelet",features_wavelet.shape)
            features_wavelet = features_wavelet.reshape(-1, features_wavelet.shape[-1])
            print("features wavelet",features_wavelet.shape)


            features_lbp = np.asarray(features_lbp[:-1])
            features_lbp = features_lbp.mean(axis=(3, 4))
            features_lbp = features_lbp.reshape(-1, features_lbp.shape[-1])

            features_rg = np.asarray(features_rg[:-1])
            features_rg = features_rg.mean(axis=(3, 4))
            features_rg = features_rg.reshape(-1, features_rg.shape[-1])

            features_vanilla = np.asarray(features_vanilla[:-1])
            features_vanilla = features_vanilla.mean(axis=(3, 4))
            features_vanilla = features_vanilla.reshape(-1, features_vanilla.shape[-1])

            features_all = np.asarray(features_all[:-1])
            print("features all",features_all.shape)
            #features_all = features_all.mean(axis=(3, 4))
            #print("features all",features_all.shape)
            features_all = features_all.reshape(-1, features_all.shape[-1])
            print("features all",features_all.shape)


            corresponding_labels = np.ravel(np.asarray(corresponding_labels[:-1]))
            print('labels shape',corresponding_labels.shape)


            """
            chosen_classes = [10]
            pca_visualization(features_wavelet, corresponding_labels, 6,
                              f'wav{chosen_classes[0]}')
            pca_visualization(features_rg, corresponding_labels, 6,
                              f'rg{chosen_classes[0]}')
            pca_visualization(features_lbp, corresponding_labels, 6,
                              f'lbp{chosen_classes[0]}')
            pca_visualization(features_vanilla, corresponding_labels, 6,
                              f'van{chosen_classes[0]}')
            pca_visualization(features_all, corresponding_labels, 6,
                              f'all{chosen_classes[0]}')

            tsne_visualization(features_wavelet, corresponding_labels, 6,
                               f'wav{chosen_classes[0]}')
            tsne_visualization(features_rg, corresponding_labels, 6,
                               f'rg{chosen_classes[0]}')
            tsne_visualization(features_lbp, corresponding_labels, 6,
                               f'lbp{chosen_classes[0]}')
            tsne_visualization(features_vanilla, corresponding_labels, 6,
                               f'van{chosen_classes[0]}')
            tsne_visualization(features_all, corresponding_labels, 6,
                               f'all{chosen_classes[0]}')
            """
            #View examples 1 GTSRB: [0,2,18,20,30,33]
            #View examples 2 GTSRB: [10,12,30,33,40,41]

            for chosen_classes in [[0,3,6,9]]: #[0,1,5,7],[0,2,18,20],[1,5,6,9],[4,8,6,3],  0,35,15,24
                relevant_labels= np.where((corresponding_labels==chosen_classes[0]) |(corresponding_labels==chosen_classes[1])|
                                          (corresponding_labels==chosen_classes[2])| (corresponding_labels==chosen_classes[3]
                                       ))

                # | (corresponding_labels==chosen_classes[4])| (corresponding_labels==chosen_classes[5])

                #(corresponding_labels==0) |
                pca_visualization(features_wavelet[relevant_labels], corresponding_labels[relevant_labels], 6,f'wav{chosen_classes[0]}')
                pca_visualization(features_rg[relevant_labels], corresponding_labels[relevant_labels], 6,f'rg{chosen_classes[0]}')
                pca_visualization(features_lbp[relevant_labels], corresponding_labels[relevant_labels], 6,f'lbp{chosen_classes[0]}')
                pca_visualization(features_vanilla[relevant_labels], corresponding_labels[relevant_labels], 6,f'van{chosen_classes[0]}')
                pca_visualization(features_all[relevant_labels], corresponding_labels[relevant_labels], 6,f'all{chosen_classes[0]}')

                tsne_visualization(features_wavelet[relevant_labels], corresponding_labels[relevant_labels], 6,f'wav{chosen_classes[0]}')
                tsne_visualization(features_rg[relevant_labels], corresponding_labels[relevant_labels], 6,f'rg{chosen_classes[0]}')
                tsne_visualization(features_lbp[relevant_labels], corresponding_labels[relevant_labels], 6,f'lbp{chosen_classes[0]}')
                tsne_visualization(features_vanilla[relevant_labels], corresponding_labels[relevant_labels], 6,f'van{chosen_classes[0]}')
                tsne_visualization(features_all[relevant_labels], corresponding_labels[relevant_labels], 6,f'all{chosen_classes[0]}')


if __name__ == "__main__":

    args1 = arg_parser.parse_args()
    args2 = arg_parser.parse_args()
    args3 = arg_parser.parse_args()
    args4 = arg_parser.parse_args()

    args3.model_path = '/home/ajaziri/Thesis_work/src/vision/main/workspace/logs_lbp/vgg6/GTSRB/2022-07-23_17-49-19'
    args3.train_mode = 'CPC'
    args3.backprop = True
    args3.res_epoch = 199
    args3.input_mode = 'lbp'
    args3.input_ch = 1



    args2.model_path = '/home/ajaziri/Thesis_work/src/vision/main/workspace/logs_rgNorm/vgg6/GTSRB/2022-07-22_20-55-09'
    args2.train_mode = 'CPC'
    args2.backprop = True
    args2.res_epoch = 199
    args2.input_mode = 'rgNorm'
    args2.input_ch = 2



    args1.model_path = '/home/ajaziri/Thesis_work/src/vision/main/workspace/logs_dtcwt/vgg6/GTSRB/2022-07-22_20-48-59'
    args1.train_mode = 'CPC'
    args1.backprop = True
    args1.res_epoch = 199
    args1.input_mode = 'dtcwt'
    args1.input_ch = 3
    args1.input_dim = 256
    args1.resize_input = True



    args4.model_path = "/home/ajaziri/Thesis_work/src/vision/main/workspace/logs/vgg6/GTSRB/2022-07-20_18-10-21"
    args4.train_mode = 'CPC'
    args4.backprop = True
    args4.res_epoch = 199
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

    #_, _, test_loader, _ = get_multi_modal_stl10_dataloader(args4, args2, args3, args1)

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

    train_loader, _, test_loader, _ = get_multi_modal_stl10_dataloader(args4, args2, args3, args1)
    #path_csv = '/data/aj_data/data/GTSRB/trainingset/training.csv'
    #csv_data = pd.read_csv(path_csv)


    #testset = MultiGTSRB(
    #    root_dir=args1.data_input_dir, train=False, transform=data_transforms_vanilla,
    #    transform2=data_transforms_rg, transform3=data_transforms_lbp, transform4=data_transforms_wavelet)

    #test_loader = torch.utils.data.DataLoader(
    #    testset, batch_size=args1.batch_size, shuffle=False, num_workers=16)

    create_pcas(args4, test_loader=test_loader,
                context_model1=wavelet_model, context_model2=rg_model, context_model3=lbp_model,
                context_model4=vanilla_model)
