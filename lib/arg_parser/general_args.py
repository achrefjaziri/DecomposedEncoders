import argparse

def parse_general_args(parser):

    #General model parameters
    parser.add_argument('--vgg_name', default='vgg6',
                        help='model, mlp, vgg13, vgg16, vgg19, vgg8b, vgg11b, resnet18, resnet34, wresnet28-10 and more (default: vgg8b)')
    parser.add_argument('--train_mode', default='hinge',
                        help='Training modes are: predSim,CPC, hinge,var_hinge, CE, contrastive, classifier   Default (hinge)')

    parser.add_argument('--input_mode', default='vanilla',
                        help='Training modes are: lbp,gabor,color,vanilla,crop_contrastive')

    parser.add_argument('--hebb_extension', default='no_hebb',
                        help='Training modes are: hebb,no_hebb,hebb_byol')
    parser.add_argument(
        "--backprop",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Boolean to determine whether we use local training rules (for trainig mode predSim,hinge,contrastive) or backprop across the whole model (for CE, contrastive, hinge modes)",
    )

    parser.add_argument("--num_epochs",default=400,type=int,help='Number of training epochs')

    parser.add_argument('--no_print_stats', default=False,
                        help='Print Stats after certainnumber of steps')

    parser.add_argument(
        "--device",
        default="cuda",
        help="cpu or cuda to use GPUs",
    )

    parser.add_argument(
        "--port",
        default="6010",
        help="Port",
    )

    parser.add_argument(
        "--test_mode",
        default=False,
        help="Boolean to activate test mode",
    )
    parser.add_argument('--resume', default=False,
                      help='checkpoint to resume training from a certain checkpoint or training a classifier network')


    parser.add_argument(
        "--save_dir",
        default="./logs",
        help="If given, uses this string to create directory to save results in "
             "(be careful, this can overwrite previous results); "
             "otherwise saves logs according to time-stamp",
    )


    #Parameters related to the dataset
    parser.add_argument(
        "--dataset",
        default='STL10',
        help="Available Datasets: MNIST,FashionMNIST,KuzushijiMNIST,CIFAR10,CIFAR100,GTSRB,SVHN,STL10,ImageNet",
    )
    parser.add_argument(
        "--data_input_dir",
        default='/data/aj_data/data',
        help="Boolean to decide whether to use special weight initialization (delta orthogonal)",
    )

    parser.add_argument(
        "--input_dim",
        default=64,
        type=int,
        help="Size of the input image. Recommended: 224 for ImageNet,64 for STL and for the rest ",
    )

    parser.add_argument(
        "--resize_input",
        default=False,
        help="Size of the input image. Recommended: 224 for ImageNet,64 for STL and 32 for the rest ",
    )
    parser.add_argument(
        "--resize_size",
        default=256,
        type=int,
        help="Resize of the input image before cropping.",
    )

    parser.add_argument(
        "--input_ch",
        default=3,
        type=int,
        help="Number of input channels. This parameter does not matter for one channel datasets like MNIST",
    )
    parser.add_argument(
        "--num_classes",
        default=10,
        type=int,
        help="Number of classes for the training data (needed only for the supervised models CE and predSim) ",
    )

    parser.add_argument(
        "--cutout",
        default=False,
        help="Apply cutout regularization",
    )

    parser.add_argument('--n_holes', type=int, default=1,
                        help='number of holes to cut out from image')
    parser.add_argument('--length', type=int, default=16,
                        help='length of the cutout holes in pixels')

    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size",
    )

    # Parameters related to multi GPU training
    parser.add_argument('--nodes',  default=1,
                        type=int, metavar='N')
    parser.add_argument('--gpus',  default=2,
                        help='number of gpus per node')
    parser.add_argument('--world_size', default=None,
                      help='number of gpus per node')

    parser.add_argument('--nr', default=0,
                        help='ranking within the nodes')

    parser.add_argument('--train_classifier', default=False,
                        help='Training a classifier instead of resuming the training of the encoder')


    ### Parameters to resume training the encoder
    parser.add_argument(
        "--start_epoch",
        default=0,
        type=int,
        help="Epoch to start training from: "
             "v=0 - start training from scratch, "
             "v>0 - load pre-trained model that was trained for v epochs and continue training "
             "(path to pre-trained model needs to be specified in opt.model_path)",
    )

    parser.add_argument(
        "--model_path",
        default='/home/ajaziri/Thesis_work/src/vision/main/workspace/logs/vgg6/STL10/2022-03-04_15-37-55', #2022-02-08_11-54-08/ 2022-02-18_20-55-23
        help="Path to the directory containing the model",
    )
    ### Parameters for the training of the classifier
    parser.add_argument(
        "--csv_file",
        default='Classification_results_test_dorsal.csv',
        help="Percentage of images in the training set from the whole set",
    )

    parser.add_argument(
        "--nb_samples",
        default=1.0,
        type=float,
        help="Percentage of images in the training set from the whole set",
    )
    parser.add_argument(
        "--res_epoch",
        default=199,
        type=int,
        help="Resume epoch",
    )
    parser.add_argument("--aug_eval", default=False, help="Use image perturbations during the model evaluation")
    parser.add_argument("--aug_train", default=False,
                        help="Use image perturbations during the training of the classifier")
    parser.add_argument("--linear_classifier", default=True,
                        help="The classifier does not have a non-linearity layer")

    ### Parameters for comparing the learned representations of trained models
    parser.add_argument('--train_mode1', default='CE',
                        help='Training modes are: predSim, hinge, CE, contrastive, classifier   Default (hinge)')

    parser.add_argument(
        "--backprop1",
        default=True,
        help="Boolean to determine whether we use local training rules (for trainig mode predSim,hinge,contrastive) or backprop across the whole model (for CE, contrastive, hinge modes)",
    )
    parser.add_argument(
        "--model_path1",
        default='/home/ajaziri/Thesis_work/src/vision/main/workspace/logs/vgg6/STL10/2022-05-29_07-44-23',# #'/home/ajaziri/Thesis_work/src/vision/main/workspace/logs/vgg6/STL10/2022-03-06_21-16-34'
        # 2022-02-08_11-54-08
        help="Path to the directory containing the model",
    )
    parser.add_argument(
        "--res_epoch1",
        default=99,
        type=int,
        help="Resume epoch",
    )

    parser.add_argument('--train_mode2', default='CPC',
                        help='Training modes are: predSim, hinge, CE, contrastive, classifier   Default (hinge)')

    parser.add_argument(
        "--backprop2",
        default=True,
        help="Boolean to determine whether we use local training rules (for trainig mode predSim,hinge,contrastive) or backprop across the whole model (for CE, contrastive, hinge modes)",
    )
    parser.add_argument(
        "--model_path2",
        default='/home/ajaziri/Thesis_work/src/vision/main/workspace/logs/vgg6/STL10/2022-03-06_21-16-34',
        help="Path to the directory containing the model",
    )
    parser.add_argument(
        "--res_epoch2",
        default=199,
        type=int,
        help="Resume epoch",
    )

    parser.add_argument(
        "--model_path_van",
        default='/home/ajaziri/Thesis_work/src/vision/main/workspace/logs/vgg6/STL10/2022-03-06_21-16-34',
        help="Path to the directory containing the model",
    )
    parser.add_argument(
        "--res_epoch_van",
        default=199,
        type=int,
        help="Resume epoch",
    )

    parser.add_argument(
        "--model_path_dtcwt",
        default='/home/ajaziri/Thesis_work/src/vision/main/workspace/logs/vgg6/STL10/2022-03-06_21-16-34',
        help="Path to the directory containing the model",
    )
    parser.add_argument(
        "--res_epoch_dtcwt",
        default=199,
        type=int,
        help="Resume epoch",
    )

    parser.add_argument(
        "--model_path_lbp",
        default='/home/ajaziri/Thesis_work/src/vision/main/workspace/logs/vgg6/STL10/2022-03-06_21-16-34',
        help="Path to the directory containing the model",
    )
    parser.add_argument(
        "--res_epoch_lbp",
        default=199,
        type=int,
        help="Resume epoch",
    )
    parser.add_argument(
        "--model_path_rg",
        default='/home/ajaziri/Thesis_work/src/vision/main/workspace/logs/vgg6/STL10/2022-03-06_21-16-34',
        help="Path to the directory containing the model",
    )
    parser.add_argument(
        "--res_epoch_rg",
        default=199,
        type=int,
        help="Resume epoch",
    )
    return parser
