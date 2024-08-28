from optparse import OptionGroup

def parser_reload_args(parser):

    parser.add_argument('--train_classifier', default=False, type=str,
                        help='Training a classifier instead of resuming the training of the encoder')



    ### Parameters to resume training the encoder
    parser.add_argument(
        "--start_epoch",
        type="int",
        default=0,
        help="Epoch to start training from: "
        "v=0 - start training from scratch, "
        "v>0 - load pre-trained model that was trained for v epochs and continue training "
        "(path to pre-trained model needs to be specified in opt.model_path)",
    )

    parser.add_argument(
        "--model_path",
        default='',
        help="Path to the directory containing the model",
    )


    ### Parameters for the training of the classifier
    parser.add_argument(
        "--nb_samples",
        default=1.0,
        help="Percentage of images in the training set from the whole set",
    )
    parser.add_argument("--perturbed_eval",default=False,help="Use image perturbations during the model evaluation")
    parser.add_argument("--perturbed_train",default=False,help="Use image perturbations during the training of the classifier")


    return parser
