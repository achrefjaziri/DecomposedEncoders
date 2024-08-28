from optparse import OptionGroup

def parse_train_args(parser):

    #General Training parameters
    parser.add_argument('--no_batch_norm', default=False,
                        help='ll')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout after each nonlinearity (default: 0.0)')
    parser.add_argument('--nonlin', default='relu',
                        help='nonlinearity, relu or leakyrelu (default: relu)')
    parser.add_argument('--optim', default='adam',
                        help='optimizer to be used')
    parser.add_argument('--beta', default=0.1,type=float,
                        help='beta value for ADAM optimizer')
    parser.add_argument('--lr', type=float, default=5e-4, #5e-4
                        help='initial learning rate (default: 5e-4)')

    parser.add_argument('--lr_var', type=float, default=5e-5,  # 5e-4
                        help='initial learning rate (default: 5e-4)')
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="weight decay or l2-penalty on weights (for ADAM optimiser, default = 0., i.e. no l2-penalty)"
    )

    ############ Next parameters are specific for Hinge loss. These parameters are taken from the CLAPP repo

    parser.add_argument("--prediction_step", default=5,type=int,
                        help="(Number of) Time steps to predict into future for the hinge loss")
    parser.add_argument('--asymmetric_W_pred', default=True,
                        help="Boolean: solve weight transport in W_pred by using two distinct W_pred(1,2) and splitting the score:"
            "Loss(u) -> Loss1(u1) + Loss2(u2) for both, pos. and neg. samples, with"
            "u = z*W_pred*c -> u1 = drop_grad(z)*W_pred1*c, u2 = z*W_pred2*drop_grad(c)")

    parser.add_argument('--negative_samples', default=1,
                        help="Number of negative samples to be used for training")

    parser.add_argument('--gating_av_over_preds', default=False,
                        help='Boolean: average feedback gating (--feedback_gating) from higher layers over different prediction steps (k)')
    parser.add_argument('--detach_c', default=False,
                        help='"Boolean whether the gradient of the context c should be dropped (detached)')
    parser.add_argument('--current_rep_as_negative', default=False,
                        help='Use the current feature vector (context at time t as opposed to predicted time step t+k) itself as/for sampling the negative sample')
    parser.add_argument('--sample_negs_locally', default=True,
                        help='Sample neg. samples from batch but within same location in image, i.e. no shuffling across locations')
    parser.add_argument('--sample_negs_locally_same_everywhere', default=True,
                        help='Extension of --sample_negs_locally_same_everywhere (must be True). No shuffling across locations and same sample (from batch) for all locations. I.e. negative sample is simply a new input without any scrambling')

    parser.add_argument('--either_pos_or_neg_update', default=False,
                        help='Randomly chose to do either pos or neg update in Hinge loss. --negative_samples should be 1. Only used with --current_rep_as_negative True')
    parser.add_argument(
        "--freeze_W_pred",
        default=False,
        help="Boolean whether the k prediction weights W_pred (W_k in ContrastiveLoss) are frozen (require_grad=False).",
    )
    parser.add_argument(
        "--unfreeze_last_W_pred",
        default=False,
        help="Boolean whether the k prediction weights W_pred of the last module should be unfrozen.",
    )

    parser.add_argument(
        "--weight_init",
        default=False,
        help="Boolean to decide whether to use special weight initialization (delta orthogonal)",
    )
    parser.add_argument(
        "--no_pred",
        default=False,
        help="Boolean whether Wpred * c is set to 1 (no prediction). i.e. fourth factor omitted in learning rule",
    )
    parser.add_argument(
        "--no_gamma",
        action="store_true",
        default=False,
        help="Boolean whether gamma (factor which sets the opposite sign of the update for pos and neg samples) is set to 1. i.e. third factor omitted in learning rule",
    )
    parser.add_argument('--overlap_factor', default=2,type=int,
                        help="Overlap factor of the image patches during Hinge training")

    parser.add_argument('--patch_size', default=16,type=int,
                        help="Overlap factor of the image patches during Hinge training")

    parser.add_argument(
        "--varied_video_speed",
        default=False,
        help="Use varying video speeds for video datasets.",
    )

    return parser
