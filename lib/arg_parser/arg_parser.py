import time
import argparse
import os
import torch
import numpy as np
from arg_parser import general_args,train_args


def parse_args():
    # load parameters and options
    parser = argparse.ArgumentParser(description='PyTorch local error training')

    parser = general_args.parse_general_args(parser)
    parser = train_args.parse_train_args(parser)

    args = parser.parse_args()

    args.time = time.ctime()

    # Device configuration
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args.experiment = "vision"

    return args


def create_log_path(opt, add_path_var=""):
    unique_path = False

    if opt.save_dir != "":
        opt.log_path = os.path.join(opt.data_output_dir, "logs", opt.save_dir)
        unique_path = True
    elif add_path_var == "features" or add_path_var == "images":
        opt.log_path = os.path.join(opt.data_output_dir, "logs", add_path_var, os.path.basename(opt.model_path))
        unique_path = True
    else:
        opt.log_path = os.path.join(opt.data_output_dir, "logs", add_path_var, opt.time)

    # hacky way to avoid overwriting results of experiments when they start at exactly the same time
    while os.path.exists(opt.log_path) and not unique_path:
        opt.log_path += "_" + str(np.random.randint(100))

    if not os.path.exists(opt.log_path):
        os.makedirs(opt.log_path)

