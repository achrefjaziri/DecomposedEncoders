
This is the code for the paper: Representation Learning in a Decomposed Encoder Design for Bio-inspired Hebbian Learning published in ECCV 2024 Workshop Proceedings.

This code is an extension of the following work: 
https://github.com/EPFL-LCN/pub-illing2021-neurips  

## Setup ##

To setup the conda environment, simply run
```
bash ./setup_env.sh
```
After that activate the conda environment and run the bash scripts specified in the following sections.



## Training Encoders ##

An example of training command of the encoder extended with wavelet operator.
```
python train_enc.py --port port_number --save_dir workspace/logs_dtcwt --batch_size 64 --no-backprop --train_mode hinge --input_mode dtcwt --input_dim 256 --resize_input True --input_ch 3 --dataset GTSRB
```
For more details about the training parameters check this document.  

Port_number needs to be specified since we are using the pytorch distributed training module. This allows us to use multiple GPUs to train our models. 
The current default is set to 2 GPUs. 


We also provide a bash script to train all encoders presented in the paper. 


## Evaluation ##

To train a classifier and evaluate the classification performance for a single encoder model use the following command:
```
python ..
```

For a multi-encoder models use the following command:

```
python ..
```

On the CODEBRIM dataset, we included some tests on perturbed images to replicate the experiments use the test_perturbations.py and test_perturbations_multi.py scripts with the same parameters.
To plot the t-SNE dimensionality reduction plots, use the dim_reduction_plots.py script.
