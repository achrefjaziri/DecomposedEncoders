
This is the code for the paper: "On Decomposed Encoder Design with Bio-inspired Inductive Biases and its Implications on Classifier Performance"

This code is an extension of the following works: 
https://github.com/EPFL-LCN/pub-illing2021-neurips and 

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
The current default is set to 2 GPUs. In our experiments, training each encoder takes around 2 days using two Nvidia-A100 GPUs.  


We also provide a bash script to train all encoders presented in the paper. 
Additionally, all pretrained encoders can be downloaded from here:


## Evaluation ##

To train a classifier and evaluate the classification performance for a single encoder model use the following command:
```

```

For a multi-encoder models use the following command:

```

```

On the CODEBRIM dataset, we included some tests on perturbed images to replicate the experiments use the test_perturbations.py and test_perturbations_multi.py scripts with the same parameters.
To plot the t-SNE dimensionality reduction plots, use the dim_reduction_plots.py script.
