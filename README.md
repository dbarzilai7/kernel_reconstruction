<h1 align="center"> Querying Kernel Methods Suffices for Reconstructing their Training Data </h1>

## Abstract
Over-parameterized models have raised concerns about their potential to memorize training data, even when achieving strong generalization. The privacy implications of such memorization are generally unclear, particularly in scenarios where only model outputs are accessible. We study this question in the context of kernel methods, and demonstrate both empirically and theoretically that querying kernel models at various points suffices to reconstruct their training data, even without access to model parameters. Our results hold for a range of kernel methods, including kernel regression, support vector machines, and kernel density estimation. Our hope is that this work can illuminate potential privacy concerns for such models.

## File Overview
reconstruction.py - Main file to run the reconstruction attacks.
run_commands.py - includes the command line arguments needed for the results in the paper. 
run_analysis.ipynb - Once the commands are run, can use this notebook to create the figures and compute metrics. Computing the run statistics may be slow due to DSSIMs. It is advised to let this run on a GPU in the background. 

## Datasets
The paper makes use of several datasets. CIFAR10/100 and celebA should be straightforward. Regarding the CIFAR5M dataset - It should be downloaded following the instructions in https://github.com/preetum/cifar5m. The directory where the files are downloaded needs to be specified in the file cifar5m.py. This is only relevant when using --dataset CIFAR5M.

## Getting Started
A good place to start is to run one of the commands from run_commands.py. For example, the following command reconstructs data from a Laplace kernel trained on 100 images. This is a relatively lightweight setting; the remaining commands in run_commands.py reconstruct 500 images.

```
python reconstruction.py --recovery_size 100 --train_size 100 --init_std 0.3 --lr 2e-2 --coefs_lr 1e-2 --scheduler onecycle --plot_all_recons --save_metrics --output_dir outputs --save_recons --use_test_time_seed --recovery_model laplace --recovery_gamma 0.15 --target_model laplace --target_gamma 0.15 --dataset CIFAR10 --batch_size 50000 --num_steps 150000
```

