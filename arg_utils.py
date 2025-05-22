import argparse
from models.model_utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for reconstruction script")

    parser.add_argument('--dataset', type=str, required=False,default="MNIST", help='Dataset name')
    parser.add_argument('--max_classes', type=int, default=-1, help='Number of classes to use')
    parser.add_argument('--max_img_size', type=int, default=None, help='Limit the hight and width of the images')
    parser.add_argument('--normalize_to_sphere', default=False, action='store_true', help='If data should be normalized to sphere')
    parser.add_argument('--use_test_time_seed', default=False, action='store_true', help='If true, uses the seed for the data for the paper results. ')

    # recovery args
    parser.add_argument('--recovery_size', type=int, default=10, help='Number of Images to recover')
    parser.add_argument('--recovery_model', type=str, choices=MODEL_LIST, default='laplace')
    parser.add_argument('--recovery_gamma', type=float, default=1.0, help='Gamma parameter for kernels (if relevant)')
    parser.add_argument('--recovery_layers', type=int, default=3, help='Number of layers parameter (if relevant)')
    parser.add_argument('--recovery_pca_dim', type=int, default=0, help='Recover inputs that lie within the top directions of the data distribution. Set 0 to ignore')

    # target model args
    parser.add_argument('--target_model', type=str, choices=MODEL_LIST, default='laplace')
    parser.add_argument('--target_gamma', type=float, default=1.0, help='Gamma parameter for kernels (if relevant)')
    parser.add_argument('--target_layers', type=int, default=3, help='Number of layers parameter (if relevant)')
    parser.add_argument('--target_regularization', type=float, default=0, help='Regularization parameter for kernels (if relevant)')
    parser.add_argument('--train_size', type=int, default=10, help='Training size')
    parser.add_argument('--target_task', type=str, default='krr', help='Task to train kernel if relevant.')

    # optimization params
    parser.add_argument('--num_steps', type=int, required=False,default=30000, help='number of optimization steps')
    parser.add_argument('--batch_size', type=int, default=100000, help='Batch size')
    parser.add_argument('--num_recovery_equations', type=int, default=-1, help='Number of points to use as equations for the reconstructions. -1 for the same as the batch size')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='none', help='Learning rate scheduler')
    parser.add_argument('--resample_each_epoch', default=False, action='store_true', help='Resample each epoch flag')
    parser.add_argument('--noise_as_data', default=False, action='store_true', help="Uses only noise as the recovery data")
    parser.add_argument('--recovery_noise_std', type=float, default=0.0, help='Noise std for the recovery data')
    parser.add_argument('--lr', type=float, required=False,default=1e-3, help='Learning rate for recovered inputs')
    parser.add_argument('--coefs_lr', type=float, required=False,default=1e-3, help='Learning rate for the coefficients')
    parser.add_argument('--gamma_lr', type=float, required=False,default=0.0, help='Learning rate for the gamma in Laplace kernel')
    parser.add_argument('--near_init', default=False, action='store_true', help='Near initialization flag')
    parser.add_argument('--init_std', type=float,default=0.3, help='Near initialization std')
    parser.add_argument('--alpha_init_std', type=float,default=0.05, help='Near initialization std')
    parser.add_argument('--alpha_init_dist', type=str, default='normal', help='Initial distribution of the alphas', choices=['normal', 'uniform', 'rademacher'])
    parser.add_argument('--vae_path', type=str, default=None, help='Path to VAE used to map between latents and images')
    parser.add_argument('--optimize_latents',default=False, action='store_true')
    parser.add_argument('--init_using_data_mean', default=False, action='store_true', help='Initialize using data mean')
    parser.add_argument('--alpha_init_bias', default=0.0, type=float, help='Initial bias for the alpha')


    # Losses
    parser.add_argument('--high_freq_loss', type=float, default=0., help='Penalize Sharpness')
    parser.add_argument('--image_range_loss', type=float, default=0.1, help='Penalize values that are large/small')
    parser.add_argument('--predictions_at_recoveries_loss', type=float, default=0., help='Penalize Sharpness')
    parser.add_argument('--alpha_regularization', type=float, default=0., help='Penalize large values of alpha')
    parser.add_argument('--alpha_regularization_norm', type=float, default=2, help='Norm of regularization')

    # augmentation params    
    parser.add_argument('--horiz_flip', default=False, action='store_true', help='Near initialization std')
    parser.add_argument('--vert_flip', default=False, action='store_true', help='Near initialization std')
    parser.add_argument('--vert_horiz_flip', default=False, action='store_true', help='Near initialization std')   


    # plot and save params
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default='')
    parser.add_argument('--wandb_notes', type=str, default='')
    parser.add_argument('--disable_wandb', default=False, action='store_true')
    parser.add_argument('--plot_init', type=bool, default=True, help='Initial plot flag')
    parser.add_argument('--plot_result', type=bool, default=True, help='Result plot flag')
    parser.add_argument('--plot_all_recons', default=False, action='store_true', help='Plot all reconstructions')
    parser.add_argument('--plot_train_images',type=bool, default=True, help='Plot 10 of the train images')
    parser.add_argument('--save_recons', default=False, action='store_true', help='Save the reconstructions')
    parser.add_argument('--save_metrics', default=False, action='store_true', help='Save the metrics')
    parser.add_argument('--output_dir', type=str, required=False, default='reconstructions', help='where to save the reconstructions')


    return parser.parse_args()