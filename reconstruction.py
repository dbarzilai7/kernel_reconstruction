import copy
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data import LightweightDataLoader, get_dataset, normalization_params
from plot_utils import *
from utils import *
from models.model_utils import *
from arg_utils import parse_args
from skimage.metrics import peak_signal_noise_ratio as psnr
from kornia.metrics import ssim as kornia_ssim
import kornia.augmentation
from torchvision.transforms import functional as F
import seaborn as sns
import multiprocessing
import random
import time
import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


plt.switch_backend('agg')

## maybe move to utils
def init_recovery_params(cfg, dim, out_shape):
    # Initialize the recovery parameters
    # cfg: dict, the configuration dictionary
    # dim: int, the dimension of the data
    # Returns: recovered_inputs, alpha_hat
    data_dist = lambda n, d: torch.randn(n, d) * cfg['init_std']

    recovered_inputs = data_dist(cfg['recovery_size'], dim).to(dtype=torch.float64, device=DEVICE)        
        
    alpha_hat_size = cfg['recovery_size'] if out_shape == 1 else  (cfg['recovery_size'], out_shape)
    if cfg['alpha_init_dist'] == 'uniform':
        alpha_hat = (torch.rand(alpha_hat_size) -0.5).to(dtype=torch.float64, device=DEVICE) * cfg['alpha_init_std'] * sqrt(12)
    elif cfg['alpha_init_dist'] == 'normal':
        alpha_hat = torch.randn(alpha_hat_size, device=DEVICE).to(dtype=torch.float64) * cfg['alpha_init_std']
        alpha_hat += torch.sign(alpha_hat) * cfg['alpha_init_bias'] 
    elif cfg['alpha_init_dist'] == 'rademacher':
        alpha_hat = (torch.randint(2, size=alpha_hat_size).to(dtype=torch.float64, device=DEVICE) - 0.5) * 2 * cfg['alpha_init_std']
    else:
        raise NotImplementedError(f"Unknown alpha_init_dist: {cfg['alpha_init_dist']}")

    recovered_inputs.requires_grad = True
    alpha_hat.requires_grad = True

    return recovered_inputs, alpha_hat


def compute_loss(cfg, pred_labels, recovery_train_labels, recovered_inputs, alpha_hat, data_norm_mean, img_shape, max_value, min_value):
    ### Compute the loss for the recovery
    # cfg: dict, the configuration dictionary
    
    losses = {}
    losses['reconstruction_loss'] = criterion(pred_labels, recovery_train_labels.squeeze())
    if cfg['high_freq_loss'] > 0:
        losses['prior_loss'] = high_freq_loss(recovered_inputs.reshape(-1, *img_shape) * data_norm_mean, cfg['high_freq_loss'])
    if cfg['predictions_at_recoveries_loss'] > 0:
        with torch.no_grad():
            y_recovered_inputs = target_model(recovered_inputs)
        y_hat_recovered_inputs = recovery_model(recovered_inputs, recovered_inputs, alpha_hat)
        losses['predictions_at_recoveries_loss'] = criterion(y_recovered_inputs.squeeze(), y_hat_recovered_inputs.squeeze()) * cfg['predictions_at_recoveries_loss']
    if cfg['alpha_regularization'] > 0:
        losses['alpha_norm_loss'] = torch.norm(alpha_hat, p=cfg['alpha_regularization_norm']) * cfg['alpha_regularization']
    if cfg['image_range_loss'] > 0:
        losses['img_loss'] = ((nn.functional.relu(recovered_inputs * data_norm_mean - max_value) ** 2).mean() + \
                            (nn.functional.relu(-recovered_inputs * data_norm_mean + min_value) ** 2).mean()) * cfg['image_range_loss'] * \
                                recovered_inputs.shape[0]
    return losses




def train_recovery(recovered_inputs):
    all_recovery_train_labels = []
    num_batches = len(recovery_dataloader)    
    for step in range(cfg['num_steps']):
        if  step==0 or num_batches > 1 or cfg['resample_each_epoch']:
            if step % num_batches == 0 :
                dataloader_iter = iter(recovery_dataloader)
            recovery_train_data, _ = next(dataloader_iter)
            recovery_train_data = augment_data(cfg,recovery_train_data)
            recovery_train_data = recovery_train_data.squeeze().flatten(1).to(dtype=torch.float64, device=DEVICE)
            if cfg['normalize_to_sphere']:
                    recovery_train_data /= data_norm_mean                                   
            if step < num_batches: 
                with torch.no_grad():
                    recovery_train_labels = target_model(recovery_train_data)
                    all_recovery_train_labels.append(recovery_train_labels)
            else:
                recovery_train_labels = all_recovery_train_labels[step % num_batches]
            
        optimizer.zero_grad()

        if cfg['recover_from_latents']:
            recovered_inputs = upscale(recovered_latents)
            if cfg['normalize_to_sphere']:
                recovered_inputs /= data_norm_mean
        pred_labels = recovery_model(recovery_train_data, recovered_inputs, alpha_hat)
        losses = compute_loss(cfg, pred_labels, recovery_train_labels, recovered_inputs, alpha_hat, data_norm_mean, img_shape, max_value, min_value)
        loss = sum(losses.values())
        loss.backward()
        optimizer.step()        
        scheduler.step()

        if cfg['recovery_pca_dim'] > 0:
            with torch.no_grad():
                recovered_inputs.data = project(recovered_inputs.data)

        log_dict = losses
        log_dict.update({'Loss': loss.item(), "Learning Rate": scheduler.get_last_lr()[0]})
        if cfg['gamma_lr'] > 0:
            log_dict.update({'Gamma': recovery_model.gamma.item()})
                    
        log_dict.update({'Alpha Distance': (alpha_hat - alpha_hat_orig).norm().item()})
        log_dict.update({'Alpha Mean Absolute Value': (alpha_hat).abs().mean().item()})
        with torch.no_grad():
            if (step % 1000) == 0:
                recoveries, _, _ = compute_recovery_order_MSE(data, recovered_inputs)
                if cfg['normalize_to_sphere']:
                    recoveries *= data_norm_mean
                mean_recovery = torch.mean(recoveries)
                top_recovery = torch.min(recoveries)
                progress_log = log_recovery_progress(step, loss, recoveries, mean_recovery, top_recovery, scheduler, recovered_inputs, img_shape)
                log_dict.update(progress_log)
        wandb.log(log_dict)

    return recovered_inputs


if __name__ == '__main__':
    cfg = vars(parse_args())
    cfg = wandb_init(cfg)

    if cfg['use_test_time_seed']:
        RANDOM_SEED = 66856 # Randomly generated between 1 and 100,000. Used for reproducibility. No hyperparameter tuning or experiments were done with this seed.

    # Set random seed so that first batch of train is the same as the NN was trained on in train_nn.py
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # train and test are flipped here since we need more data for the recovery
    train_dataset, num_classes, _, _, _ = get_dataset(cfg['dataset'], max_classes=cfg['max_classes'], train=False,
                                              max_samples=cfg['train_size'], size=cfg['max_img_size'])
    train_dataloader = DataLoader(train_dataset, batch_size=cfg["train_size"], shuffle=True)

    # will be used for optimizing the reconstruction. Ensures the target model was not trained on the same data we optimize over.
    num_equations = cfg['num_recovery_equations'] if cfg['num_recovery_equations'] > 0 else cfg['batch_size']
    recovery_dataset, _, data_norm_mean, min_value, max_value = get_dataset(cfg['dataset'], max_classes=cfg['max_classes'], max_samples=num_equations, train=True, size=cfg['max_img_size'])
    # shuffle just once before training
    shuffled_dataset = torch.utils.data.Subset(recovery_dataset, torch.randperm(len(recovery_dataset)))
    recovery_dataloader = LightweightDataLoader(shuffled_dataset, batch_size=cfg['batch_size'])    
    
    target_model = get_model(cfg['target_model'], gamma=cfg['target_gamma'], layers=cfg['target_layers'], regularization=cfg['target_regularization'])
    recovery_model = get_model(cfg['recovery_model'], gamma=cfg['recovery_gamma'], layers=cfg['recovery_layers'])
    if not cfg['disable_wandb']:
        # compiling the models improves runtime with large reconstructions. Since it makes it harder to debug, using disable_wandb as a flag for a debug run.
        recovery_model_orig = recovery_model
        recovery_model = torch.compile(recovery_model)

    # gather the data that the target model was/is trained one
    data, labels = next(iter(train_dataloader))
    img_shape = data.shape[1:]
    original_data = data.clone().to(dtype=torch.float64, device=DEVICE)
    data = data.squeeze().flatten(1).to(dtype=torch.float64, device=DEVICE)
    

    # cfg['recover_from_latents'] = cfg['recovery_pca_dim'] > 0 or cfg['vae_path']
    cfg['recover_from_latents'] = cfg['vae_path']
    recovered_inputs, alpha_hat = init_recovery_params(cfg, data.shape[1], out_shape=1 if num_classes <= 2 else num_classes)

    if not cfg['recover_from_latents']:
        if cfg['recovery_pca_dim']:
            if cfg['dataset'] == "CIFAR5M":
                pca_dataset_name = "CIFAR10"
            else:
                pca_dataset_name = cfg['dataset']
            # again, we flip the train and test datasets
            pca_dataset = get_dataset(pca_dataset_name, max_classes=cfg['max_classes'], max_samples=50000, train=True, size=cfg['max_img_size'])[0]
            pca_dataloader = LightweightDataLoader(pca_dataset, 50000)
            # compute on PCA on CPU and limit samples to avoid GPU memory issues
            X = next(iter(pca_dataloader))[0].flatten(1).to(device="cpu")
            
            # via SVD. FIrst rows of V correspond to most important eigenvectors
            _, D, Vh = torch.linalg.svd((X - torch.mean(X, dim=0)) / sqrt(X.shape[0]), full_matrices=False)
            Vh_top = Vh[:cfg['recovery_pca_dim'],:].to(dtype=torch.float64, device=DEVICE)
            # upscale = lambda Z: Z @ Vh_top
            project = lambda Z: Z @ Vh_top.T @ Vh_top  

        optimizer = get_optimizer(cfg['optimizer'], [{"params": recovered_inputs, "lr": cfg['lr']}, {"params": alpha_hat, "lr": cfg['coefs_lr']}], lr=cfg['lr'], eps=1e-15)
    else:
        if cfg['vae_path']:
            vae_model, latent_dim = get_vae(cfg['vae_path'], img_shape)               
            
            vae_model = torch.compile(vae_model)
            num_channels = 4 if cfg['vae_path'] in ['taesd','hybridsd','taesdxl'] else 16
            # current VAE outputs pixel values in [0, 1]
            upscale = lambda Z: F.normalize(vae_model.decode(Z.view(Z.shape[0], num_channels, int(sqrt(latent_dim / num_channels)), int(sqrt(latent_dim / num_channels))).to(torch.float32)).sample,  *normalization_params(cfg['dataset'])).flatten(1).to(torch.float64)                
                                                    
            with torch.no_grad():
                recovered_latents = vae_model.encode(recovered_inputs.view(-1, *img_shape).to(torch.float32)).latents.to(torch.float64)
                recovered_latents.requires_grad = True
            optimizer = get_optimizer(cfg['optimizer'], [{"params": recovered_latents, "lr": cfg['lr']}, {"params": alpha_hat, "lr": cfg['coefs_lr']}], lr=cfg['lr'], eps=1e-15)

    alpha_hat_orig = copy.deepcopy(alpha_hat.detach())

    with torch.no_grad():
        if cfg['normalize_to_sphere']:            
            recovered_inputs /= data_norm_mean / cfg['init_std']
            data /= data_norm_mean
        else:
            data_norm_mean = 1

    # train kernel    
    labels = labels.to(dtype=torch.float64, device=DEVICE)
    if num_classes <= 2:
        labels = (labels - labels.mean()) / labels.std()
    elif labels.squeeze().ndim ==1:
        labels = scalar_to_vector_labels(labels)
    target_model.train_kernel(data, labels, task=cfg['target_task']) 
    max_lr=[cfg['lr'], cfg['coefs_lr']]
    if cfg['gamma_lr'] > 0:
        optimizer.add_param_group({"params": recovery_model.gamma, "lr": cfg['gamma_lr']})
        max_lr.append(cfg['gamma_lr'])  
    
    scheduler = get_scheduler(cfg['scheduler'], optimizer, max_lr=max_lr, total_steps=cfg['num_steps'])

    # Plot initial recovery
    num_recon_to_plot = 10 if img_shape.numel() > 3072 else 20
    num_recon_to_plot = min(num_recon_to_plot, cfg['recovery_size'])
    

    criterion = torch.nn.MSELoss()

    # train recovery
    recovered_inputs = train_recovery(recovered_inputs)

    # undo normalization
    if cfg['normalize_to_sphere']:
        with torch.no_grad():
            data *= data_norm_mean
            recovered_inputs *= data_norm_mean


    with torch.no_grad():
        if cfg['recover_from_latents']:
            recovered_inputs = upscale(recovered_latents)
    
    # avoid GPU memory issues later by moving to cpu
    recovered_images = unnormalize(recovered_inputs.detach().to("cpu").view(-1, *img_shape), *normalization_params(cfg['dataset']))
    original_data = unnormalize(original_data.detach().to("cpu"), *normalization_params(cfg['dataset']))
    recovered_images.requires_grad = False

    # visualize results
    recoveries, nearest_neighbor_indices_mse, recovery_order_mse = compute_recovery_order_MSE(original_data, recovered_images)

    print(f"Top 10 Recoveries: {recoveries[recovery_order_mse[:10]]}")
    ## Plot final recovery order by MSE
    if cfg['plot_result']:
        fig, ax =plot_by_order2(original_data, recovered_images, nearest_neighbor_indices_mse, recovery_order_mse, img_shape,num_recon_to_plot)
        wandb.log({'Top Recoveries by MSE': wandb.Image(fig)})

    q = torch.tensor([0.25, 0.5, 0.75], device=recoveries.device, dtype = recoveries.dtype)
    quantiles_mse = torch.quantile(torch.tensor(recoveries, device=q.device, dtype=q.dtype), q)
    wandb.log({"L2 25th Percentile": quantiles_mse[0], "L2 50th Percentile": quantiles_mse[1], "L2 75th Percentile": quantiles_mse[2], 
               "Unique Recons MSE": len(set(nearest_neighbor_indices_mse.cpu().numpy()))})

    if cfg['plot_all_recons']:
        fig, ax = plot_all_by_order(original_data, recovered_images, nearest_neighbor_indices_mse, recovery_order_mse, img_shape,compute_num_plots(cfg['recovery_size']))
        wandb.log({'ALL recoveries by MSE': wandb.Image(fig)})

    
    
    nearest_neighbor_indices_ssim=[]
    ssims_best,nearest_neighbor_indices_ssim=get_ssim_all(recovered_images.detach(), original_data.detach()).max(dim=1)  
    recovery_order_ssim = torch.argsort(ssims_best,descending=True)
    print(f"Top 10 Recoveries: {ssims_best[recovery_order_ssim[:num_recon_to_plot]]}")
    ## Plot final recovery prder by SSIM
    if cfg['plot_result']:
        fig, ax = plot_by_order2(original_data, recovered_images, nearest_neighbor_indices_ssim, recovery_order_ssim, img_shape,num_recon_to_plot)
        wandb.log({'Top Recoveries by SSIM': wandb.Image(fig)})
        #blocking method

    unique_recons_ssim, unique_recons_mse = len(set(nearest_neighbor_indices_ssim.cpu().numpy())), len(set(nearest_neighbor_indices_mse))
    print('recons MSE<5:',(recoveries<5).sum(),'recons SSIM>0.4:' ,((ssims_best)>0.4).sum().item(),'unique recons SSIM:',unique_recons_ssim,'unique recons MSE:', unique_recons_mse)
    print('order by MSE, Avg SSIM: {:.4f}, Avg MSE: {:.4f}'.format(ssims_best.mean(), recoveries.mean()))
    print('order by MSE, median SSIM: {:.4f}, median MSE: {:.4f}'.format(ssims_best.median(), recoveries.median()))
    wandb.log({"Recons SSIM>0.4":(ssims_best>0.4).sum().item(),"Unique recons SSIM":len(set(nearest_neighbor_indices_ssim.cpu().numpy()))})
    dssim = (1-ssims_best) / 2
    quantiles_dssim = torch.quantile(dssim, q)
    wandb.log({"DSSIM 25th Percentile": quantiles_dssim[0], "DSSIM 50th Percentile": quantiles_dssim[1], "DSSIM 75th Percentile": quantiles_dssim[2]})

    try:
        print("recovered gamma: ",recovery_model.gamma, "target gamma: ", cfg['target_gamma'])
    except AttributeError:
        pass
        
    # Save reconstruction results and metrics
    unique_recons_ssim, unique_recons_mse = save_reconstruction_results(
        cfg, recovered_images, original_data, img_shape, recoveries, 
        nearest_neighbor_indices_mse, nearest_neighbor_indices_ssim, 
        ssims_best, target_model, alpha_hat, dssim,recovery_order_ssim
    )
    
    wandb.finish()
    plt.close()
    print("DONE!")