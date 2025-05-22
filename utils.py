from bisect import bisect_right
import json
from pathlib import Path
from typing import List
from matplotlib import pyplot as plt
import numpy as np
import torch
import wandb
from plot_utils import *
from torch import optim, nn
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, LambdaLR, StepLR, CyclicLR, ReduceLROnPlateau
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur, normalize
import kornia
from kornia.metrics import ssim as kornia_ssim
import os


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42
px = 1/plt.rcParams['figure.dpi']
project_dict = {'CIFAR10_v_a': 'CIFAR_v_a-recovery','CIFAR5M_v_a':'CIFAR_v_a-recovery', 'MNIST_odd_even': 'MNIST_odd_even-recovery','celebA':'celebA-recovery','CIFAR10': 'CIFAR10-recovery','CIFAR5M': 'CIFAR10-recovery','CIFAR100':'CIFAR100-recovery'}


def tv_loss(imgs, weight):
    num_imgs = imgs.shape[0]
    tv_h = ((imgs[:,:,1:,:] - imgs[:,:,:-1,:]).pow(2)).sum()
    tv_w = ((imgs[:,:,:,1:] - imgs[:,:,:,:-1]).pow(2)).sum()
    return weight * (tv_h + tv_w) / num_imgs


def high_freq_loss(imgs, weight):
    blurred = gaussian_blur(imgs, kernel_size=3)
    high_freq = imgs - blurred

    return (weight * torch.sum((high_freq ** 2).flatten(1), dim=1)).mean()


def unit_sphere(n, d):
    x = torch.randn(n, d).to(torch.float64)
    x /= torch.norm(x, dim=1).reshape(-1, 1)
    return x


def uniform_cube(n, d):
    x = torch.rand(n, d).to(torch.float64)
    return x



def wandb_init(cfg):
    mode = 'disabled' if cfg['disable_wandb'] else 'online'
    if cfg['wandb_project'] is None:
        cfg['wandb_project'] = project_dict[cfg['dataset'] ]

    print(f"wandb project: {cfg['wandb_project']},"f"wandb entity: {cfg['wandb_entity']}")
    wandb.init(project=cfg['wandb_project'], config=cfg, entity=cfg['wandb_entity'],
               mode=mode, notes=cfg['wandb_notes'])
    return wandb.config


def get_optimizer(optimizer_name, parameters, lr, **kwargs):
    if optimizer_name == 'SGD':
        return optim.SGD(parameters, lr=lr)
    elif optimizer_name == 'Adam':
        return optim.Adam(parameters, lr=lr, betas=(0.9, 0.99), **kwargs)    
    else:
        raise ValueError("Unsupported optimizer")


def get_criterion(criterion):
    if criterion == 'CrossEntropy':
        return nn.CrossEntropyLoss()
    elif criterion == 'MSE':
        return nn.MSELoss()
    else:
        raise ValueError("Unsupported criterion")


def get_scheduler(scheduler_name, optimizer, **kwargs):
    if scheduler_name == 'cosine':
        return CosineAnnealingLR(optimizer,eta_min=1e-5,T_max=kwargs['total_steps'])
    elif scheduler_name == 'onecycle':        
        return OneCycleLR(optimizer,pct_start=0.15,div_factor=10,final_div_factor=1e2,three_phase=True, **kwargs)            
    elif scheduler_name == 'cyclic':
        return CyclicLR(optimizer,base_lr=1e-4,step_size_up=10000,step_size_down=10000,mode='triangular2', max_lr=kwargs['max_lr'])   
    elif scheduler_name == 'plateau':
        return ReduceLROnPlateau(optimizer,factor=0.5,patience=250,threshold=0.0001,threshold_mode='rel',cooldown=200,min_lr=1e-4,eps=1e-08,verbose=True)       
    elif scheduler_name == 'none':          
        return LambdaLR(optimizer, [lambda x: 1 for _ in kwargs['max_lr']])
    else:
        return None


def augment_data(cfg,recovery_train_data):
    augmentations=[]                
    if cfg['horiz_flip']:
        augmentations += [kornia.augmentation.RandomHorizontalFlip(p=1)(recovery_train_data)]
    if cfg['vert_flip']:
        augmentations += [ kornia.augmentation.RandomVerticalFlip(p=1)(recovery_train_data)]
    if cfg['vert_horiz_flip']:
        augmentations += [ kornia.augmentation.RandomHorizontalFlip(p=1)(kornia.augmentation.RandomVerticalFlip(p=1)(recovery_train_data))]   

    if len(augmentations)>0:
        recovery_train_data = torch.cat([recovery_train_data]+augmentations)
    return recovery_train_data

def compute_num_plots(recovery_size):
   for i in range(25,0,-1):
        if recovery_size % i == 0:
            return i

def get_ssim_pairs_kornia(x, y):
    return kornia_ssim(x, y, window_size=3).reshape(x.shape[0], -1).mean(dim=1)

@torch.no_grad()
def get_ssim_all(x, y):    
    ssims = []
    for i in range(y.shape[0]):
        scores = get_ssim_pairs_kornia(x, y[i:i + 1].expand(x.shape[0], -1, -1, -1))
        ssims.append(scores)
    return torch.stack(ssims).t()
   

def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor

def path_to_ckpt_path(path):
        path = Path(path)
        return path.with_suffix('.pt')
    
def path_to_ckpt_path_epoch(path,epoch):
        path = Path(path)
        return path.with_suffix(f'_{epoch}.pt')

def path_to_json_path(path):
        path = Path(path)
        return path.with_suffix('.json')

def scalar_to_vector_labels(labels):
    one_hot = torch.nn.functional.one_hot(labels.to(torch.long)).to(device=labels.device, dtype=labels.dtype) 
    # one_hot = (one_hot - 0.5) / 5
    one_hot = one_hot - 1 / one_hot.shape[-1]
    return one_hot / one_hot[0].std()
    # return one_hot

def unnormalize(data, orig_means, orig_stds):
    """
    reverses Torchvision's Normalize
    """
    return normalize(data, mean=[-m / s for m, s in zip(orig_means, orig_stds)], std=[1.0 / s for s in orig_stds])

class MultiClassHingeLoss(nn.Module):
    def __init__(self):
        super(MultiClassHingeLoss, self).__init__()

    def forward(self, output, target):
        # output: (batch_size, num_classes)
        # target: (batch_size, num_classes) should be a one hot vector with 1 in the correct class
        
        output_diffs = output - output[target.to(bool)].unsqueeze(1)
        output_diffs[target.to(bool)] = -float('inf')
        return torch.mean(torch.clamp(1 + torch.max(output_diffs, dim=1).values, min=0))

def save_checkpoint(directory, step, img_shape, data_norm_mean, min_value, max_value, target_model, recovery_model, recoveries, alpha_hat, alpha_hat_orig, optimizer, scheduler):
    if not os.path.exists(directory):
        os.makedirs(directory)
    checkpoint = {
        'step': step,
        'img_shape': img_shape,
        'data_norm_mean': data_norm_mean,
        'max_value': max_value,
        'min_value': min_value, 
        'target_model': target_model,
        'recovery_model': recovery_model,
        'recoveries': recoveries,
        'alpha_hat': alpha_hat,  
        'alpha_hat_orig': alpha_hat_orig,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler
    }
    torch.save(checkpoint, os.path.join(directory, 'checkpoint_last.pt'))

def load_checkpoint(directory):
    if directory is None or not os.path.exists(directory):
        return None
    checkpoints = [f for f in os.listdir(directory) if f.startswith('checkpoint_')]
    if not checkpoints:
        return None    
    checkpoint = torch.load(os.path.join(directory, 'checkpoint_last.pt'))
    return checkpoint

def compute_recovery_order_MSE(data, recovered_inputs):
    # compute the nearest neighbor distance for each image in the recovered_inputs
    # returns the distances, the indices of the nearest neighbors and the order of the recoveries
    # based on the MSE
    # data: torch.Tensor, the original data
    # recovered_inputs: torch.Tensor, the recovered inputs
    # Returns: recoveries, nearest_neighbor_indices, recovery_order
    norms = torch.cdist(recovered_inputs.flatten(1), data.flatten(1), p=2)
    recoveries, nearest_neighbor_indices_mse = norms.min(dim=1)
    recovery_order_mse = torch.argsort(recoveries)
    return recoveries, nearest_neighbor_indices_mse, recovery_order_mse


def save_reconstruction_results(cfg, recovered_images, original_data, img_shape, recoveries, nearest_neighbor_indices_mse, nearest_neighbor_indices_ssim, ssims_best, target_model, alpha_hat, dssim,recovery_order_ssim):
    """Save reconstruction results to disk and plot additional visualizations."""
    unique_recons_ssim, unique_recons_mse = len(set(nearest_neighbor_indices_ssim.cpu().numpy())), len(set(nearest_neighbor_indices_mse))
    
    # Plot all reconstructions if requested
    if cfg['plot_all_recons']:
        fig, ax = plot_all_by_order(original_data, recovered_images, nearest_neighbor_indices_ssim, recovery_order_ssim, img_shape, compute_num_plots(cfg['recovery_size']))
        wandb.log({'ALL recoveries by SSIM': wandb.Image(fig)})
    
    # Create output directory if needed
    if (cfg['save_recons'] or cfg['save_metrics']) and not os.path.exists(cfg["output_dir"]):
        os.makedirs(cfg['output_dir'])
    
    # Save reconstructions if requested
    if cfg['save_recons']:
        reconstructions = recovered_images.reshape(-1, *img_shape).cpu().numpy()
        trainig_data = original_data.reshape(-1, *img_shape).cpu().numpy()
        np.savez(f'{cfg["output_dir"]}/{wandb.run.name}_results.npz', reconstructions=reconstructions, trainig_data=trainig_data)
    
    # Plot SSIM vs alpha if requested
    if cfg['plot_ssim_vs_alpha']:
        ssims_best_gt_data_to_recon, nearest_neighbor_indices_ssim_gt_data_to_recon = get_ssim_all(original_data.detach(), recovered_images.detach()).max(dim=1)
        fig, ax = scatter_plot_alpha_vs_ssim(target_model.alpha.norm(dim=1).cpu().detach().numpy(), ssims_best_gt_data_to_recon.cpu().detach().numpy())
        wandb.log({'SSIM vs. alpha': wandb.Image(fig)})
    
    # Save metrics if requested
    if cfg['save_metrics']:
        save_path = f'{cfg["output_dir"]}/{wandb.run.name}_info.json'
        run_data = dict(cfg)  # detach from wandb
        run_data['wandb_run_name'] = wandb.run.name
        run_data['MSEs'] = recoveries.cpu().detach().tolist()
        run_data['DSSIMs'] = dssim.cpu().detach().tolist()
        run_data['MSE NNs'] = nearest_neighbor_indices_mse.cpu().detach().tolist()
        run_data['DSSIM NNs'] = nearest_neighbor_indices_ssim.cpu().detach().tolist()
        run_data['Unique Recons SSIM'] = unique_recons_ssim
        run_data['Unique Recons MSE'] = unique_recons_mse        
        run_data['Target Alpha'] = target_model.alpha.cpu().detach().numpy().tolist()
        run_data['Recovery Alpha'] = alpha_hat.cpu().detach().numpy().tolist()
        run_data['Params'] = recovered_images.numel() + alpha_hat.numel()
        with open(save_path, "w") as f:
            json.dump(run_data, f)
            
    return unique_recons_ssim, unique_recons_mse

def log_recovery_progress(step, loss, recoveries, mean_recovery, top_recovery, scheduler, recovered_inputs, img_shape):
    """Log recovery progress by creating visualizations and logging statistics."""
    print(f"Epoch: {step}, Loss: {loss.item()}, "
          f"Recovery Error: {mean_recovery.item()}, Best Recovery: {top_recovery.item()}, "
          f"LR:{scheduler.get_last_lr()}")
    # plot some arbitrary pictures
    num_plots = min(10, recovered_inputs.shape[0])
    fig, ax = plt.subplots(1, num_plots, figsize=(10, 5))
    for i in range(num_plots):
        ax[i].imshow(tensor_to_image(recovered_inputs[i], img_shape))
    log_dict = {'Train Progress': wandb.Image(fig), 'Recovery Error': mean_recovery.item(), 'Best Recovery': top_recovery.item()}
    plt.close()
    return log_dict