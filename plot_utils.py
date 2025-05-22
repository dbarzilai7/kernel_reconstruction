import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np


def tensor_to_image(tensor, shape=None, min_val=None, max_val=None, numpy=True):
    tensor = tensor.squeeze().clone().detach().cpu()
    if shape is not None:
        tensor = tensor.view(shape)
    if min_val is None:
        min_val = tensor.min()
    if max_val is None:
        max_val = tensor.max()
    tensor = ((tensor - min_val) / (max_val - min_val)).clip(0,1) # unnormalize

    if numpy:
        tensor = tensor.movedim(-3, -1).numpy()
    return tensor

def save_reconstructions_as_png(recovered_inputs, data, nearest_neighbor_indices,psnrs,ssims,img_shape=(3,32,32), output_dir='reconstructions'):
    ## Save the reconstructions as PNG files
    # Args:
    # recovered_inputs: torch.Tensor, the recovered inputs
    # data: torch.Tensor, the original data
    # nearest_neighbor_indices: list, the indices of the nearest neighbors
    # output_dir: str, the directory to save the images
    # Returns: None
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, idx in enumerate(nearest_neighbor_indices):
        fig, axes = plt.subplots(1, 2, figsize=(5, 5))
        reconstruction = tensor_to_image(recovered_inputs[i],img_shape)
        original = tensor_to_image(data[idx],img_shape)
        # Plot the recovered input
        axes[0].imshow(reconstruction)
        axes[0].set_title('Reconstruction')
        axes[0].axis('off')
        
        # Plot the nearest neighbor
        axes[1].imshow(original)
        axes[1].set_title('Nearest Neighbor')
        axes[1].axis('off')

        # Calculate MSE
        mse = F.mse_loss(torch.tensor(reconstruction), torch.tensor(original),reduction='sum').item()
        
        # Calculate PSNR
        psnr_value = psnrs[i]
        
        # Calculate SSIM
        ssim_value = ssims[i]
        
               
        
        # Add MSE to the plot
        fig.suptitle(f'MSE: {mse:.4f}, PSNR: {psnr_value:.4f}, SSIM:{ssim_value:.4f}', fontsize=12)
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, f'reconstruction_{i}.png'))
        plt.close(fig)

def plot_by_order(data, recovered_inputs, nearest_neighbor_indices, recovery_order, img_shape,num_plots=5):
    ## Plot the images in the order of recovery
    # Args: 
    # data: torch.Tensor, the original data
    # recovered_inputs: torch.Tensor, the recovered inputs
    # nearest_neighbor_indices: list, the indices of the nearest neighbors
    # recovery_order: list, the order of recovery
    # img_shape: tuple, the shape of the images
    # Returns: fig, ax: the figure and axes of the plot
    fig, ax = plt.subplots(2, num_plots, figsize=(10, 5))
    for i in range(num_plots):
        ax[0][i].imshow(tensor_to_image(recovered_inputs[recovery_order[i]], img_shape))
        ax[1][i].imshow(tensor_to_image(data[nearest_neighbor_indices[recovery_order[i]]], img_shape))
    return fig, ax



def plot_by_order2(data, recovered_inputs, nearest_neighbor_indices, recovery_order, img_shape, num_plots=5):
    fig, axes = plt.subplots(2, num_plots, figsize=(num_plots * 2, 4))

    for i in range(num_plots):
        # Plot the first line of images
        axes[0, i].imshow(tensor_to_image(recovered_inputs[recovery_order[i]], img_shape))
        axes[0, i].axis('off')

        # Plot the second line of images directly below the first line
        axes[1, i].imshow(tensor_to_image(data[nearest_neighbor_indices[recovery_order[i]]], img_shape))
        axes[1, i].axis('off')


    plt.tight_layout()
    return fig, axes

def plot_all_by_order(data, recovered_inputs, nearest_neighbor_indices, recovery_order, img_shape, num_plots=5):
    px = 1/plt.rcParams['figure.dpi']
    num_rows = int(len(recovery_order)//num_plots) + 1*(len(recovery_order)%num_plots>0)
    height_ratios = [1 if i%3 != 2 else 0.2 for i in range(3 * num_rows)]
    fig, axes = plt.subplots(3*num_rows , num_plots, 
                             figsize=(num_plots * 2, 4.4*num_rows),
                            #  figsize=(num_plots * (img_shape[0]*px+2), num_rows*2*(img_shape[1]*px+2)), 
                             gridspec_kw = {'height_ratios':height_ratios})

    for j in range(num_rows):
        for i in range(num_plots):
            # Plot the first line of images
            axes[3*j, i].imshow(tensor_to_image(recovered_inputs[recovery_order[num_plots*j+i]], img_shape))
            axes[3*j, i].axis('off')

            # Plot the second line of images directly below the first line
            axes[3*j+1, i].imshow(tensor_to_image(data[nearest_neighbor_indices[recovery_order[num_plots*j+i]]], img_shape))
            axes[3*j+1, i].axis('off')

            axes[3*j+2, i].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)  # Remove space between plots
    plt.tight_layout()
    return fig, axes

def plot_by_order_interval(data, recovered_inputs, nearest_neighbor_indices, recovery_order, img_shape, num_plots=5,interlval=6):
    recovery_order = recovery_order[::interlval]
    # nearest_neighbor_indices = nearest_neighbor_indices[recovery_order]
    px = 1/plt.rcParams['figure.dpi']
    num_rows = int(len(recovery_order)//(num_plots))
    fig, axes = plt.subplots(2*num_rows, num_plots, figsize=((num_plots) * 2*img_shape[0]*px, img_shape[1]*num_rows*px*2))

    for j in range(num_rows):
        for i in range(num_plots):
            # Plot the first line of images
            axes[2*j, i].imshow(tensor_to_image(recovered_inputs[recovery_order[num_plots*j+i]], img_shape))
            axes[2*j, i].axis('off')

            # Plot the second line of images directly below the first line
            axes[2*j+1, i].imshow(tensor_to_image(data[nearest_neighbor_indices[recovery_order[num_plots*j+i]]], img_shape))
            axes[2*j+1, i].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)  # Remove space between plots
    plt.tight_layout()
    return fig, axes

def scatter_plot_alpha_vs_ssim(alpha_values, ssim_values):
    fig, ax = plt.subplots()
    ax.scatter(alpha_values, ssim_values)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('SSIM')
    return fig, ax