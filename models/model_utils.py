import json
import os
import torch

from .kernels import *
from pathlib import Path
from utils import DEVICE, path_to_ckpt_path, path_to_json_path
from math import ceil

MODEL_LIST = KERNEL_MODELS = ['laplace', 'gaussian', 'ntk_analytic', 'ntk_empirical', 'gpk_analytic', 'poly_kernel', 'gaussian_rff']
HUGGINGFACE_VAES = {"taesd": "madebyollin/taesd", "sd-vae": "stabilityai/sd-vae-ft-mse",'taesd3':'madebyollin/taesd3','taef':'madebyollin/taef1',
                    'hybridsd':'cqyan/hybrid-sd-tinyvae','taesdxl':'madebyollin/taesdxl','hybridsdxl':'cqyan/hybrid-sd-small-vae-xl',"sd-vae": "stabilityai/sd-vae-ft-mse"}

def get_model(model_name, **kwargs):
    if model_name == 'laplace':
        return LaplaceKernel(**kwargs)
    elif model_name == "gaussian":
        return GaussianKernel(**kwargs)
    elif model_name == 'ntk_analytic':
        return NTKAnalytic(**kwargs)
    elif model_name == 'gpk_analytic':
        return GPKAnalytic(**kwargs)
    elif model_name == 'poly_kernel':
        return PolyKernel(**kwargs)
    elif model_name == 'gaussian_rff':
        return GaussianRFFKernel(**kwargs) 
    raise NotImplementedError


def get_vae(vae_path, img_shape):
    if vae_path in HUGGINGFACE_VAES:
        import diffusers # requires the diffuesers library from huggingface
        if vae_path in["taesd",'taesd3','taef','hybridsd','taesdxl']:
            vae_model = diffusers.AutoencoderTiny.from_pretrained(HUGGINGFACE_VAES[vae_path]).to(DEVICE)
            latent_size= 4 * int(ceil(img_shape[-2] / 8) * ceil(img_shape[-1] / 8)) if vae_path in ['taesd','hybridsd','hybridsdxls','taesdxl'] else 16 * int(ceil(img_shape[-2] / 8) * ceil(img_shape[-1] / 8))

        else:
            vae_model = diffusers.AutoencoderKL.from_pretrained(HUGGINGFACE_VAES[vae_path]).to(DEVICE)
            latent_size = 4 * int(ceil(img_shape[-2] / 8) * ceil(img_shape[-1] / 8))
        vae_model.eval()
        return vae_model, latent_size # at least for taesd, latent dim is 4 x (width // 8) x (height // 8)
    else:
        raise ValueError("Bad VAE model")



def save_model(model, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
