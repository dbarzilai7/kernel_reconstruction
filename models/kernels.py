import torch
from torch.func import jacrev, functional_call
import torch.nn as nn
from math import sqrt
import numpy as np
import torch.optim as optim
from models.model_utils import *
from utils import *

eps = 1e-15
dtype = torch.float64

class KernelModule(nn.Module):
    def __init__(self, normalize_to_sphere=False, regularization=0, **kwargs):
        super().__init__()
        self.alpha = None
        self.train_x = None
        self.regularization = regularization
        self.normalize_to_sphere = normalize_to_sphere

    def _compute_kernel(self, X, Y=None):
        raise NotImplementedError

    def compute_kernel(self, X, Y=None):
        if Y is None:
            Y = X

        if self.normalize_to_sphere:
            x_norm = torch.norm(X, dim=1)
            y_norm = torch.norm(Y, dim=1)
            X = X / x_norm[:, None]
            Y = Y / y_norm[:, None]

        kernel = self._compute_kernel(X, Y)

        if self.normalize_to_sphere:
            # undo normalization
            kernel = kernel * (x_norm[:, None] * y_norm[None, :])

        return kernel

    def train_kernel(self, X, labels,  task='krr'):
        kernel = self.compute_kernel(X)
        if self.regularization > 0:
            kernel += torch.eye(X.shape[0]).to(dtype=dtype, device=X.device) * (X.shape[0] * self.regularization)
        if task == 'krr':
            self.alpha = torch.linalg.solve(kernel, labels)
        elif task == 'svm':
            labels = nn.functional.one_hot(torch.argmax(labels, dim=1))
            self.gd_trainer(X, labels, target_task="svm", epochs=100000)
        elif task == 'svm_ovr':
            #one vs rest, essentially C (number of classes) independent SVMs of class i vs class not i
            labels = torch.sign(labels)
            self.gd_trainer(X, labels, target_task="svm_ovr", epochs=100000)
        elif task == 'logistic':
            self.gd_trainer(X, labels, target_task="logistic", epochs=1000)
        elif task == 'feature_space_krr':
            self.alpha = torch.linalg.solve(kernel, labels)
            features = self.feature_mapping(X)
            #w^i = \sum_j \alpha_i,j \phi(x_j)
            self.w_star = (features.T @ self.alpha)      
        else:
            ValueError("Invalid task")
        self.train_x = X
    
    def gd_trainer(self, X, labels, target_task="svm", epochs=5000, verbose=True):
        self.alpha = torch.zeros(X.shape[0], labels.shape[1], device=X.device, dtype=dtype)
        self.alpha.requires_grad = True
        binary_classification = self.alpha.shape[1] == 1
        if target_task == 'svm':
            if binary_classification:
                loss_fn = lambda y_1, y_2: torch.mean(torch.clamp(1 - y_1 * y_2, min=0))
            else:
                loss_fn = MultiClassHingeLoss()
        elif target_task == 'svm_ovr':
            loss_fn = lambda y_1, y_2: torch.mean(torch.clamp(1 - y_1 * y_2, min=0))
        elif target_task == 'krr':
            loss_fn = nn.MSELoss()
        elif target_task == 'logistic':
            if binary_classification:
                loss_fn = nn.SoftMarginLoss()
            else:
                loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD([self.alpha], lr=1e-2)
        scheduler = get_scheduler("onecycle", optimizer, total_steps=epochs, max_lr=1e-2)

        with torch.no_grad():
            kernel = self.compute_kernel(X)

        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()

            outputs = kernel @ self.alpha
            loss = loss_fn(outputs.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            if verbose and epoch % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
        self.alpha = self.alpha.detach()
        self.alpha.requires_grad = False


    def forward(self, test_x, train_x=None, alpha=None):
        alpha = self.alpha if alpha is None else alpha
        train_x = train_x if self.train_x is None else self.train_x
        if alpha is None:
            raise ValueError("Must either pass an alpha or call train_kernel before forward")
        return self.compute_kernel(test_x, train_x) @ alpha

class LaplaceKernel(KernelModule):
    def __init__(self, gamma=None, normalize_to_sphere=False,a=None,b=None, **kwargs):
        super().__init__(normalize_to_sphere=normalize_to_sphere, **kwargs)
        self.gamma = nn.Parameter(torch.tensor(gamma if gamma is not None else 1.0))
        self.a = nn.Parameter(torch.tensor(a if a is not None else 0.0))
        self.b = nn.Parameter(torch.tensor(b if b is not None else 1.0))

    def _compute_kernel(self, X, Y=None):
        gamma = 1 / X.shape[1] if self.gamma is None else self.gamma
        norms = torch.cdist(X, Y, p=2)
        K = self.a + self.b*torch.exp(-gamma * norms)
        return K


class GaussianKernel(KernelModule):
    def __init__(self, gamma=None, normalize_to_sphere=False, **kwargs):
        super().__init__(normalize_to_sphere=normalize_to_sphere, **kwargs)
        self.gamma = nn.Parameter(torch.tensor(gamma if gamma is not None else 1.0))

    def _compute_kernel(self, X, Y=None):
        gamma = 1 / X.shape[1] if self.gamma is None else self.gamma
        norms = torch.cdist(X, Y, p=2) ** 2
        K = torch.exp(-gamma * norms)
        return K


class GaussianRFFKernel(KernelModule):
    def __init__(self, gamma=None, normalize_to_sphere=False, num_features=40000, input_dimension=3072, **kwargs):
        super().__init__(normalize_to_sphere=normalize_to_sphere, **kwargs)
        self.gamma = nn.Parameter(torch.tensor(gamma if gamma is not None else 1.0))
        self.num_features = num_features
        self.rff = torch.randn(num_features, input_dimension, device=DEVICE, dtype=torch.float64)
        self.biases = torch.rand(num_features, device=DEVICE, dtype=torch.float64) * 2 * np.pi

    def _compute_kernel(self, X, Y=None):
        X_proj = torch.cos(X @ self.rff.T * torch.sqrt(self.gamma) + self.biases[None, :])
        Y_proj = torch.cos(Y @ self.rff.T * torch.sqrt(self.gamma) + self.biases[None, :])
        K = X_proj @ Y_proj.T * (2 / self.num_features)
        return K


class NTKAnalytic(KernelModule):
    def __init__(self, layers=2,**kwargs):
        super().__init__(normalize_to_sphere=True, **kwargs)
        self.layers = layers

    def _compute_kernel(self, X, Y=None):
        ntk = gpk = X @ Y.T
        for l in range(self.layers - 1):
            k0 = kappa0(gpk)
            gpk = kappa1(gpk)
            ntk = ntk * k0 + gpk

        return ntk / self.layers

class GPKAnalytic(KernelModule):
    def __init__(self, layers=2, **kwargs):
        super().__init__(normalize_to_sphere=True, **kwargs)
        self.layers = layers

    def _compute_kernel(self, X, Y=None):
        gpk = X @ Y.T
        for l in range(self.layers - 1):
            gpk = kappa1(gpk)

        return gpk


class PolyKernel(KernelModule):
    def __init__(self, gamma=None, normalize_to_sphere=False, degree=3, c=1.0, **kwargs):
        super().__init__(normalize_to_sphere=normalize_to_sphere, **kwargs)
        self.gamma = gamma
        self.degree = degree
        self.c = c
        self.mean_norm = None
        
    def _compute_kernel(self, X, Y=None):
        return (self.c + X @ Y.T * self.gamma) ** self.degree


def kappa1(u):
    clipped = torch.clip(u, -1+eps, 1-eps)    
    return (clipped * (torch.pi - torch.arccos(clipped)) + torch.sqrt(1 - clipped ** 2)) / torch.pi
    # return ((clipped * (torch.pi - torch.arccos(clipped)) + torch.sqrt(1 - clipped ** 2)) / torch.pi - 1/ sqrt(np.pi)) / (1 - 1/ sqrt(np.pi))


def kappa0(u):
    clipped = torch.clip(u, -1+eps, 1-eps)
    return (torch.pi - torch.arccos(clipped)) / torch.pi
    # return 2 * ((torch.pi - torch.arccos(clipped)) / torch.pi - 0.5)
