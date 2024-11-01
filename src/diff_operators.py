import torch
from torch.autograd import grad
import numpy as np

def hessian(y, x):
    g = gradient(y, x).squeeze(0)
    h0 = gradient(g[:,0], x).squeeze(0)[:,None,:]
    h1 = gradient(g[:,1], x).squeeze(0)[:,None,:]
    h2 = gradient(g[:,2], x).squeeze(0)[:,None,:]
    h = torch.cat((h0,h1,h2), dim=1)
    return h.unsqueeze(0)


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True, allow_unused=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    gradient = grad(y, x, grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0]
    return gradient

def jacobian(y, x):
    ''' jacobian of y wrt x '''
    meta_batch_size, num_observations = y.shape[:2]
    jac = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1]).to(y.device) # (meta_batch_size*num_points, 2, 2)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[...,i].view(-1, 1)
        jac[:, :, i, :] = grad(y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]

    return jac




