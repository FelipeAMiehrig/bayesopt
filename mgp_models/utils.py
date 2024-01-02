import torch
from torch import Tensor
import math

def convert_bounds(tensor: Tensor, bounds, dim: int)-> Tensor:

    if bounds is not None:
        for i in range(dim):
            a, b = bounds[i]
            tensor[:, i] = a + (b - a) * tensor[:, i]
    return tensor

def get_candidate_pool(dim, bounds, size=1000):
    U = torch.rand(size, dim)
    U = convert_bounds(U, bounds, dim)
    return U


def get_test_set(synthetic_function, dim, bounds, size=1000):
    X_test = torch.rand(size, dim)
    X_test = convert_bounds(X_test, bounds, dim)
    Y_test = synthetic_function(X_test)
    return X_test, Y_test

def get_acq_values_pool(acq_function, candidates):
    acq_values = acq_function(candidates)
    best_index = torch.argmax(acq_values)
    candidates = candidates[best_index].unsqueeze(-1)
    return candidates.T, acq_values[best_index].unsqueeze(-1)


def eval_rmse(gp, test_X, test_Y, ll=None):
    posterior = gp.posterior(test_X, ll=ll)
    if ll is not None:
        Y_hat = posterior.best_mixture_mean
    else:
        Y_hat = posterior.mixture_mean
    return torch.sqrt(torch.mean((Y_hat-test_Y)**2))

def eval_nll(gp, test_X, test_Y, ll=None, variance=1.0):
    posterior = gp.posterior(test_X, ll=ll)
    if ll is not None:
        Y_hat = posterior.best_mixture_mean
    else:
        Y_hat = posterior.mixture_mean
    sigma2 = torch.std(Y_hat).pow(2)
    p1 = 0.5*torch.log(2*math.pi*sigma2).mul(Y_hat.shape[1])
    p2 = torch.sub(test_Y, Y_hat).pow(2).sum().div(2*sigma2)
    return p1+p2
    #sigma2 = posterior.best_mixture_variance
    #p1 = 0.5*torch.log(2*math.pi*sigma2).sum()
    #p2 = torch.sub(test_Y, Y_hat).pow(2).div(2*sigma2).sum()
    #return p1+p2