import torch
from torch import Tensor
import math

def convert_bounds(tensor: Tensor, bounds, dim: int)-> Tensor:
    tensor_scaled = tensor.detach().clone()
    if bounds is not None:
        for i in range(dim):
            a, b = bounds[i]
            tensor_scaled[:, i] = a + (b - a) * tensor[:, i]
    return tensor_scaled

def get_candidate_pool(dim, bounds, size=1000):
    U = torch.rand(size, dim)
    #U = convert_bounds(U, bounds, dim)
    return U


def get_test_set(synthetic_function, dim, bounds, size=1000):
    X_test = torch.rand(size, dim)
    X_test_scaled = convert_bounds(X_test, bounds, dim)
    Y_test = synthetic_function(X_test_scaled)
    return X_test, Y_test

def get_acq_values_pool(acq_function, candidates):
    acq_values = acq_function(candidates)
    best_index = torch.argmax(acq_values)
    candidates = candidates[best_index].unsqueeze(-1)
    return candidates.T, acq_values[best_index].unsqueeze(-1)


def eval_rmse(gp, test_X, test_Y, ll=None, batch_size = 100):
    """
    Evaluate the Root Mean Square Error (RMSE) for a Gaussian Process model in batches.

    Parameters:
    gp (GaussianProcess): The Gaussian Process model.
    test_X (Tensor): The test inputs.
    test_Y (Tensor): The test targets.
    ll (Optional): Log likelihood function. Default is None.

    Returns:
    float: The calculated RMSE.
    """
    total_batches = test_X.size(0) // batch_size
    rmse_accum = 0.0

    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_X = test_X[start_idx:end_idx]
        batch_Y = test_Y[start_idx:end_idx]

        posterior = gp.posterior(batch_X, ll=ll)
        Y_hat = posterior.best_mixture_mean if ll is not None else posterior.mixture_mean
        rmse_accum += torch.sqrt(torch.mean((Y_hat - batch_Y) ** 2)).item()

    # Calculate average RMSE over all batches
    rmse = rmse_accum / total_batches
    return rmse




# def eval_rmse(gp, test_X, test_Y, ll=None):
#     print('trying eval rmse')
#     posterior = gp.posterior(test_X, ll=ll)
#     print('computed posterior')
#     if ll is not None:
#         Y_hat = posterior.best_mixture_mean
#     else:
#         Y_hat = posterior.mixture_mean
#     print('computed mixture mean')
#     return torch.sqrt(torch.mean((Y_hat-test_Y)**2))


def eval_nll(gp, test_X, test_Y, tkwargs, ll=None, batch_size = 100, eps=1e-6):
    """
    Evaluate the Negative Log Likelihood (NLL) for a Gaussian Process model in batches.

    Parameters:
    gp (GaussianProcess): The Gaussian Process model.
    test_X (Tensor): The test inputs.
    test_Y (Tensor): The test targets.
    ll (Optional): Log likelihood function. Default is None.
    variance (float): The variance used in NLL calculation. Default is 1.0.

    Returns:
    float: The calculated NLL.
    """
    total_batches = test_X.size(0) // batch_size
    nll_accum = 0.0
    Y_full = torch.Tensor().to(**tkwargs)
    var_full =torch.Tensor().to(**tkwargs)
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_X = test_X[start_idx:end_idx]
        batch_Y = test_Y[start_idx:end_idx]

        posterior = gp.posterior(batch_X, ll=ll)
        Y_hat = posterior.best_mixture_mean if ll is not None else posterior.mixture_mean
        var_hat=  posterior.best_mixture_variance if ll is not None else posterior.mixture_variance
        Y_full, var_full  = torch.cat((Y_full, Y_hat),0), torch.cat((var_full, var_hat),0)
    eps_tensor = torch.Tensor([eps]).repeat(var_full.size()[0]).unsqueeze(-1).to(**tkwargs)
    concated = torch.cat((var_full, eps_tensor),1)
    sigma2 = torch.max(concated, 1)[0]
    p1 = torch.log(2 * math.pi * sigma2)
    p2 = torch.sub(test_Y, Y_full).pow(2).div(sigma2).sum()
    p3 = p1+p2
    return 0.5*p3[0]
    # sigma2 = torch.max(torch.std(Y_full).pow(2), torch.Tensor([eps]))
    # p1 = torch.log(2 * math.pi * sigma2).mul(Y_full.shape[1])
    # p2 = torch.sub(test_Y, Y_full).pow(2).div(sigma2).sum()
    # p3 = p1+p2

    # return 0.5*p3[0]

# def eval_nll(gp, test_X, test_Y, ll=None, variance=1.0):
#     print('trying eval nll')
#     posterior = gp.posterior(test_X, ll=ll)
#     print('computed posterior')
#     if ll is not None:
#         Y_hat = posterior.best_mixture_mean
#     else:
#         Y_hat = posterior.mixture_mean
#     print('computed mixture mean')
#     sigma2 = torch.std(Y_hat).pow(2)
#     print('computed sigma2')
#     p1 = 0.5*torch.log(2*math.pi*sigma2).mul(Y_hat.shape[1])
#     print('computed p1')
#     p2 = torch.sub(test_Y, Y_hat).pow(2).sum().div(2*sigma2)
#     print('computed p2')
#     return p1+p2
    #sigma2 = posterior.best_mixture_variance
    #p1 = 0.5*torch.log(2*math.pi*sigma2).sum()
    #p2 = torch.sub(test_Y, Y_hat).pow(2).div(2*sigma2).sum()
    #return p1+p2