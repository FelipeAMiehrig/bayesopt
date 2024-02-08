import torch
from torch import Tensor
import math
from gpytorch.models import ExactGP
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from mgp_models.fully_bayesian import reshape_and_detach

def convert_bounds(tensor: Tensor, bounds, dim: int)-> Tensor:
    tensor_scaled = tensor.detach().clone()
    if bounds is not None:
        for i in range(dim):
            a, b = bounds[i]
            tensor_scaled[:, i] = a + (b - a) * tensor[:, i]
    return tensor_scaled

def get_candidate_pool(dim, bounds, size=1000, seed=0):
    local_rng = torch.Generator()
    local_rng.manual_seed(seed)
    U = torch.rand(size, dim, generator=local_rng)
    #U = convert_bounds(U, bounds, dim)
    return U


def get_test_set(synthetic_function, dim, bounds,noise_std, size=1000, seed=0):
    local_rng = torch.Generator()
    local_rng.manual_seed(seed)
    X_test = torch.rand(size, dim, generator=local_rng)
    X_test_scaled = convert_bounds(X_test, bounds, dim)
    Y_test = synthetic_function.evaluate_true(X_test_scaled)

    noise = torch.normal(0, noise_std, size=(size,1), generator=local_rng).squeeze()
    Y_test = Y_test+noise
    return X_test, Y_test

def get_acq_values_pool(acq_function, poolU):

    acq_values = acq_function(poolU)
    best_index = torch.argmax(acq_values)
    candidates = poolU[best_index].unsqueeze(-1)
    new_pool = torch.cat((poolU[:best_index], poolU[best_index + 1:]), dim=0)
    return candidates.T, acq_values[best_index].unsqueeze(-1), new_pool

def normalize_tensors(train_Y, test_Y, Y_hat):
    means = train_Y.mean().float()
    stds = train_Y.std().float()
    normalized_test = (test_Y - means) / stds
    normalized_hat= (Y_hat - means) / stds
    return normalized_test, normalized_hat


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


def eval_nll(gp, test_X, test_Y,train_Y, tkwargs, ll=None, batch_size = 100, eps=1e-6):
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
    best_gp_index = ll.argmax()
    noise = gp.likelihood.noise_covar.noise[best_gp_index]
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
    var_full = var_full + noise
    concated = torch.cat((var_full, eps_tensor),1)
    sigma2 = torch.max(concated, 1)[0]
    test_Y, Y_full = normalize_tensors(train_Y,test_Y, Y_full)
    p1 = torch.log(2 * math.pi * sigma2)
    p2 = torch.sub(test_Y, Y_full).pow(2).div(sigma2).sum()
    p3 = p1+p2
    return 0.5*p3[0]/test_X.size(0)
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


class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscales, outputscale, noise, dim, tkwargs):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean().to(**tkwargs)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=dim)
        ).to(**tkwargs)

        # Set the parameters
        self.covar_module.base_kernel.lengthscale = lengthscales
        self.covar_module.outputscale = outputscale
        self.likelihood.noise = noise

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


#---------------------------------------CHECK IF BEING UPDATED

def eval_mll(gp, test_X, test_Y, train_X, train_Y, tkwargs, ll=None):

    dict_params = get_best_model_params(gp, ll=ll)
    train_x, train_y = None, None # Your training data
    new_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(**tkwargs)
    dim = test_X.size()[1]
# Given parameters
    new_model = ExactGPModel(None, None, new_likelihood, dict_params['lengthscales'],
                              dict_params['outputscale'], dict_params['noise'], dim, tkwargs)
    new_model.set_train_data(train_X, train_Y.squeeze(), strict=False)
    #mll = get_mll_best_model(gp, best_gp_index, train_X, train_Y, test_X, test_Y)
    new_model.eval()
    new_likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = new_model(test_X)
        mll = ExactMarginalLogLikelihood(new_likelihood, new_model)
        test_mll = mll(predictions, test_Y)
        rmse = gpytorch.metrics.mean_squared_error(predictions, test_Y, squared=True)
    return -test_mll, rmse


def get_best_model_params(gp, ll=None):
    if ll is None:
        best_gp_index = 0
    else:
        best_gp_index = ll.argmax()
    dict_params = {}
    dict_params['noise'] = gp.likelihood.noise_covar.noise.clone().detach()[best_gp_index]
    dict_params['lengthscale'] = gp.covar_module.base_kernel.lengthscale.clone().detach()[best_gp_index]
    dict_params['outputscale'] = gp.covar_module.outputscale.clone().detach()[best_gp_index].unsqueeze(-1)
    dict_params['mean'] = gp.mean_module.constant.clone().detach()[best_gp_index].unsqueeze(-1)
    return dict_params

def log_best_params(gp, dict_params, index = 0):
    gp.likelihood.noise_covar.noise[index] = reshape_and_detach(target=gp.likelihood.noise_covar.noise[index], 
                                                                new_value=dict_params['noise'])
    gp.covar_module.base_kernel.lengthscale[index] = reshape_and_detach(target=gp.covar_module.base_kernel.lengthscale[index], 
                                                                new_value=dict_params['lengthscale'])
    gp.covar_module.outputscale[index] = reshape_and_detach(target=gp.covar_module.outputscale[index], 
                                                                new_value=dict_params['outputscale'])
    return gp