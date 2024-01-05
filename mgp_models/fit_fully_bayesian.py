
r"""Model fitting routines."""

from __future__ import annotations
import torch
import logging
from contextlib import nullcontext
from functools import partial
from itertools import filterfalse
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Type, Union
from warnings import catch_warnings, simplefilter, warn, warn_explicit, WarningMessage

from botorch.exceptions.errors import ModelFittingError, UnsupportedError
from botorch.exceptions.warnings import OptimizationWarning
from botorch.models.approximate_gp import ApproximateGPyTorchModel
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim.closures import get_loss_closure_with_grads
from botorch.optim.core import _LBFGSB_MAXITER_MAXFUN_REGEX
from botorch.optim.fit import fit_gpytorch_mll_scipy, fit_gpytorch_mll_torch
from botorch.optim.utils import (
    _warning_handler_template,
    get_parameters,
    sample_all_priors,
)
from botorch.settings import debug
from botorch.utils.context_managers import (
    module_rollback_ctx,
    parameter_rollback_ctx,
    requires_grad_ctx,
    TensorCheckpoint,
)
from botorch.utils.dispatcher import Dispatcher, type_bypassing_encoder
from gpytorch.likelihoods import Likelihood
from gpytorch.mlls._approximate_mll import _ApproximateMarginalLogLikelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from linear_operator.utils.errors import NotPSDError
from pyro.infer.mcmc import MCMC, NUTS
from torch import device, Tensor
from torch.nn import Parameter
from torch.utils.data import DataLoader
import gpytorch

from mgp_models.fully_bayesian import MGPFullyBayesianSingleTaskGP

import torch


def combine_and_concat_tensors(*dicts):
    result_dict = {}

    # Collect all keys
    all_keys = set().union(*[d.keys() for d in dicts])

    # Iterate over keys
    for key in all_keys:
        # Concatenate tensors with the same key
        tensors_with_same_key = [d[key] for d in dicts if key in d]
        if tensors_with_same_key:
            result_dict[key] = torch.cat(tensors_with_same_key, dim=0)

    return result_dict

def fit_fully_bayesian_mgp_model_nuts(
    model: Union[MGPFullyBayesianSingleTaskGP, SaasFullyBayesianSingleTaskGP, SaasFullyBayesianMultiTaskGP],   #change
    max_tree_depth: int = 6,
    warmup_steps: int = 512,
    num_samples: int = 256,
    thinning: int = 16,
    disable_progbar: bool = False,
    jit_compile: bool = False,
) -> None:
    r"""Fit a fully Bayesian model using the No-U-Turn-Sampler (NUTS)


    Args:
        model: SaasFullyBayesianSingleTaskGP to be fitted.
        max_tree_depth: Maximum tree depth for NUTS
        warmup_steps: The number of burn-in steps for NUTS.
        num_samples:  The number of MCMC samples. Note that with thinning,
            num_samples / thinning samples are retained.
        thinning: The amount of thinning. Every nth sample is retained.
        disable_progbar: A boolean indicating whether to print the progress
            bar and diagnostics during MCMC.
        jit_compile: Whether to use jit. Using jit may be ~2X faster (rough estimate),
            but it will also increase the memory usage and sometimes result in runtime
            errors, e.g., https://github.com/pyro-ppl/pyro/issues/3136.

    Example:
        >>> gp = SaasFullyBayesianSingleTaskGP(train_X, train_Y)
        >>> fit_fully_bayesian_model_nuts(gp)
    """
    model.train()

    # Do inference with NUTS
    nuts = NUTS(
        model.pyro_model.sample,
        jit_compile=jit_compile,
        full_mass=True,
        ignore_jit_warnings=True,
        max_tree_depth=max_tree_depth,
    )
    mcmc = MCMC(
        nuts,
        warmup_steps=warmup_steps,
        num_samples=num_samples,
        disable_progbar=disable_progbar,
    )
    mcmc.run()

    # Get final MCMC samples from the Pyro model
    mcmc_samples = model.pyro_model.postprocess_mcmc_samples(
        mcmc_samples=mcmc.get_samples()
    )
    for k, v in mcmc_samples.items():
        mcmc_samples[k] = v[::thinning]

    # Load the MCMC samples back into the BoTorch model
    model.load_mcmc_samples(mcmc_samples)
    model.eval()

def fit_partially_bayesian_mgp_model(
    model: Union[MGPFullyBayesianSingleTaskGP, SaasFullyBayesianSingleTaskGP, SaasFullyBayesianMultiTaskGP],   #change
    num_samples: int = 16,
    lr : float = 0.1,
    learning_steps : int = 10,
    jit_compile: bool = False,
    print_iter: bool = False
) -> Tensor:
    r"""Fit a fully Bayesian model using the No-U-Turn-Sampler (NUTS)

    Args:
        model: SaasFullyBayesianSingleTaskGP to be fitted.
        max_tree_depth: Maximum tree depth for NUTS
        warmup_steps: The number of burn-in steps for NUTS.
        num_samples:  The number of MCMC samples. Note that with thinning,
            num_samples / thinning samples are retained.
        thinning: The amount of thinning. Every nth sample is retained.
        disable_progbar: A boolean indicating whether to print the progress
            bar and diagnostics during MCMC.
        jit_compile: Whether to use jit. Using jit may be ~2X faster (rough estimate),
            but it will also increase the memory usage and sometimes result in runtime
            errors, e.g., https://github.com/pyro-ppl/pyro/issues/3136.

    Example:
        >>> gp = SaasFullyBayesianSingleTaskGP(train_X, train_Y)
        >>> fit_fully_bayesian_model_nuts(gp)
    """
    model.train()

    train_y = model.pyro_model.train_Y.repeat(num_samples, 1, 1).squeeze()
    # Do inference with NUTS
    list_dict_prior_samples = [model.pyro_model.sample_prior() for i in range(num_samples)]
    prior_samples = combine_and_concat_tensors(*list_dict_prior_samples)
    model.load_mcmc_samples(prior_samples)
    model.likelihood.train()
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    # Set up early stopping parameters
    batch_size = 1
    best_loss = float('inf')
    patience = 15  # Adjust this value based on your preference
    print("start train loop")
    for i in range(learning_steps):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        output = model(model.pyro_model.train_X)
        loss = -mll(output, train_y).sum() 
        loss.backward()
        if print_iter or i % 50 == 0:
            print('Iter %d/%d - Loss: %.3f' % (i + 1, learning_steps, loss.item()))
        if i == 0 or loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                #if print_iter:
                print(f'Early stopping at iteration {i + 1} with best loss: {best_loss}')
                break
        optimizer.step()
    print("ended train loop")
        #maybe add likelihoods to the model parameters
    model.eval()
    return mll(output, train_y)
