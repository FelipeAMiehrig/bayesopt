from __future__ import annotations

from typing import Callable, Optional, Tuple
from warnings import warn

import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from torch import Tensor
from botorch.posteriors.fully_bayesian import GaussianMixturePosterior, MCMC_DIM
import numpy as np

class WeightedGMPosterior(GaussianMixturePosterior):

    def __init__(self, distribution: MultivariateNormal, weights: Optional[Tensor] = None, ll: Optional[Tensor] = None, alpha: int = 1, quantile:int=50) -> None:
        super().__init__(distribution=distribution)
        if ll is not None:
            likelihoods = ll.detach().exp()
            percentile = np.percentile(likelihoods, quantile)
            likelihoods[likelihoods < percentile] = 0
            mask = likelihoods.clone()
            mask[mask >= percentile] = 1
            weights = likelihoods.pow(alpha).squeeze().div(likelihoods.pow(alpha).sum())
        if ll is None and weights is None:
            n_models = self._mean.shape[MCMC_DIM]
            weights = torch.ones(n_models).div(n_models).to('cpu')
            mask = torch.ones(n_models)
        self.weights = weights
        self.mask = mask
        self._weighted_mixture_mean: Optional[Tensor] = None
        self._weighted_variance: Optional[Tensor] = None
        self._best_mixture_mean: Optional[Tensor] = None
        self._best_mixture_variance: Optional[Tensor] = None
        self._selected_mixture_mean: Optional[Tensor] = None
        self._selected_variance: Optional[Tensor] = None
        self._BQBC: Optional[Tensor] = None
        self._QBMGP: Optional[Tensor] = None
        self.test_size = self._mean.shape[1]
        self.shaped_weights = weights.repeat(self.test_size, 1).t().unsqueeze(-1)
        self.shaped_mask = mask.repeat(self.test_size, 1).t().unsqueeze(-1)
        self.n_active_models = mask.sum()


    @property
    def weighted_mixture_mean(self) ->Tensor:
        if self._weighted_mixture_mean is None:
            self._weighted_mixture_mean = self._mean.mul(self.shaped_weights).sum(dim=MCMC_DIM)
        return self._weighted_mixture_mean
    
    @property
    def weighted_variance(self) ->Tensor:
        if self._weighted_variance is None:
            self._weighted_variance = self._variance.mul(self.shaped_weights).sum(dim=MCMC_DIM)
        return self._weighted_variance
    
    @property
    def best_mixture_mean(self) ->Tensor:
        if self._best_mixture_mean is None:
            argmax_likelihood =self.weights.argmax()
            self._best_mixture_mean = self._mean[argmax_likelihood]
        return self._best_mixture_mean
    
    @property
    def best_mixture_variance(self) ->Tensor:
        if self._best_mixture_variance is None:
            argmax_likelihood =self.weights.argmax()
            self._best_mixture_variance = self._variance[argmax_likelihood]
        return self._best_mixture_variance
    
    @property
    def selected_mixture_mean(self) ->Tensor:
        if self._selected_mixture_mean is None:
            self._selected_mixture_mean = self._mean.mul(self.shaped_mask).sum(dim=MCMC_DIM).div(self.n_active_models)
        return self._selected_mixture_mean
    
    @property
    def selected_variance(self) ->Tensor:
        if self._selected_variance is None:
            self._selected_variance = self._variance.mul(self.shaped_mask).sum(dim=MCMC_DIM).div(self.n_active_models)
        return self._selected_variance
    #----take a look at this_____________________________________________________________________________________
    @property
    def BQBC(self) ->Tensor:
        if self._BQBC is None:
            n_models = self._mean.shape[MCMC_DIM]
            mean_minus_mgpmean = self._mean - self.selected_mixture_mean.repeat(n_models,1,1)
            self._BQBC = mean_minus_mgpmean.pow(2).mul(self.shaped_weights).sum(dim=MCMC_DIM)
        return self._BQBC
    
    @property
    def QBMGP(self) ->Tensor:
        if self._QBMGP is None:
            self._QBMGP = self.BQBC + self.weighted_variance
        return self._QBMGP
    
    def get_marginal_moments(self):
        n_models = self._mean.shape[MCMC_DIM]
        mean_minus_mgpmean = self._mean - self.selected_mixture_mean.repeat(n_models,1,1)
        BQBC = mean_minus_mgpmean.pow(2).mul(self.shaped_mask).sum(dim=MCMC_DIM).div(self.n_active_models)
        var = self.selected_variance
        mixture_variance = BQBC + var
        sigma_1 = mixture_variance.repeat(n_models,1,1)
        mixture_mean = self._mean.sum(dim=MCMC_DIM)
        mu_1 = self.selected_mixture_mean.repeat(n_models,1,1) #mixture_mean.repeat(n_models,1,1)
        return mu_1, sigma_1
    
    def get_conditional_moments(self):
        sigma_2 = self.variance
        mu_2 = self.mean
        return mu_2, sigma_2