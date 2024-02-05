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
            likelihoods[likelihoods < np.percentile(likelihoods, quantile)] = 0
            weights = likelihoods.pow(alpha).squeeze().div(likelihoods.pow(alpha).sum())
        if ll is None and weights is None:
            n_models = self._mean.shape[MCMC_DIM]
            weights = torch.ones(n_models).div(n_models).to('cpu')
        self.weights = weights
        self._weighted_mixture_mean: Optional[Tensor] = None
        self._weighted_variance: Optional[Tensor] = None
        self._best_mixture_mean: Optional[Tensor] = None
        self._best_mixture_variance: Optional[Tensor] = None
        self._BQBC: Optional[Tensor] = None
        self._QBMGP: Optional[Tensor] = None
        self.test_size = self._mean.shape[1]
        self.shaped_weights = weights.repeat(self.test_size, 1).t().unsqueeze(-1)

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
    #----take a look at this_____________________________________________________________________________________
    @property
    def BQBC(self) ->Tensor:
        if self._BQBC is None:
            n_models = self._mean.shape[MCMC_DIM]
            mean_minus_mgpmean = self._mean - self.mixture_mean.repeat(n_models,1,1)
            self._BQBC = mean_minus_mgpmean.pow(2).mul(self.shaped_weights).sum(dim=MCMC_DIM)
        return self._BQBC
    
    @property
    def QBMGP(self) ->Tensor:
        if self._QBMGP is None:
            self._QBMGP = self.BQBC + self.weighted_variance
        return self._QBMGP