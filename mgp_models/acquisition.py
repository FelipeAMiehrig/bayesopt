import math
from typing import Optional

from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils import t_batch_mode_transform
from torch import Tensor
import torch
from botorch.posteriors.fully_bayesian import MCMC_DIM

from botorch.acquisition import AnalyticAcquisitionFunction
from mgp_models.fully_bayesian import  MGPFullyBayesianSingleTaskGP

class BALMAcquisitionFunction(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: MGPFullyBayesianSingleTaskGP,
        maximize: bool = True,
        ll: Optional[Tensor] = None
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # AnalyticAcquisitionFunction performs some validity checks that we don't want here
        super(AnalyticAcquisitionFunction, self).__init__(model)
        self.maximize = maximize
        self.ll = ll

    def forward(self, X: Tensor) -> Tensor:


        posterior = self.model.posterior(X, ll= self.ll)
        return posterior.weighted_variance
    

class BQBCAcquisitionFunction(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: MGPFullyBayesianSingleTaskGP,
        maximize: bool = True,
        ll: Optional[Tensor] = None
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # AnalyticAcquisitionFunction performs some validity checks that we don't want here
        super(AnalyticAcquisitionFunction, self).__init__(model)
        self.maximize = maximize
        self.ll = ll

    def forward(self, X: Tensor) -> Tensor:


        posterior = self.model.posterior(X, ll= self.ll)
        return posterior.BQBC
    

class QBMGPAcquisitionFunction(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: MGPFullyBayesianSingleTaskGP,
        maximize: bool = True,
        ll: Optional[Tensor] = None
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # AnalyticAcquisitionFunction performs some validity checks that we don't want here
        super(AnalyticAcquisitionFunction, self).__init__(model)
        self.maximize = maximize
        self.ll = ll

    def forward(self, X: Tensor) -> Tensor:


        posterior = self.model.posterior(X, ll= self.ll)
        return posterior.QBMGP
    
class RandomAcquisitionFunction(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: MGPFullyBayesianSingleTaskGP,
        maximize: bool = True,
        ll: Optional[Tensor] = None
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # AnalyticAcquisitionFunction performs some validity checks that we don't want here
        super(AnalyticAcquisitionFunction, self).__init__(model)
        self.maximize = maximize
        self.ll = ll

    def forward(self, X: Tensor) -> Tensor:


        
        return torch.rand(X.size()[0])
    

class SALHellingerMMAcquisitionFunction(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: MGPFullyBayesianSingleTaskGP,
        maximize: bool = True,
        ll: Optional[Tensor] = None
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # AnalyticAcquisitionFunction performs some validity checks that we don't want here
        super(AnalyticAcquisitionFunction, self).__init__(model)
        self.maximize = maximize
        self.ll = ll

    def forward(self, X: Tensor) -> Tensor:


        posterior = self.model.posterior(X, ll= self.ll)
        n_models = posterior._mean.shape[MCMC_DIM]
        mean_minus_mgpmean = posterior._mean - posterior.mixture_mean.repeat(n_models,1,1)
        BQBC = mean_minus_mgpmean.pow(2).sum(dim=MCMC_DIM)
        var = posterior._variance.sum(dim=MCMC_DIM)
        mixture_variance = BQBC + var
        sigma_1 = mixture_variance.repeat(n_models,1,1)
        mixture_mean = posterior._mean.sum(dim=MCMC_DIM)
        mu_1 = mixture_mean.repeat(n_models,1,1)
        sigma_2 = posterior.variance
        mu_2 = posterior.mean
        up = 2*torch.sqrt(sigma_1)*torch.sqrt(sigma_2)
        down = sigma_1+sigma_2
        to_sqrt = up.div(down)
        sqrted = torch.sqrt(to_sqrt)
        mean_up = mu_1 - mu_2
        mean_up = mean_up.pow(2)
        exped = torch.exp(-0.25*mean_up.div(down))
        right = sqrted* exped
        hellinger = 1 - right
        return hellinger.mul(posterior.shaped_weights).sum(dim=MCMC_DIM)
        




class SALWassersteinMMAcquisitionFunction(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: MGPFullyBayesianSingleTaskGP,
        maximize: bool = True,
        ll: Optional[Tensor] = None
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # AnalyticAcquisitionFunction performs some validity checks that we don't want here
        super(AnalyticAcquisitionFunction, self).__init__(model)
        self.maximize = maximize
        self.ll = ll

    def forward(self, X: Tensor) -> Tensor:


        posterior = self.model.posterior(X, ll= self.ll)
        n_models = posterior._mean.shape[MCMC_DIM]
        mean_minus_mgpmean = posterior._mean - posterior.mixture_mean.repeat(n_models,1,1)
        BQBC = mean_minus_mgpmean.pow(2).sum(dim=MCMC_DIM)
        var = posterior._variance.sum(dim=MCMC_DIM)
        mixture_variance = BQBC + var
        sigma_1 = mixture_variance.repeat(n_models,1,1)
        mixture_mean = posterior._mean.sum(dim=MCMC_DIM)
        mu_1 = mixture_mean.repeat(n_models,1,1)
        sigma_2 = posterior.variance
        mu_2 = posterior.mean
        diff_means = mu_1-mu_2
        diff_stds = torch.sqrt(sigma_2) - torch.sqrt(sigma_1)
        wasserstein = torch.sqrt(diff_means.pow(2)+diff_stds.pow(2))
        return wasserstein.mul(posterior.shaped_weights).sum(dim=MCMC_DIM)
        

class SALKLMMAcquisitionFunction(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: MGPFullyBayesianSingleTaskGP,
        maximize: bool = True,
        ll: Optional[Tensor] = None
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # AnalyticAcquisitionFunction performs some validity checks that we don't want here
        super(AnalyticAcquisitionFunction, self).__init__(model)
        self.maximize = maximize
        self.ll = ll

    def forward(self, X: Tensor) -> Tensor:


        posterior = self.model.posterior(X, ll= self.ll)
        n_models = posterior._mean.shape[MCMC_DIM]
        mean_minus_mgpmean = posterior._mean - posterior.mixture_mean.repeat(n_models,1,1)
        BQBC = mean_minus_mgpmean.pow(2).sum(dim=MCMC_DIM)
        var = posterior._variance.sum(dim=MCMC_DIM)
        mixture_variance = BQBC + var
        sigma_1 = mixture_variance.repeat(n_models,1,1)
        mixture_mean = posterior._mean.sum(dim=MCMC_DIM)
        mu_1 = mixture_mean.repeat(n_models,1,1)
        sigma_2 = posterior.variance
        mu_2 = posterior.mean
        left = torch.log(torch.sqrt(sigma_1).div(torch.sqrt(sigma_2)))
        dif_means = mu_1-mu_2
        up = sigma_1 + dif_means.pow(2)
        KL = left + up.div(sigma_2) - 0.5
        return KL.mul(posterior.shaped_weights).sum(dim=MCMC_DIM)