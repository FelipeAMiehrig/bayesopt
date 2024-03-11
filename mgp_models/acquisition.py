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
from torch.quasirandom import SobolEngine
from botorch.acquisition import AnalyticAcquisitionFunction
from mgp_models.fully_bayesian import  MGPFullyBayesianSingleTaskGP
from mgp_models.utils import get_truncated_moments, normal_to_truncnorm, get_mode_index
from mgp_models.scorebo_functions import optimize_posterior_samples, get_optimal_samples

def calculate_hellinger(posterior, mcmc_dim):
    n_models = posterior._mean.shape[mcmc_dim]
    mean_minus_mgpmean = posterior._mean - posterior.selected_mixture_mean.repeat(n_models,1,1)
    BQBC = mean_minus_mgpmean.pow(2).mul(posterior.shaped_weights).sum(dim=mcmc_dim)#.div(posterior.n_active_models)
    var = posterior.weighted_variance
    mixture_variance = BQBC + var
    sigma_1 = mixture_variance.repeat(n_models,1,1)
    mu_1 = posterior.weighted_mixture_mean.repeat(n_models,1,1) #mixture_mean.repeat(n_models,1,1)
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
    return hellinger.mul(posterior.shaped_mask).sum(dim=mcmc_dim)

def calculate_wasserstein(posterior, mcmc_dim):
    n_models = posterior._mean.shape[mcmc_dim]
    mean_minus_mgpmean = posterior._mean - posterior.selected_mixture_mean.repeat(n_models,1,1)
    BQBC = mean_minus_mgpmean.pow(2).mul(posterior.shaped_weights).sum(dim=mcmc_dim)#.div(posterior.n_active_models)
    var = posterior.weighted_variance
    mixture_variance = BQBC + var
    sigma_1 = mixture_variance.repeat(n_models,1,1)
    mu_1 = posterior.weighted_mixture_mean.repeat(n_models,1,1) #mixture_mean.repeat(n_models,1,1)
    sigma_2 = posterior.variance
    mu_2 = posterior.mean
    diff_means = mu_1-mu_2
    diff_stds = torch.sqrt(sigma_2) - torch.sqrt(sigma_1)
    wasserstein = torch.sqrt(diff_means.pow(2)+diff_stds.pow(2))
    return wasserstein.mul(posterior.shaped_mask).sum(dim=mcmc_dim)

def calculate_KL(posterior, mcmc_dim):
    n_models = posterior._mean.shape[mcmc_dim]
    mean_minus_mgpmean = posterior._mean - posterior.selected_mixture_mean.repeat(n_models,1,1)
    BQBC = mean_minus_mgpmean.pow(2).mul(posterior.shaped_weights).sum(dim=mcmc_dim)#.div(posterior.n_active_models)
    var = posterior.weighted_variance
    mixture_variance = BQBC + var
    sigma_1 = mixture_variance.repeat(n_models,1,1)
    mu_1 = posterior.weighted_mixture_mean.repeat(n_models,1,1) #mixture_mean.repeat(n_models,1,1)
    sigma_2 = posterior.variance
    mu_2 = posterior.mean
    left = torch.log(torch.sqrt(sigma_2).div(torch.sqrt(sigma_1)))
    dif_means = mu_1-mu_2
    up = sigma_1 + dif_means.pow(2)
    KL = left + up.div(2*sigma_2) - 0.5
    return KL.mul(posterior.shaped_mask).sum(dim=mcmc_dim)

class BestInitGP(AnalyticAcquisitionFunction):
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
        if self.ll is None:
            max_index = get_mode_index(self.model)
        else: 
            max_index = torch.argmax(self.ll)
        var = posterior.variance[max_index]
        return var

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
        mean_minus_mgpmean = posterior._mean - posterior.selected_mixture_mean.repeat(n_models,1,1)
        BQBC = mean_minus_mgpmean.pow(2).mul(posterior.shaped_mask).sum(dim=MCMC_DIM).div(posterior.n_active_models)
        var = posterior.selected_variance
        mixture_variance = BQBC + var
        sigma_1 = mixture_variance.repeat(n_models,1,1)
        mixture_mean = posterior._mean.sum(dim=MCMC_DIM)
        mu_1 = posterior.selected_mixture_mean.repeat(n_models,1,1) #mixture_mean.repeat(n_models,1,1)
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
        mean_minus_mgpmean = posterior._mean - posterior.selected_mixture_mean.repeat(n_models,1,1)
        BQBC = mean_minus_mgpmean.pow(2).mul(posterior.shaped_mask).sum(dim=MCMC_DIM).div(posterior.n_active_models)
        var = posterior.selected_variance
        mixture_variance = BQBC + var
        sigma_1 = mixture_variance.repeat(n_models,1,1)
        mixture_mean = posterior._mean.sum(dim=MCMC_DIM)
        mu_1 = posterior.selected_mixture_mean.repeat(n_models,1,1) #mixture_mean.repeat(n_models,1,1)
        sigma_2 = posterior.variance
        mu_2 = posterior.mean
        diff_means = mu_1-mu_2
        diff_stds = torch.sqrt(sigma_2) - torch.sqrt(sigma_1)
        wasserstein = torch.sqrt(diff_means.pow(2)+diff_stds.pow(2))
        return wasserstein.mul(posterior.shaped_weights).sum(dim=MCMC_DIM)
        

class BALDKLMMAcquisitionFunction(AnalyticAcquisitionFunction):
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
        mean_minus_mgpmean = posterior._mean - posterior.selected_mixture_mean.repeat(n_models,1,1)
        BQBC = mean_minus_mgpmean.pow(2).mul(posterior.shaped_mask).sum(dim=MCMC_DIM).div(posterior.n_active_models)
        var = posterior.selected_variance
        mixture_variance = BQBC + var
        sigma_1 = mixture_variance.repeat(n_models,1,1)
        mixture_mean = posterior._mean.sum(dim=MCMC_DIM)
        mu_1 = posterior.selected_mixture_mean.repeat(n_models,1,1) #mixture_mean.repeat(n_models,1,1)
        sigma_2 = posterior.variance
        mu_2 = posterior.mean
        left = torch.log(torch.sqrt(sigma_2).div(torch.sqrt(sigma_1)))
        dif_means = mu_1-mu_2
        up = sigma_1 + dif_means.pow(2)
        KL = left + up.div(2*sigma_2) - 0.5
        return KL.mul(posterior.shaped_weights).sum(dim=MCMC_DIM)
    
class SALHLAcquisitionFunction(AnalyticAcquisitionFunction):
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
        return calculate_hellinger(posterior, MCMC_DIM)
        
class SALWSAcquisitionFunction(AnalyticAcquisitionFunction):
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
        return calculate_wasserstein(posterior, MCMC_DIM)
    
class SALKLAcquisitionFunction(AnalyticAcquisitionFunction):
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
        return calculate_KL(posterior, MCMC_DIM)
    
class BestInitGPUCB(AnalyticAcquisitionFunction):
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
        if self.ll is None:
            max_index = get_mode_index(self.model)
        else: 
            max_index = torch.argmax(self.ll)
        d = self.model.train_inputs[0].size()[1]
        t = self.model.train_inputs[0].size()[0]
        beta_t = 0.2*d*torch.log(torch.tensor([2*t]))
        var = posterior.variance[max_index]
        mu = posterior.mean[max_index]
        return mu + torch.sqrt(beta_t)*torch.sqrt(var)
    
class GPUCB_MGP(AnalyticAcquisitionFunction):
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
        var = posterior.QBMGP
        d = self.model.train_inputs[0].size()[1]
        t = self.model.train_inputs[0].size()[0]
        beta_t = 0.2*d*torch.log(torch.tensor([2*t]))
        mu = posterior.weighted_mixture_mean
        return mu + torch.sqrt(beta_t)*torch.sqrt(var)
    
class RGPUCB_MGP(AnalyticAcquisitionFunction):
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
        var = posterior.QBMGP
        d = self.model.train_inputs[0].size()[1]
        t = self.model.train_inputs[0].size()[0]
        k_t = 0.2*d*torch.log(torch.tensor([2*t]))
        theta = 1
        xi_t = torch.distributions.Gamma(k_t , 1/theta).sample()
        mu = posterior.weighted_mixture_mean
        return mu + torch.sqrt(xi_t)*torch.sqrt(var)

class IRGPUCB_MGP(AnalyticAcquisitionFunction):
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
        var = posterior.QBMGP
        d = self.model.train_inputs[0].size()[1]
        s = 1/d
        lambd = 1/2
        xi_t = torch.distributions.Exponential(lambd).sample() + s
        mu = posterior.weighted_mixture_mean
        return mu + torch.sqrt(xi_t)*torch.sqrt(var)
    
    
class GPUCB_Hellinger(AnalyticAcquisitionFunction):
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
        distance = calculate_hellinger(posterior, MCMC_DIM)
        d = self.model.train_inputs[0].size()[1]
        t = self.model.train_inputs[0].size()[0]
        beta_t = 0.2*d*torch.log(torch.tensor([2*t]))
        mu = posterior.weighted_mixture_mean
        return mu + torch.sqrt(beta_t)*distance


    
class RGPUCB_Hellinger(AnalyticAcquisitionFunction):
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
        distance = calculate_hellinger(posterior, MCMC_DIM)
        d = self.model.train_inputs[0].size()[1]
        t = self.model.train_inputs[0].size()[0]
        k_t = 0.2*d*torch.log(torch.tensor([2*t]))
        theta = 1
        xi_t = torch.distributions.Gamma(k_t , 1/theta).sample()
        mu = posterior.weighted_mixture_mean
        return mu + torch.sqrt(xi_t)*distance
    
class IRGPUCB_Hellinger(AnalyticAcquisitionFunction):
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
        distance = calculate_hellinger(posterior, MCMC_DIM)
        d = self.model.train_inputs[0].size()[1]
        s = 1/d
        lambd = 1/2
        xi_t = torch.distributions.Exponential(lambd).sample() + s
        mu = posterior.weighted_mixture_mean
        return mu + torch.sqrt(xi_t)*distance

    

class ScoreBOHellinger(AnalyticAcquisitionFunction):
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
        num_optima = 8
        model_dim = self.model.train_inputs[0].size()[1]
        bounds = torch.tensor([[0],[1]]).repeat(1,model_dim)
        opt_inputs, opt_outputs = get_optimal_samples(model=self.model,bounds=bounds, num_optima=num_optima)
        posterior = self.model.posterior(X, ll= self.ll)
        n_models = posterior._mean.shape[MCMC_DIM]
        tnorm_mean, tnorm_var = get_truncated_moments(gp=self.model,
                                                        X_to_condition_complete=opt_inputs,
                                                        Y_to_condition_complete=opt_outputs,
                                                        X=self.model.train_inputs[0],
                                                        Y=self.model.original_targets,
                                                        X_test=X,
                                                        num_optima=num_optima)
        mean_minus_mgpmean = posterior._mean.repeat(1,1,num_optima) - posterior.selected_mixture_mean.unsqueeze(0).repeat(n_models,1,num_optima)
        BQBC = mean_minus_mgpmean.pow(2).mul(posterior.shaped_weights.repeat(1,1,num_optima)).sum(dim=MCMC_DIM)#.div(posterior.n_active_models)
        var = posterior.weighted_variance.repeat(1,num_optima)
        mixture_variance = BQBC + var
        sigma_1 = mixture_variance.repeat(n_models,1,1)
        mu_1 = posterior.weighted_mixture_mean.repeat(n_models,1,num_optima)
        sigma_2 = tnorm_var
        mu_2 = tnorm_mean
        up = 2*torch.sqrt(sigma_1)*torch.sqrt(sigma_2)
        down = sigma_1+sigma_2
        to_sqrt = up.div(down)
        sqrted = torch.sqrt(to_sqrt)
        mean_up = mu_1 - mu_2
        mean_up = mean_up.pow(2)
        exped = torch.exp(-0.25*mean_up.div(down))
        right = sqrted* exped
        hellinger = 1 - right
        return hellinger.mul(posterior.shaped_mask.repeat(1,1,num_optima)).sum(dim=MCMC_DIM).sum(dim=-1)
        
class ScoreBOWasserstein(AnalyticAcquisitionFunction):
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


        num_optima = 8
        model_dim = self.model.train_inputs[0].size()[1]
        bounds = torch.tensor([[0],[1]]).repeat(1,model_dim)
        opt_inputs, opt_outputs = get_optimal_samples(model=self.model,bounds=bounds, num_optima=num_optima)

        posterior = self.model.posterior(X, ll= self.ll)
        n_models = posterior._mean.shape[MCMC_DIM]

        tnorm_mean, tnorm_var = get_truncated_moments(gp=self.model,
                                                        X_to_condition_complete=opt_inputs,
                                                        Y_to_condition_complete=opt_outputs,
                                                        X=self.model.train_inputs[0],
                                                        Y=self.model.original_targets,
                                                        X_test=X,
                                                        num_optima=num_optima)

        mean_minus_mgpmean = posterior._mean.repeat(1,1,num_optima)  - posterior.selected_mixture_mean.unsqueeze(0).repeat(n_models,1,num_optima)
        BQBC = mean_minus_mgpmean.pow(2).mul(posterior.shaped_weights.repeat(1,1,num_optima)).sum(dim=MCMC_DIM)#.div(posterior.n_active_models)
        var = posterior.weighted_variance.repeat(1,num_optima)
        mixture_variance = BQBC + var
        sigma_1 = mixture_variance.repeat(n_models,1,1)
        mu_1 = posterior.weighted_mixture_mean.repeat(n_models,1,num_optima)
        sigma_2 = tnorm_var
        mu_2 = tnorm_mean
        diff_means = mu_1-mu_2
        diff_stds = torch.sqrt(sigma_2) - torch.sqrt(sigma_1)
        wasserstein = torch.sqrt(diff_means.pow(2)+diff_stds.pow(2))
        wasserstein = wasserstein.mul(posterior.shaped_mask.repeat(1,1,num_optima)).sum(dim=MCMC_DIM).sum(dim=-1)
        return wasserstein
    

class ScoreBOKL(AnalyticAcquisitionFunction):
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

        num_optima = 8
        model_dim = self.model.train_inputs[0].size()[1]
        bounds = torch.tensor([[0],[1]]).repeat(1,model_dim)
        opt_inputs, opt_outputs = get_optimal_samples(model=self.model,bounds=bounds, num_optima=num_optima)

        posterior = self.model.posterior(X, ll= self.ll)
        n_models = posterior._mean.shape[MCMC_DIM]

        tnorm_mean, tnorm_var = get_truncated_moments(gp=self.model,
                                                        X_to_condition_complete=opt_inputs,
                                                        Y_to_condition_complete=opt_outputs,
                                                        X=self.model.train_inputs[0],
                                                        Y=self.model.original_targets,
                                                        X_test=X,
                                                        num_optima=num_optima)

        mean_minus_mgpmean = posterior._mean.repeat(1,1,num_optima) - posterior.selected_mixture_mean.unsqueeze(0).repeat(n_models,1,num_optima)
        BQBC = mean_minus_mgpmean.pow(2).mul(posterior.shaped_weights.repeat(1,1,num_optima)).sum(dim=MCMC_DIM)#.div(posterior.n_active_models)
        var = posterior.weighted_variance.repeat(1,num_optima)
        mixture_variance = BQBC + var
        sigma_1 = mixture_variance.repeat(n_models,1,1)
        mu_1 = posterior.weighted_mixture_mean.repeat(n_models,1,num_optima)
        sigma_2 = tnorm_var
        mu_2 = tnorm_mean
        left = torch.log(torch.sqrt(sigma_2).div(torch.sqrt(sigma_1)))
        dif_means = mu_1-mu_2
        up = sigma_1 + dif_means.pow(2)
        KL = left + up.div(2*sigma_2) - 0.5
        KL =KL.mul(posterior.shaped_mask.repeat(1,1,num_optima)).sum(dim=MCMC_DIM).sum(dim=-1)
        return KL

