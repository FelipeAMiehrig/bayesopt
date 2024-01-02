from torch.quasirandom import SobolEngine
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import hydra
import torch
import wandb
from botorch.models.transforms import Standardize
from botorch.models.transforms.input import Normalize
from mgp_models.fully_bayesian import  MGPFullyBayesianSingleTaskGP
from mgp_models.fit_fully_bayesian import fit_fully_bayesian_mgp_model_nuts, fit_partially_bayesian_mgp_model
from mgp_models.utils import get_candidate_pool, get_test_set, get_acq_values_pool, eval_nll, eval_rmse, convert_bounds

@hydra.main(config_path='conf', config_name='config.yaml', version_base=None)
def run_experiment(cfg:DictConfig):

    tkwargs = {
    "device": torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.double,
    }

    wandb.init(
    # set the wandb project where this run will be logged
    project="bayesopt",
    name= cfg.functions.name + "_" + cfg.acquisition.name,
    settings=wandb.Settings(start_method="thread"),
    # track hyperparameters and run metadata
    config={
    "task": 'AL',
    "type": "partially bayesian",
    "function": cfg.functions.name,
    "n_dim": cfg.functions.dim,
    "n_iter": cfg.functions.n_iter,
    "acquisition": cfg.acquisition.name,
    "n_samples": cfg.general.partially_bayesian.n_samples,
    "lr": cfg.general.partially_bayesian.learning_rate,
    "n_steps" : cfg.general.partially_bayesian.n_steps, 
    "n_init": cfg.general.n_init,
    "test_size": cfg.general.test_size,
    "pool_size": cfg.general.pool_size,
    },
    #    mode='disabled'
        )


    synthetic_function = instantiate(cfg.functions.function).to(**tkwargs)
    #synthetic_function =synthetic_function.to(**tkwargs)
    X = SobolEngine(dimension=cfg.functions.dim, scramble=True, seed=0).draw(cfg.general.n_init).to(**tkwargs)
    X = convert_bounds(X, cfg.functions.bounds, cfg.functions.dim)
    Y = synthetic_function(X).unsqueeze(-1)
    print(X)
    print(Y)
    poolU = get_candidate_pool(dim=cfg.functions.dim, bounds=cfg.functions.bounds, size=cfg.general.pool_size)
    X_test, Y_test = get_test_set(synthetic_function=synthetic_function, bounds=cfg.functions.bounds, dim=cfg.functions.dim, size=cfg.general.test_size)    
    for i in range(cfg.functions.n_iter):
        train_Y = Y  # Flip the sign since we want to minimize f(x)
        gp = MGPFullyBayesianSingleTaskGP(
            train_X=X, 
            train_Y=train_Y, 
            #train_Yvar=torch.full_like(train_Y, 1e-6),
            #input_transform=Normalize(d=cfg.functions.dim, bounds=synthetic_function.bounds),
            outcome_transform=Standardize(m=1)
        )

        ll = fit_partially_bayesian_mgp_model(gp,
                                              cfg.general.partially_bayesian.n_samples,
                                              cfg.general.partially_bayesian.learning_rate,
                                              cfg.general.partially_bayesian.n_steps)
        #print("fitted")
        acq_function = instantiate(cfg.acquisition.function, _partial_=True)
        acq_function = acq_function(gp, ll=ll)
        candidates, acq_values = get_acq_values_pool(acq_function, poolU)
        #print("got candidate acq fucntion values")
        Y_next = synthetic_function(candidates).unsqueeze(-1)
        if cfg.functions.dim ==1:
            Y_next=Y_next.unsqueeze(-1)
        X = torch.cat((X, candidates))
        Y = torch.cat((Y, Y_next))
        rmse = eval_rmse(gp, X_test, Y_test, ll=ll)
        nll = eval_nll(gp, X_test, Y_test, ll=ll)
        wandb.log({"rmse": rmse, "nll": nll})
        print(f"new nll: {nll}")
    wandb.finish()
if __name__ =='__main__':
    run_experiment()
