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
import time
@hydra.main(config_path='conf', config_name='config.yaml', version_base=None)
def run_experiment(cfg:DictConfig):

    tkwargs = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.double,
    }
    print(torch.cuda.is_available() )
    run_name = cfg.functions.name + "_" + cfg.acquisition.name
    print('--------------------------------------------------------------------------------------------------------------------------')
    print(run_name)
    wandb.init(
    # set the wandb project where this run will be logged
    project="bayesopt",
    name= run_name,
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
    "run": cfg.run.num
    },
    # mode='disabled'
        )

    # only scale when passing to acquisition func
    synthetic_function = instantiate(cfg.functions.function).to(**tkwargs)
    bounds = synthetic_function.bounds
    #print(bounds)
    X = SobolEngine(dimension=cfg.functions.dim, scramble=True, seed=0).draw(cfg.general.n_init).to(**tkwargs)
    #print(X)
    X_scaled = convert_bounds(X, cfg.functions.bounds, cfg.functions.dim)
    Y = synthetic_function(X_scaled).unsqueeze(-1)
    poolU = get_candidate_pool(dim=cfg.functions.dim, bounds=cfg.functions.bounds, size=cfg.general.pool_size).to(**tkwargs)
    X_test, Y_test = get_test_set(synthetic_function=synthetic_function, bounds=cfg.functions.bounds, dim=cfg.functions.dim, size=cfg.general.test_size)   
    X_test, Y_test = X_test.to(**tkwargs), Y_test.to(**tkwargs)
    for i in range(cfg.functions.n_iter):
        print(i)
        train_Y = Y  # Flip the sign since we want to minimize f(x)
        gp = MGPFullyBayesianSingleTaskGP(
            train_X=X, 
            train_Y=train_Y, 
            #train_Yvar=torch.full_like(train_Y, 1e-6),
            #input_transform=Normalize(d=cfg.functions.dim, bounds=bountensor_scaledds),
            outcome_transform=Standardize(m=1)
        )
        #print("instantiated")

        ll = fit_partially_bayesian_mgp_model(gp,
                                              cfg.general.partially_bayesian.n_samples,
                                              cfg.general.partially_bayesian.learning_rate,
                                              cfg.general.partially_bayesian.n_steps,
                                              print_iter=False)
        #print("fitted")
        acq_function = instantiate(cfg.acquisition.function, _partial_=True)
        acq_function = acq_function(gp, ll=ll)
        #print('instantiated acquisition func')
        candidates, acq_values = get_acq_values_pool(acq_function, poolU)
        #print('got candidate')
        #print("got candidate acq fucntion values")
        candidates_scaled = convert_bounds(candidates, cfg.functions.bounds, cfg.functions.dim)
        Y_next = synthetic_function(candidates_scaled).unsqueeze(-1)
        #print('got Y next')
        if cfg.functions.dim ==1:
            Y_next=Y_next.unsqueeze(-1)
        X = torch.cat((X, candidates)).to(**tkwargs)
        #print('concated X')
        Y = torch.cat((Y, Y_next)).to(**tkwargs)
        #print('concated Y')
        rmse = eval_rmse(gp, X_test, Y_test, ll=ll)
        #print('evaled rmse')
        nll = eval_nll(gp, X_test, Y_test,tkwargs, ll=ll)
        #print('evaled nll')
        wandb.log({"rmse": rmse, "nll": nll})
        #print('evaled logged wandb')
        print(f"new nll: {nll}")
    time.sleep(1)
    wandb.finish()
if __name__ =='__main__':
    run_experiment()
# look for some lazy forward 