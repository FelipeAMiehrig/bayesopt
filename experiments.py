from torch.quasirandom import SobolEngine
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import hydra
import matplotlib.pyplot as plt
import torch
import numpy as np
import wandb
from pyDOE2 import lhs
from botorch.models.transforms import Standardize
from botorch.models.transforms.input import Normalize
from mgp_models.fully_bayesian import  MGPFullyBayesianSingleTaskGP
from mgp_models.fit_fully_bayesian import fit_fully_bayesian_mgp_model_nuts, fit_partially_bayesian_mgp_model
from mgp_models.utils import get_candidate_pool, get_test_set, get_acq_values_pool, eval_nll,eval_mll, eval_rmse, convert_bounds, eval_new_mll, calculate_regret
import time
@hydra.main(config_path='conf', config_name='config.yaml', version_base=None)
def run_experiment(cfg:DictConfig):
    
    torch.set_default_dtype(torch.double)
    tkwargs = {
    "device": torch.device("cpu" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.double
    }
    run_name = cfg.functions.name + "_" + cfg.acquisition.name
    print('--------------------------------------------------------------------------------------------------------------------------')
    print(run_name)
    wandb.login(key="a57a0466fc25b416b4e026ec6970ba6eff09e105")
    wandb.init(
    # set the wandb project where this run will be logged
    project="bayesopt",
    name= run_name,
    settings=wandb.Settings(start_method="thread"),
    dir = r"C:\Users\felip\wandb_logs" ,
    # track hyperparameters and run metadata
    config={
    "task": cfg.task.name,
    "type": cfg.type.name,
    "function": cfg.functions.name,
    "n_dim": cfg.functions.dim,
    "n_iter": cfg.functions.n_iter,
    "acquisition": cfg.acquisition.name,
    "n_samples_pb": cfg.general.part_bayesian.n_samples,
    "n_samples_b": cfg.general.fully_bayesian.num_samples,
    "warmup_steps": cfg.general.fully_bayesian.warmup_steps, 
    "thinning": cfg.general.fully_bayesian.thinning,
    "lr": cfg.general.part_bayesian.learning_rate,
    "n_steps" : cfg.general.part_bayesian.n_steps, 
    "n_init": cfg.general.n_init,
    "test_size": cfg.general.test_size,
    "pool_size": cfg.general.pool_size,
    "run": cfg.run.num
    },
    mode='disabled'
        )

    # only scale when passing to acquisition func
    synthetic_function = instantiate(cfg.functions.function).to(**tkwargs)
    bounds = synthetic_function.bounds
    #print(bounds)
    #X = SobolEngine(dimension=cfg.functions.dim, scramble=True).draw(cfg.general.n_init).to(**tkwargs)
    X = torch.from_numpy(np.array(lhs(cfg.functions.dim, samples=cfg.general.n_init, criterion='maximin'))).to(**tkwargs)
    #print(X)
    X_scaled = convert_bounds(X, cfg.functions.bounds, cfg.functions.dim)
    Y = synthetic_function(X_scaled).unsqueeze(-1)
    Y_true = synthetic_function.evaluate_true(X_scaled).unsqueeze(-1)
    poolU = get_candidate_pool(dim=cfg.functions.dim, bounds=cfg.functions.bounds, size=cfg.general.pool_size).to(**tkwargs)
    X_test, Y_test = get_test_set(synthetic_function=synthetic_function, 
                                  bounds=cfg.functions.bounds, 
                                  dim=cfg.functions.dim, 
                                  noise_std=cfg.functions.function.noise_std,
                                  size=cfg.functions.test_size, 
                                  eval_true=cfg.general.eval_true)  
    results = -synthetic_function.evaluate_true(convert_bounds(poolU, cfg.functions.bounds, cfg.functions.dim))
    sorted, indices = torch.sort(results, descending=True)
    #plt.hist(sorted[:100].detach().numpy())
    #plt.show()
    #print(torch.max(results))
    X_test, Y_test = X_test.to(**tkwargs), Y_test.to(**tkwargs)
    cum_regret = 0
    log_dict = {}
    for i in range(cfg.functions.n_iter):
        print(i)
        start_time = time.time()
        train_Y = Y  # Flip the sign since we want to minimize f(x)
        gp = MGPFullyBayesianSingleTaskGP(
            train_X=X, 
            train_Y=train_Y, 
            #train_Yvar=torch.full_like(train_Y, 1e-6),
            #input_transform=Normalize(d=cfg.functions.dim, bounds=bountensor_scaledds),
            outcome_transform=Standardize(m=1)
        )
        if cfg.type.name =='part_bayesian':
            ll = fit_partially_bayesian_mgp_model(gp,
                                                cfg.general.part_bayesian.n_samples,
                                                cfg.general.part_bayesian.learning_rate,
                                                cfg.general.part_bayesian.n_steps,
                                                print_iter=False)
        else:
            ll = fit_fully_bayesian_mgp_model_nuts(gp,
                                                   warmup_steps=cfg.general.fully_bayesian.warmup_steps,
                                                   num_samples=cfg.general.fully_bayesian.num_samples,
                                                   thinning=cfg.general.fully_bayesian.thinning,
                                                   disable_progbar=True)
        #print("fitted")
        acq_function = instantiate(cfg.acquisition.function, _partial_=True)
        acq_function = acq_function(gp, ll=ll)
        end_time = time.time()
        del gp
        #print('instantiated acquisition func')
        candidates, acq_values, poolU = get_acq_values_pool(acq_function, poolU)
        
        #print('got candidate')
        #print("got candidate acq fucntion values")
        candidates_scaled = convert_bounds(candidates, cfg.functions.bounds, cfg.functions.dim)
        Y_next = synthetic_function(candidates_scaled).unsqueeze(-1)
        Y_next_true = synthetic_function.evaluate_true(candidates_scaled).unsqueeze(-1)
        if cfg.functions.dim ==1:
            Y_next=Y_next.unsqueeze(-1)
            Y_next_true=Y_next_true.unsqueeze(-1)
            log_dict["X_1"] = candidates.squeeze().float()
        else:
            for xi in range(cfg.functions.dim):
                log_dict["X_"+str(xi+1)] = candidates.squeeze()[xi].float()
        log_dict["Y"] = Y_next.float()
        log_dict["Y_true"] = Y_next_true.float()
        if cfg.task.name == "AL":
            if cfg.general.eval_true:
                nmll, rmse = eval_new_mll(X_test, Y_test, X, Y_true, tkwargs)#change Y_true to train_Y
            else:
                nmll, rmse = eval_new_mll(X_test, Y_test, X, train_Y, tkwargs)#change Y_true to train_Y
            log_dict.update({"rmse": rmse, "-MLL":nmll})
            print(f"new mll: {nmll}")
            print(f"new rmse: {rmse}")
        if cfg.task.name == "BO":
            regret = calculate_regret(Y=Y_true, synthetic_function=synthetic_function, negate=cfg.functions.function.negate, func_name=cfg.functions.name)
            cum_regret = cum_regret + regret
            log_dict.update({"regret":regret, "log_regret":np.log(regret), "cum_regret":cum_regret})
            print(f"new regret: {regret}")
        X = torch.cat((X, candidates)).to(**tkwargs)
        #print('concated X')
        Y = torch.cat((Y, Y_next)).to(**tkwargs)
        Y_true = torch.cat((Y_true, Y_next_true)).to(**tkwargs)
        #print('concated Y')
        #rmse = eval_rmse(gp, X_test, Y_test, tkwargs, ll=ll)
        #print('evaled rmse')
        #nll = eval_nll(gp, X_test, Y_test, train_Y, tkwargs, ll=ll)
        #print('evaled nll')
        log_dict.update({ "iteration_time":end_time-start_time})
        wandb.log(log_dict)
        #print('evaled logged wandb')
        #print(f"new nll: {nll}")


    wandb.finish()
    torch.cuda.empty_cache()
if __name__ =='__main__':
    run_experiment()
# look for some lazy forward 