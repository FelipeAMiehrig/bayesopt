from botorch.utils.transforms import unnormalize
from typing import Any, Generator, Iterable, List, Optional, Tuple, TYPE_CHECKING
from botorch.sampling.pathwise.paths import SamplePath
from botorch.sampling.pathwise.posterior_samplers import draw_matheron_paths
import torch
from torch import Tensor
from torch.quasirandom import SobolEngine
from botorch.models.model import Model

def optimize_posterior_samples(
    paths: SamplePath,
    bounds: Tensor,
    candidates: Optional[Tensor] = None,
    raw_samples: Optional[int] = 1024,
    num_restarts: int = 20,
    maximize: bool = True,
    **kwargs: Any,
) -> Tuple[Tensor, Tensor]:
    r"""Cheaply maximizes posterior samples by random querying followed by vanilla
    gradient descent on the best num_restarts points.

    Args:
        paths: Random Fourier Feature-based sample paths from the GP
        bounds: The bounds on the search space.
        candidates: A priori good candidates (typically previous design points)
            which acts as extra initial guesses for the optimization routine.
        raw_samples: The number of samples with which to query the samples initially.
        num_restarts: The number of points selected for gradient-based optimization.
        maximize: Boolean indicating whether to maimize or minimize

    Returns:
        A two-element tuple containing:
            - X_opt: A `num_optima x [batch_size] x d`-dim tensor of optimal inputs x*.
            - f_opt: A `num_optima x [batch_size] x 1`-dim tensor of optimal outputs f*.
    """
    if maximize:

        def path_func(x):
            return paths(x)

    else:

        def path_func(x):
            return -paths(x)

    candidate_set = unnormalize(
        SobolEngine(dimension=bounds.shape[1], scramble=True).draw(raw_samples), bounds
    )

    # queries all samples on all candidates - output shape
    # raw_samples * num_optima * num_models
    candidate_queries = path_func(candidate_set)
    argtop_k = torch.topk(candidate_queries, num_restarts, dim=-1).indices
    X_top_k = candidate_set[argtop_k, :]

    # to avoid circular import, the import occurs here
    from botorch.generation.gen import gen_candidates_torch

    X_top_k, f_top_k = gen_candidates_torch(
        X_top_k, path_func, lower_bounds=bounds[0], upper_bounds=bounds[1], **kwargs
    )
    f_opt, arg_opt = f_top_k.max(dim=-1, keepdim=True)

    # For each sample (and possibly for every model in the batch of models), this
    # retrieves the argmax. We flatten, pick out the indices and then reshape to
    # the original batch shapes (so instead of pickig out the argmax of a
    # (3, 7, num_restarts, D)) along the num_restarts dim, we pick it out of a
    # (21  , num_restarts, D)
    final_shape = candidate_queries.shape[:-1]
    X_opt = X_top_k.reshape(final_shape.numel(), num_restarts, -1)[
        torch.arange(final_shape.numel()), arg_opt.flatten()
    ].reshape(*final_shape, -1)
    if not maximize:
        f_opt = -f_opt
    return X_opt, f_opt




def get_optimal_samples(
    model: Model,
    bounds: Tensor,
    num_optima: int,
    raw_samples: int = 1024,
    num_restarts: int = 20,
    maximize: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Draws sample paths from the posterior and maximizes the samples using GD.

    Args:
        model (Model): The model from which samples are drawn.
        bounds: (Tensor): Bounds of the search space. If the model inputs are
            normalized, the bounds should be normalized as well.
        num_optima (int): The number of paths to be drawn and optimized.
        raw_samples (int, optional): The number of candidates randomly sample.
            Defaults to 1024.
        num_restarts (int, optional): The number of candidates to do gradient-based
            optimization on. Defaults to 20.
        maximize: Whether to maximize or minimize the samples.
    Returns:
        Tuple[Tensor, Tensor]: The optimal input locations and corresponding
        outputs, x* and f*.

    """
    paths = draw_matheron_paths(model, sample_shape=torch.Size([num_optima]))
    optimal_inputs, optimal_outputs = optimize_posterior_samples(
        paths,
        bounds=bounds,
        raw_samples=raw_samples,
        num_restarts=num_restarts,
        maximize=maximize,
    )
    return optimal_inputs, optimal_outputs