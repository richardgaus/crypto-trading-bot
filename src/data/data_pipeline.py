import pandas as pd

from skopt.space import Dimension, Real

space = [
    Real(10 ** -5, 1.25, "log-uniform", name='learning_rate'),
    Real(10 ** -6, 4096, "log-uniform", name='alpha'),  # L1 regularization term on weights
    Real(10 ** -6, 2048, "log-uniform", name='lambda')  # L2 regularization term on weights
]

def optimize_hyperparameters(dataset, hyperparameter_space, num_calls) -> dict:
    tuned_hps = None
    return tuned_hps

def split_timeseries(testset_length, testset_start=None, random_seed=0) -> (pd.DataFrame, pd.DataFrame):
    if type(testset_length) is int:
        # number of units
    elif type(testset_length) is float:
        # percentage from whole dataset

    if testset_start is None:
        # select random testset_start

def evaluate_performance(pnl_history) -> dict:
    """ Evaluates PNL history considering maximum reached loss and final PNL

    Args:
        pnl_history:

    Returns:
        Dict with key-item pairs:
            final_pnl: Final PNL
            max_loss:  Max. loss
            aggregate: Aggregated final PNL and max. loss
    """

    return {
        'final_pnl': 0,
        'max_loss': 0,
        'aggregate': aggregate_pnl_loss(0, 0)
    }

def aggregate_pnl_loss(final_pnl:float,
                       max_loss:float) -> float:
    return 0