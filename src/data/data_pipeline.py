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
        pass
    elif type(testset_length) is float:
        # percentage from whole dataset
        pass

    if testset_start is None:
        # select random testset_start
        pass

def evaluate_performance(pnl_history:pd.Series) -> dict:
    """ Evaluates PNL history considering maximum reached loss and final PNL

    Args:
        pnl_history:

    Returns:
        Dict with key-item pairs:
            final_pnl: Final PNL
            max_loss:  Max. loss
            aggregate: Aggregated final PNL and max. loss
    """
    recent_max_pnl = 0
    max_loss = 0

    for pnl in pnl_history:
        if pnl > recent_max_pnl:
            recent_max_pnl = pnl
        loss = (pnl - recent_max_pnl) / (1 + recent_max_pnl)
        if loss < max_loss:
            max_loss = loss

    return {
        'final_pnl': pnl_history.iloc[-1],
        'max_loss': max_loss,
        'aggregate': aggregate_pnl_loss(pnl_history.iloc[-1], max_loss)
    }

def aggregate_pnl_loss(final_pnl:float,
                       max_loss:float) -> float:
    return - final_pnl / max_loss