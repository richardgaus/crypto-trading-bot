from typing import List

import pandas as pd
from random import seed
from random import random
import random

from skopt.space import Dimension
from skopt import gp_minimize
from skopt.utils import use_named_args

from src.models.strategies import Strategy


def optimize_hyperparameters(dataset:pd.DataFrame,
                             asset_name:str,
                             strategy:Strategy,
                             hp_search_space:List[Dimension],
                             initial_params:List,
                             num_calls:int,
                             random_state:int=0) -> dict:
    """
    Optimize hyperparameters using Bayesian optimization and return best setting

    Args:
        dataset: OHLC timeseries
        asset_name: Name of asset
        strategy: Trading strategy
        hp_search_space: Sampling space of hyperparameters
        initial_params: Initial values of hyperparameters
        num_calls: Number of function calls
        random_stat: Random state

    Returns:
        dict with optimized hyperparameter values
    """

    @use_named_args(dimensions=hp_search_space)
    def objective_function(**params):
        strategy_rsi = strategy(**params)
        results = strategy_rsi.apply(
            ohlcv_timeseries=dataset,
            asset_name=asset_name
        )
        results.evaluation()
        performance = evaluate_performance(results.pnl_history)
        return -performance['aggregate']

    tuned_hps = gp_minimize(
        func=objective_function,
        dimensions=hp_search_space,
        x0=initial_params,
        n_calls=num_calls,
        random_state=random_state,
        verbose=True
    )

    keys = [dim.name for dim in hp_search_space]
    values = tuned_hps.x
    return dict(zip(keys, values))

def split_timeseries(dataset, testset_length, testset_start=None, random_seed=0) -> (pd.DataFrame, pd.DataFrame):
    if type(testset_length) is int:
        # number of units
        testset_units = testset_length
    elif type(testset_length) is float:
        # percentage from whole dataset
        testset_units = int(round(len(dataset.index)*testset_length))
    if testset_start is None:
        # select random testset_start
        seed(random_seed)
        testset_start = random.randint(0, len(dataset.index)-testset_units)

    test_set = dataset.iloc[testset_start:(testset_start+testset_units)]
    training_set = dataset.iloc[0:testset_start].append(dataset.iloc[(testset_start+testset_units):len(dataset.index)])
    return test_set, training_set


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
    if max_loss == 0:
        max_loss = 0.000001

    return {
        'final_pnl': pnl_history.iloc[-1],
        'max_loss': max_loss,
        'aggregate': aggregate_pnl_loss(pnl_history.iloc[-1], max_loss)
    }

def aggregate_pnl_loss(final_pnl:float,
                       max_loss:float) -> float:
    return - final_pnl / max_loss