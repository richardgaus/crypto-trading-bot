import pandas as pd

# seed the pseudorandom number generator
from random import seed
from random import random
from skopt.space import Dimension, Real
import random


def optimize_hyperparameters(dataset, hyperparameter_space, num_calls) -> dict:
    tuned_hps = None
    return tuned_hps

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
    return (test_set, training_set)

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