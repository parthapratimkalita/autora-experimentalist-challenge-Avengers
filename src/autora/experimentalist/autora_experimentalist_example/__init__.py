"""
Example Experimentalist
"""
import numpy as np
import pandas as pd
from autora.state import State, StandardState, on_state, estimator_on_state, Delta, VariableCollection
from autora.experimentalist.falsification import falsification_sample
from autora.experimentalist.model_disagreement import model_disagreement_sample
from autora.experimentalist.uncertainty import uncertainty_sample
from autora.experimentalist.random import random_pool, random_sample

from typing import Union, List


def sample(
        all_conditions: Union[pd.DataFrame, np.ndarray],
        models: List,
        reference_conditions: Union[pd.DataFrame, np.ndarray] = None,
        num: int = 1) -> pd.DataFrame:
    """
    Add a description of the sampler here.

    Args:
        conditions: The pool to sample from.
            Attention: `conditions` is a field of the standard state
        models: The sampler might use output from the theorist.
            Attention: `models` is a field of the standard state
        reference_conditions: The sampler might use reference conditons
        num_samples: number of experimental conditions to select

    Returns:
        Sampled pool of experimental conditions

    *Optional*
    Examples:
        These examples add documentation and also work as tests
        >>> example_sampler([1, 2, 3, 4])
        1
        >>> example_sampler(range(3, 10))
        3

    """
    # **** STATE WRAPPER FOR YOUR EXPERIMENTALIST ***
    
    conditions = model_disagreement_sample(
          all_conditions,
          models,
          num_samples=10
        )
    return conditions.sample(n=num)