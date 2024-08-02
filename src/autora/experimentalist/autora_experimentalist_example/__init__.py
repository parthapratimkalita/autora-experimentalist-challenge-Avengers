"""
Example Experimentalist
"""
import numpy as np
import pandas as pd

from typing import Union, List


def sample(
    conditions: Union[pd.DataFrame, np.ndarray],
    models: List,
    reference_conditions: Union[pd.DataFrame, np.ndarray] = None,
    num_samples: int = 1) -> pd.DataFrame:
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
    if num_samples is None:
        num_samples = conditions.shape[0]

    new_conditions = model_based_exploration(conditions, models, num_samples)

    return new_conditions


def model_based_exploration(conditions, models, num_samples=1):
    predictions = np.array([model.predict(conditions) for model in models])
    avg_predictions = np.mean(predictions, axis=0).flatten()
    high_variance_indices = np.argsort(avg_predictions)[-num_samples:]
    return conditions.iloc[high_variance_indices]
