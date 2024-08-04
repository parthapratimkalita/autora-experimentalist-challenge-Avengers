"""
Example Experimentalist
"""
from typing import List, Union

import numpy as np
import pandas as pd
from autora.experimentalist.mixture import mixture_sample
from autora.experimentalist.novelty import novelty_score_sample
from numpy.random import random_sample

# Create a grid of 1000 data points
grid = np.linspace(0, 10, 1000).reshape(-1, 1)


# Define the sample function
def sample(
    conditions: Union[pd.DataFrame, np.ndarray],
    models: List,
    reference_conditions: Union[pd.DataFrame, np.ndarray] = None,
    num_samples: int = 1) -> pd.DataFrame:
    """
    Add a description of the sampler here

    Args:
        conditions: The pool to sample from.
            Attention: `conditions` is a field of the standard state
        models: The sampler might use output from the theorist.
            Attention: `models` is a field of the standard state
        reference_conditions: The sampler might use reference conditions
            num_samples: number of experimental conditions to select

        Returns:
            Sampled pool of experimental conditions
    """

    if num_samples is None:
        num_samples = conditions.shape[0]

    new_conditions = mixture_sample(
        conditions=conditions,
        temperature=0.01,
        samplers=[[novelty_score_sample, "novelty", [0.8, 0.2]], [random_sample, "random", [0.2, 0.8]]],
        params={"novelty": {"reference_conditions": reference_conditions}},
        num_samples=num_samples
    )
    return new_conditions


# Sample from the grid
# sampled_points = sample(grid, models, reference_conditions=X, num_samples=10)
# print(sampled_points)
