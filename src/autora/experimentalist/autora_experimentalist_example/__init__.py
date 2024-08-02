"""
Example Experimentalist
"""
from typing import Union, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


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

    return diversity_sampling(conditions, num_samples)


def diversity_sampling(conditions, num_samples=1):
    kmeans = KMeans(n_clusters=num_samples)
    kmeans.fit(conditions)
    cluster_centers = kmeans.cluster_centers_
    return pd.DataFrame(cluster_centers, columns=conditions.columns)
