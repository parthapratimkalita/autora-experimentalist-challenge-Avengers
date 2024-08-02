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

    return hybrid_sampling(conditions, models, num_samples=5, alpha=0.7)


def hybrid_sampling(conditions, models, num_samples=1, alpha=0.7):
    # Ensure conditions are in DataFrame format for consistent indexing
    if isinstance(conditions, np.ndarray):
        conditions = pd.DataFrame(conditions)

    total_samples = conditions.shape[0]
    num_uncertain_samples = int(num_samples * alpha)
    num_random_samples = num_samples - num_uncertain_samples

    # Get predictions from all models
    predictions = np.array([model.predict(conditions) for model in models])

    # Calculate uncertainty as the standard deviation of predictions
    uncertainty = np.std(predictions, axis=0).flatten()

    # Select indices with the highest uncertainty
    most_uncertain_indices = np.argsort(uncertainty)[-num_uncertain_samples:]

    # Select random indices
    random_indices = np.random.choice(total_samples, num_random_samples, replace=False)

    # Combine the indices
    combined_indices = np.concatenate((most_uncertain_indices, random_indices))

    return conditions.iloc[combined_indices]