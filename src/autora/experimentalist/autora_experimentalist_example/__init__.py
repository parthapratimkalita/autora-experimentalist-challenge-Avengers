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

    # print(f"Sampling {num_samples} conditions...")
    # print(f"Conditions: {conditions}")
    new_conditions = uncertainty_sampling(conditions, models, num_samples=num_samples)
    # print(f"New conditions: {new_conditions}")

    return new_conditions


def uncertainty_sampling(conditions, models, num_samples=1):
    # Ensure conditions are in DataFrame format for consistent indexing
    if isinstance(conditions, np.ndarray):
        # print("Converting conditions to DataFrame...")
        conditions = pd.DataFrame(conditions)

    # Get predictions from all models
    predictions = np.array([model.predict(conditions) for model in models])
    # print(f"Predictions: {predictions}")

    # Calculate uncertainty as the standard deviation of predictions
    uncertainty = np.std(predictions, axis=0).flatten()
    # print(f"Uncertainty: {uncertainty}")

    # print(f"Uncertainty shape: {uncertainty.shape}")

    # Select indices with the highest uncertainty
    most_uncertain_indices = np.argsort(uncertainty)[-num_samples:]

    # print(f"Most uncertain indices: {most_uncertain_indices}")
    # print(f"conditions.shape: {conditions.shape}")
    # print(f"Most uncertain conditions: {conditions.iloc[most_uncertain_indices]}")

    return conditions.iloc[most_uncertain_indices]
