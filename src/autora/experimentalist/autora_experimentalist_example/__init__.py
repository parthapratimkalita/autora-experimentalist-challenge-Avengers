"""
Mixed Disagreement Experimentalist
"""
from typing import Union, List

import numpy as np
import pandas as pd
from autora.experimentalist.model_disagreement import model_disagreement_sample


def sample(
        all_conditions: Union[pd.DataFrame, np.ndarray],
        models: List,
        reference_conditions: Union[pd.DataFrame, np.ndarray] = None,
        num: int = 1) -> pd.DataFrame:
    """
        Samples a specified number of experimental conditions based on model disagreement.

        This function generates a pool of conditions by evaluating the disagreement among 
        multiple models and then randomly samples a specified number of conditions from 
        this pool.

        Args:
            all_conditions (Union[pd.DataFrame, np.ndarray]): The pool to sample from.
                This can be a DataFrame or a NumPy array containing all possible experimental conditions.
            models (List): A list of models used to evaluate disagreement. The models should 
                be compatible with the `model_disagreement_sample` function.
            reference_conditions (Union[pd.DataFrame, np.ndarray], optional): Reference conditions 
                that might be used by the sampler to guide the sampling process. Defaults to None.
            num (int, optional): Number of experimental conditions to select. Defaults to 1.

        Returns:
            pd.DataFrame: A DataFrame containing the sampled experimental conditions.
        """
    # Generate a pool of conditions based on model disagreement
    conditions = model_disagreement_sample(
          all_conditions,
          models,
          num_samples=10 # Number of samples to generate from model disagreement
        )
    # Randomly sample `num` conditions from the generated pool
    return conditions.sample(n=num)
