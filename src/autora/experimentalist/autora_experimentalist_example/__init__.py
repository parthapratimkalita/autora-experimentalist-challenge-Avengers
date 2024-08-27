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
    Add a description of the sampler here.

    Args:
        num: number of experimental conditions to select
        all_conditions: The pool to sample from.
            Attention: `conditions` is a field of the standard state
        models: The sampler might use output from the theorist.
            Attention: `models` is a field of the standard state
        reference_conditions: The sampler might use reference conditions

    Returns:
        Sampled pool of experimental conditions
    """
    conditions = model_disagreement_sample(
          all_conditions,
          models,
          num_samples=10
        )
    return conditions.sample(n=num)