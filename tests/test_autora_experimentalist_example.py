import numpy as np
import pandas as pd
import pytest
from autora.experimentalist.autora_experimentalist_example import sample


@pytest.fixture
def seed():
    import random
    import torch
    random.seed(180)
    torch.manual_seed(180)


def output_dimensions_with_single_row(seed):
    X = np.array([[1, 2, 3, 4]])
    models = []
    X_new = sample(X, models, num=1)
    assert X_new.shape == (1, X.shape[1])


def output_dimensions_with_single_column(seed):
    X = np.array([[1], [2], [3], [4]])
    models = []
    X_new = sample(X, models, num=2)
    assert X_new.shape == (2, X.shape[1])


def output_dimensions_with_empty_array(seed):
    X = np.array([[]])
    models = []
    X_new = sample(X, models, num=0)
    assert X_new.shape == (0, X.shape[1])


def output_dimensions_with_large_num(seed):
    X = np.array([[1, 2], [3, 4], [5, 6]])
    models = []
    X_new = sample(X, models, num=10)
    assert X_new.shape == (10, X.shape[1])


def output_dimensions_with_zero_num(seed):
    X = np.array([[1, 2, 3], [4, 5, 6]])
    models = []
    X_new = sample(X, models, num=0)
    assert X_new.shape == (0, X.shape[1])


def output_with_dataframe_input(seed):
    X = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    models = []
    X_new = sample(X, models, num=2)
    assert isinstance(X_new, pd.DataFrame)
    assert X_new.shape == (2, X.shape[1])


def output_with_reference_conditions(seed):
    X = np.array([[1, 2], [3, 4], [5, 6]])
    reference_conditions = np.array([[7, 8], [9, 10]])
    models = []
    X_new = sample(X, models, reference_conditions, num=2)
    assert X_new.shape == (2, X.shape[1])
