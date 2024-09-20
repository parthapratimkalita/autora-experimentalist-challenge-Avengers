import random
import numpy as np
import pandas as pd
import pytest
import torch
from autora.experimentalist.autora_experimentalist_example import sample


class ModelA:
    def predict(self, X):
        return X.sum(axis=1)


class ModelB:
    def predict(self, X):
        return X.mean(axis=1)


@pytest.fixture()
def seed():
    random.seed(180)
    torch.manual_seed(180)
    np.random.seed(180)
    yield


def test_output_dimensions_with_single_row(seed):
    X = np.random.rand(10, 4)
    models = [ModelA(), ModelB()]
    X_new = sample(X, models, num=1)
    assert X_new.shape == (1, X.shape[1])


def test_output_dimensions_with_single_column(seed):
    X = np.random.rand(10, 1)
    models = [ModelA(), ModelB()]
    X_new = sample(X, models, num=2)
    assert X_new.shape == (2, X.shape[1])


def test_output_dimensions_with_large_num(seed):
    X = np.random.rand(10, 2)
    models = [ModelA(), ModelB()]
    X_new = sample(X, models, num=10)
    assert X_new.shape == (10, X.shape[1])


def test_output_dimensions_with_zero_num(seed):
    X = np.random.rand(10, 3)
    models = [ModelA(), ModelB()]
    X_new = sample(X, models, num=0)
    assert X_new.shape == (0, X.shape[1])


def test_output_with_dataframe_input(seed):
    X = pd.DataFrame(np.random.rand(10, 3))
    models = [ModelA(), ModelB()]
    X_new = sample(X, models, num=2)
    assert isinstance(X_new, pd.DataFrame)
    assert X_new.shape == (2, X.shape[1])


def test_output_with_reference_conditions(seed):
    X = np.random.rand(10, 2)
    reference_conditions = np.random.rand(5, 2)
    models = [ModelA(), ModelB()]
    X_new = sample(X, models, reference_conditions, num=2)
    assert X_new.shape == (2, X.shape[1])
