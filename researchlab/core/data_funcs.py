"""
data_funcs
Functions to process and check dataframe
"""

import pandas as pd
import numpy as np
from typing import Union
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

pd_type = Union[pd.Series, pd.DataFrame]

def align_indices(target, features):
    """ Aligning dataframe indices together """
    common_index = target.index.intersection(features.index)
    return target.loc[common_index], features.loc[common_index]

class AlignmentError(Exception):
    pass

def check_alignment(target, features):
    if len(target) != len(features):
        raise AlignmentError('Target and features are not aligned')

def filter_date_range(obj: pd_type, start_date = None, end_date = None) -> pd_type:
    """
    Filter a pandas Series or DataFrame with a `datetime.date` index by a specified date range.
    Returns a copy of the filtered object for data safety.
    """
    if start_date is not None: obj = obj.loc[start_date:]
    if end_date is not None: obj = obj.loc[:end_date]
    return obj.copy()

def winsorize_data(dataseries: pd.Series, threshold: int=3) -> pd.Series:
    """
    Winsorize the data series to remove outliers.
    """
    means = dataseries.mean()
    stds = dataseries.std()
    dataseries = np.minimum(dataseries, means + threshold * stds)
    dataseries = np.maximum(dataseries, means - threshold * stds)
    return dataseries

def penalised_accuracy(y_true, y_pred, lambda_penalty=0.01):
    """ Penalised accuracy function for frequent prediction changes """
    changes = np.sum(y_pred[1:] != y_pred[:-1])
    return np.mean(y_true == y_pred) - lambda_penalty * changes

class StandardScalerPD(BaseEstimator):
    """
    Provides support for pandas DataFrame input/output with the `StandardScaler()` class.
    This class extends the functionality of the standard `StandardScaler` by ensuring that
    the input and output are handled as pandas DataFrames, preserving index and column labels.
    """
    def init_scaler(self):
        scaler = StandardScaler().set_output(transform='pandas')
        return scaler

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def fit(self, X: pd.DataFrame):
        self.scaler = self.init_scaler().fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.scaler.transform(X)

# reviewed
class BaseDataClipper(BaseEstimator):
    """
    Base class for data clippers.
    This class implements the `.transform()` and `.fit_transform()` methods, but leaves the `.fit()`
    method to be implemented in subclasses. It is designed to clip data values within a specified range.
    Should be inherited by other classes that define the clipping bounds.
    """
    def __init__(self) -> None:
        self.lb = None
        self.ub = None

    def fit(self, X: pd.DataFrame):
        raise NotImplementedError()

    def fit_transform(self, X: pd.DataFrame):
        return self.fit(X).transform(X)

    def transform(self, X):
        if self.ub is None and self.lb is None: return X
        return np.clip(X, self.lb, self.ub)

# reviewed
class DataClipperStd(BaseDataClipper):
    """
    Data clipper based on feature standard deviation.
    This class performs winsorization of the data, clipping it within a specified multiple of the
    feature's standard deviation. The clipping bounds are defined as:
    lower bound = mean - (mul * std)
    upper bound = mean + (mul * std)
    """
    def __init__(self, mul: float = 3.) -> None:
        super().__init__()
        self.mul = mul

    def fit(self, X):
        mul = self.mul
        assert mul > 0, "The multiplier `mul` must be positive."
        mean, std = X.mean(axis=0), X.std(axis=0, ddof=0)
        mean = mean.to_numpy()
        std = std.to_numpy()
        self.lb = mean - mul * std
        assert isinstance(self.lb, np.ndarray)
        self.ub = mean + mul * std
        assert isinstance(self.ub, np.ndarray)
        return self
