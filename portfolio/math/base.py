"""
Base functions used in the other math functions
"""

import math
from functools import cache

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@cache
def exponential_weights(window_length, half_life):
    """
    Calculate a list of exponential weights to use in weighted regressions or other functions. The 0 index value is
    the most recent or highest weight.

    :param window_length: number of observations to use
    :param half_life: the half life for the exponential weighting

    :return list of weights
    """
    return [
        (-1 / half_life) * math.log(0.5) * math.exp((1 / half_life) * math.log(0.5) * t)
        for t in range(1, window_length + 1)
    ]


def dropna(
        x: list | NDArray | pd.Series, other: list | NDArray | pd.Series | None = None
) -> (list | NDArray | pd.Series, list | NDArray | pd.Series):
    """
    Finds all the np.nan or None in the x array and removes the elements from the x list and the other list if provided.

    :param x: list, numpy array, or pd.Series to find the np.nans or None and remove the elements from the x list
    :param other: (optional) list, numpy array or pd.Series to remove the elements this list that are nan or None in
        the x list
    :return: tuple of x, other lists with the elements removed if other is provided, otherwise just the filtered x
    """
    x_is_list = isinstance(x, list)
    other_is_list = isinstance(other, list)
    if isinstance(x, pd.Series):
        nans = x.isna().values
    else:
        # convert to numpy array, replace the None with np.nan in a list or np.array
        array = np.array(x)
        array = array.astype(float)
        array[array == None] = np.nan
        array = array.astype(float)
        nans = np.isnan(array)
        x = np.asanyarray(x)
    if other_is_list:
        other = np.asanyarray(other)
    x_filtered = x[~nans]
    other_filtered = other[~nans] if other is not None else None
    if x_is_list:
        x_filtered = list(x_filtered)
    if other_is_list:
        other_filtered = list(other_filtered)
    if other is None:
        return x_filtered
    return x_filtered, other_filtered


def trimmed_weights(x, window_length: int, half_life: int):
    """
    Returns a normalized length of the x vector and a vector of weights so that they have equal length.

    :param x: any of a list, numpy array, pd.Series of pd.DataFrame
    :param window_length: window for the exponential weighting. The window ignores any date index and assumes that
        each entry in the list/array/DataFrame/Series is equally spaced
    :param half_life: half life of the exponential decay
    :return: float if x is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """
    weights = list(reversed(exponential_weights(window_length, half_life)))
    if len(x) < len(weights):
        weights = weights[-len(x):]
    if window_length < len(x):
        x = x[-window_length:]
    return x, weights


def dispatch_calc(
        x: list | NDArray | pd.DataFrame | pd.Series, function, name: str = None, as_series: bool = False, **kwargs
) -> float | pd.Series:
    """
    Given a vector or array x, of type list, np.array, pd.Series or pd.DataFrame, this will apply the function
    _function using the parameters **kwargs and return the result. For lists, np.array and pd.Series the result
    is a float, for pd.DataFrame the result is a pd.Series with the index the column names of x and the name of the
    pd.Series is the name parameter if supplied, otherwise the name of the function.

    :param x: any of a list, numpy array, pd.Series of pd.DataFrame
    :param function: calculation function that has the argument signature of f(x, weights)
    :param name: Name for the return pd.Series if x is a pd.DataFrame
    :param as_series: if True a Series or Dataframe will be passed as a Series with index to the function
    :param kwargs: arguments to pass to _function
    :return: float if x is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """
    if isinstance(x, pd.DataFrame):
        res = {}
        name = name or function.__name__
        for column in x.columns:
            if as_series:
                res[column] = function(x[column], **kwargs)
            else:
                res[column] = function(x[column].values, **kwargs)
        return pd.Series(res, name=name)
    elif isinstance(x, pd.Series):
        if as_series:
            return function(x, **kwargs)
        else:
            return function(x.values, **kwargs)
    else:
        return function(x, **kwargs)


def rolling(
        x: pd.DataFrame | pd.Series, function, window: int | str, min_periods: int = 1, **kwargs
) -> pd.DataFrame | pd.Series:
    """
    Performs a rolling window apply of a function to a pd.DataFrame or pd.Series. The window can be expressed as an
    integer number of rows, or a date period string, see there:
    (https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases). For the early entries
    that are less than window in size, the total number of available rows will be used as long as that number is
    greater than or equal to the min_periods

    :param x: pd.DataFrame or pd.Series
    :param function: function to apply to each column in the DataFrame or the entries in the Series. The function
        should have the first parameter take a numpy array
    :param window: size of the rolling window in integer or date period string. Currently only string date periods of
        daily or less (like minute) work.
    :param min_periods: minimum number of entries
    :param kwargs: kwargs to pass to the function
    :return: pd.DataFrame or pd.Series
    """
    return x.rolling(window=window, min_periods=min_periods).apply(function, raw=True, kwargs=kwargs)


def calc_exponential_function(
        x: list | NDArray | pd.DataFrame | pd.Series,
        window_length: int,
        half_life: int,
        _function,
        name: str,
) -> float | pd.Series:
    """
    Private function that will take in the values to calculate the exponential function on, the parameters of the
    exponential weighting (window_length and half_life) and the calculation function.

    The assumption is that the highest index value of X is the most recent and the 0 index value is the furthest back
    in time and will receive the lowest weight.

    :param x: any of a list, numpy array, pd.Series of pd.DataFrame
    :param window_length: window for the exponential weighting. The window ignores any date index and assumes that
        each entry in the list/array/DataFrame/Series is equally spaced
    :param half_life: half life of the exponential decay
    :param _function: calculation function that has the argument signature of f(x, weights)
    :param name: Name for the return pd.Series if x is a pd.DataFrame
    :return: float if x is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """
    x, weights = trimmed_weights(x, window_length, half_life)
    return dispatch_calc(x, _function, name, weights=weights)
