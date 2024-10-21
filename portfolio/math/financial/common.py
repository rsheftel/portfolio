"""
Functions that are common for both PnL and Returns, or supporting functions for both
"""

from math import sqrt

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from portfolio.math.base import dispatch_calc, dropna
from portfolio.math.statistics import (
    mean_exp_weighted,
    stdev_exp_weighted,
    downside_deviation,
    downside_deviation_exp_weighted
)


def sharpe(
    x: list | NDArray | pd.DataFrame | pd.Series, annualize: bool = True, annual_periods: int = 252
) -> float | pd.Series:
    """
    Sharpe ratio assuming no risk-free rate

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :param annualize: if True the annualize the results
    :param annual_periods: number of periods in the vector in a year for annualization, ie: 252 for a daily vector
    :return: float if X is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        return np.nanmean(x) / np.nanstd(x)

    res = dispatch_calc(x, _calc, name="sharpe")
    if annualize:
        res = res * annual_periods / sqrt(annual_periods)
    return res


def sharpe_exp_weighted(
    x: list | NDArray | pd.DataFrame | pd.Series,
    window_length: int,
    half_life: int,
    annualize: bool = True,
    annual_periods: int = 252,
) -> float | pd.Series:
    """
    Exponentially weighted Sharpe ratio.

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :param window_length: window for the exponential weighting. The window ignores any date index and assumes that
        each entry in the list/array/DataFrame/Series is equally spaced
    :param half_life: half life of the exponential decay
    :param annualize: if True the annualize the results
    :param annual_periods: number of periods in the vector in a year for annualization, ie: 252 for a daily vector
    :return: float if X is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """
    res = mean_exp_weighted(x, window_length, half_life, annualize, annual_periods) / stdev_exp_weighted(
        x, window_length, half_life, annualize, annual_periods
    )
    if isinstance(res, pd.Series):
        res.name = "sharpe_exponential_weighted"
    return res


def sortino(
        x: list | NDArray | pd.DataFrame | pd.Series,
        annualize: bool = True,
        annual_periods: int = 252,
) -> float | pd.Series:
    """
    Sortino ratio. Assume the risk-free rate is 0.

    :param x: any of a list, numpy array, pd.Series of pd.DataFrame
    :param annualize: if True the annualize the results
    :param annual_periods: number of periods in the vector in a year for annualization, ie: 252 for a daily vector
    :return: float if x is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        x = dropna(x)
        negative_returns = np.minimum(x, 0.0)
        std = np.std(negative_returns)
        return np.mean(x) / std

    res = dispatch_calc(x, _calc, name="sortino")
    if annualize:
        res = res * annual_periods / sqrt(annual_periods)
    return res


def sortino_exp_weighted(
        x: list | NDArray | pd.DataFrame | pd.Series,
        window_length: int,
        half_life: int,
        annualize: bool = True,
        annual_periods: int = 252,
) -> float | pd.Series:
    """
    Exponentially weighted Sortino ratio.

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :param window_length: window for the exponential weighting. The window ignores any date index and assumes that
        each entry in the list/array/DataFrame/Series is equally spaced
    :param half_life: half life of the exponential decay
    :param annualize: if True the annualize the results
    :param annual_periods: number of periods in the vector in a year for annualization, ie: 252 for a daily vector
    :return: float if X is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    mean = mean_exp_weighted(x, window_length, half_life, annualize, annual_periods)
    negative_returns = np.minimum(x, 0.0)
    std = stdev_exp_weighted(negative_returns, window_length, half_life, annualize, annual_periods)
    res = mean / std
    if isinstance(res, pd.Series):
        res.name = "sortino_exponential_weighted"
    return res


def conditional_sortino(
        x: list | NDArray | pd.DataFrame | pd.Series,
        annualize: bool = True,
        annual_periods: int = 252,
) -> float | pd.Series:
    """
    Conditional Sortino ratio. Assume the risk-free rate is 0. Differs from the class Sortino in that the divisor is
    the standard deviation of only the negative numbers, it excludes the position numbers. A classic Sortino treats the
    positive numbers as zeros and includes in the standard deviation.

    :param x: any of a list, numpy array, pd.Series of pd.DataFrame
    :param annualize: if True the annualize the results
    :param annual_periods: number of periods in the vector in a year for annualization, ie: 252 for a daily vector
    :return: float if x is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        return np.nanmean(x)

    res = dispatch_calc(x, _calc, name="conditional_sortino")
    if annualize:
        res = res * annual_periods
    res = res / downside_deviation(x, annualize=annualize, annual_periods=annual_periods)
    if isinstance(res, pd.Series):
        res.name = "conditional_sortino"
    return res


def conditional_sortino_exp_weighted(
        x: list | NDArray | pd.DataFrame | pd.Series,
        window_length: int,
        half_life: int,
        annualize: bool = True,
        annual_periods: int = 252,
) -> float | pd.Series:
    """
    Exponentially weighted Conditional Sortino ratio.

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :param window_length: window for the exponential weighting. The window ignores any date index and assumes that
        each entry in the list/array/DataFrame/Series is equally spaced
    :param half_life: half life of the exponential decay
    :param annualize: if True the annualize the results
    :param annual_periods: number of periods in the vector in a year for annualization, ie: 252 for a daily vector
    :return: float if X is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """
    res = mean_exp_weighted(x, window_length, half_life, annualize, annual_periods) / downside_deviation_exp_weighted(
        x, window_length, half_life, annualize, annual_periods
    )
    if isinstance(res, pd.Series):
        res.name = "conditional_sortino_exponential_weighted"
    return res
