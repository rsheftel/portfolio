"""
Statistical and related functions
"""

from math import sqrt

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.odr import Data, Model, ODR
from scipy.stats import linregress
from sklearn.decomposition import PCA
from wpca import WPCA

from portfolio.math.base import dropna, calc_exponential_function, dispatch_calc


def mean_exp_weighted(
        x: list | NDArray | pd.DataFrame | pd.Series,
        window_length: int,
        half_life: int,
        annualize: bool = True,
        annual_periods: int = 252,
) -> float | pd.Series:
    """
    Exponentially weighted average. The assumption is that the highest index value of X is the most recent and the 0
    index value is the furthest back in time and will receive the lowest weight.

    :param x: any of a list, numpy array, pd.Series of pd.DataFrame
    :param window_length: window for the exponential weighting. The window ignores any date index and assumes that
        each entry in the list/array/DataFrame/Series is equally spaced
    :param half_life: half life of the exponential decay
    :param annualize: if True the annualize the results
    :param annual_periods: number of periods in the vector in a year for annualization, ie: 252 for a daily vector
    :return: float if x is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x, weights):
        x, weights = dropna(x, weights)
        return np.average(x, weights=weights)

    res = calc_exponential_function(x, window_length, half_life, _calc, "mean_exponential_weighted")
    if annualize:
        res = res * annual_periods
    return res


def stdev_exp_weighted(
        x: list | NDArray | pd.DataFrame | pd.Series,
        window_length: int,
        half_life: int,
        annualize: bool = True,
        annual_periods: int = 252,
) -> float | pd.Series:
    """
    Exponentially standard deviation. The assumption is that the highest index value of X is the most recent and the 0
    index value is the furthest back in time and will receive the lowest weight.

    :param x: any of a list, numpy array, pd.Series of pd.DataFrame
    :param window_length: window for the exponential weighting. The window ignores any date index and assumes that
        each entry in the list/array/DataFrame/Series is equally spaced
    :param half_life: half life of the exponential decay
    :param annualize: if True the annualize the results
    :param annual_periods: number of periods in the vector in a year for annualization, ie: 252 for a daily vector
    :return: float if x is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x, weights):
        x, weights = dropna(x, weights)
        average = np.average(x, weights=weights)
        variance = np.average((x - average) ** 2, weights=weights)
        return sqrt(variance)

    res = calc_exponential_function(x, window_length, half_life, _calc, "stdev_exponential_weighted")
    if annualize:
        res = res * sqrt(annual_periods)
    return res


def downside_deviation(
        x: list | NDArray | pd.DataFrame | pd.Series,
        annualize: bool = True,
        annual_periods: int = 252,
) -> float | pd.Series:
    """
    Downside semi-deviation. This is the standard deviation of all the observations less than the MAR, where the
    assumption is the MAR = 0. This excludes any values >= MAR, unlike other calculations which assign those a value
    of zero.

    :param x: any of a list, numpy array, pd.Series of pd.DataFrame
    :param annualize: if True the annualize the results
    :param annual_periods: number of periods in the vector in a year for annualization, ie: 252 for a daily vector
    :return: float if x is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        x = dropna(x)
        x = x[x < 0]
        variance = np.average(x ** 2, weights=None)
        return sqrt(variance)

    res = dispatch_calc(x, _calc, name="downside_deviation")
    if annualize:
        res = res * sqrt(annual_periods)
    return res


def downside_deviation_exp_weighted(
        x: list | NDArray | pd.DataFrame | pd.Series,
        window_length: int,
        half_life: int,
        annualize: bool = True,
        annual_periods: int = 252,
) -> float | pd.Series:
    """
    Downside semi-deviation. The assumption is the MAR = 0. The assumption is that the highest index value of X is the
    most recent and the 0 index value is the furthest back in time and will receive the lowest weight.

    :param x: any of a list, numpy array, pd.Series of pd.DataFrame
    :param window_length: window for the exponential weighting. The window ignores any date index and assumes that
        each entry in the list/array/DataFrame/Series is equally spaced
    :param half_life: half life of the exponential decay
    :param annualize: if True the annualize the results
    :param annual_periods: number of periods in the vector in a year for annualization, ie: 252 for a daily vector
    :return: float if x is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x, weights):
        x, weights = dropna(x, weights)
        filters = x < 0
        x = x[filters]
        weights = np.array(weights)[filters]
        variance = np.average(x ** 2, weights=weights)
        return sqrt(variance)

    res = calc_exponential_function(x, window_length, half_life, _calc, "downside_deviation_exponential_weighted")
    if annualize:
        res = res * sqrt(annual_periods)
    return res


def cagr(prices: list | NDArray | pd.DataFrame | pd.Series, annual_periods: int | float = 252) -> float | pd.Series:
    """
    Given a time series of prices, calculate the compound annual growth rate.

    :param prices: any of a list, numpy array, pd.Series or pd. DataFrame
    :param annual_periods: number of periods in the vector in a year for annualization, ie: 252 for a daily vector
    :return: float if x is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """
    if isinstance(prices, (pd.Series, pd.DataFrame)):
        # Calculate the total number of years
        years = (prices.index[-1] - prices.index[0]).days / 365.25
        # Calculate the ending value to beginning value ratio
        ratio = prices.iloc[-1] / prices.iloc[0]
    else:
        years = (len(prices) - 1) / annual_periods
        ratio = prices[-1] / prices[0]

    res = (ratio ** (1 / years)) - 1
    if isinstance(prices, pd.DataFrame):
        res.name = "cagr"
    return res


def _odr_objective(p, x):
    """*PRIVATE* Basic linear regression 'model' for use with ODR"""
    return (p[0] * x) + p[1]


def orthogonal_distance_regression(x, y, weights=None):
    """
    Calculate the Orthogonal Distance Regression (aka: Total Least Squares) from the x and y list. Return the slope and
    intercept.
    Uses this::  http://blog.rtwilson.com/orthogonal-distance-regression-in-python/
    :param x: list of independent values
    :param y: list of dependent values
    :param weights: list of weights, if None then ignored
    :return: slope, intercept
    """

    linreg = linregress(x, y)
    mod = Model(_odr_objective)
    dat = Data(x, y, weights)
    od = ODR(dat, mod, beta0=linreg[0:2])
    out = od.run()
    return out.beta[0], out.beta[1]


# noinspection PyUnresolvedReferences
def total_least_squares(x, y, weights=None):
    """
    Calculate the Total Least Squares (aka: Orthogonal Distance Regression) from the x and y list. Return the slope and
    intercept. For weighted TLS this library is used: https://github.com/jakevdp/wpca  It has two implementations
    of the weighted TLS either the Delchambre (WPCA) or Bailey (EMPCA), this uses WPCA but replacing the function call
    will change the method. The weights are applied to the result of the error^2 * weight, and as such provide the
    raw weights, no need to square root as that is done in the function.

    :param x: list of independent values
    :param y: list of dependent values
    :param weights: list of weights, if None then ignored
    :return: slope, intercept
    """
    xy = np.array([x, y]).T
    if weights:
        # The weights are used in the sum of squared errors minimization, so take
        # the square root because the PCA function will square them in the solution
        w = [sqrt(i) for i in weights]
        ww = np.array([w, w]).T
        model = WPCA(n_components=1).fit(xy, weights=ww)
    else:
        model = PCA(n_components=1).fit(xy)
    slope = model.components_[0][1] / model.components_[0][0]
    intercept = model.mean_[1] - model.mean_[0] * slope
    return slope, intercept
