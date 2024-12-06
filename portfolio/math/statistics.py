"""
Statistical and related functions
"""

from math import sqrt
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.odr import Data, Model, ODR
from scipy.stats import linregress, ttest_1samp, ttest_ind
from sklearn.decomposition import PCA
from wpca import WPCA

from portfolio.math.base import dropna, calc_exponential_function, dispatch_calc
from portfolio.utils import reindex_superset


def mean(x: list | NDArray | pd.DataFrame | pd.Series, name: str = None) -> float | pd.Series:
    """
    Given a list or series calculate the mean dropping all NaNs

    :param x: any of a list, numpy array, pd.Series or pd. DataFrame
    :param name: optional name for the pd.Series result if x is a pd.DataFrame. If None will use default
    :return: float if x is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        return np.nanmean(x)

    name = "mean" if name is None else name
    return dispatch_calc(x, _calc, name=name)


def mean_exp_weighted(
        x: list | NDArray | pd.DataFrame | pd.Series,
        window_length: int,
        half_life: int,
        annualize: bool = True,
        annual_periods: int = 252,
        name: str = None,
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
    :param name: optional name for the pd.Series result if x is a pd.DataFrame. If None will use default
    :return: float if x is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x, weights):
        x, weights = dropna(x, weights)
        return np.average(x, weights=weights)

    name = "mean_exponential_weighted" if name is None else name
    res = calc_exponential_function(x, window_length, half_life, _calc, name=name)
    if annualize:
        res = res * annual_periods
    return res


def min(x: list | NDArray | pd.DataFrame | pd.Series, name: str = None) -> float | pd.Series:
    """
    Given a list or series calculate the min dropping all NaNs

    :param x: any of a list, numpy array, pd.Series or pd. DataFrame
    :param name: optional name for the pd.Series result if x is a pd.DataFrame. If None will use default
    :return: float if x is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        return np.nanmin(x)

    name = "min" if name is None else name
    return dispatch_calc(x, _calc, name=name)


def max(x: list | NDArray | pd.DataFrame | pd.Series, name: str = None) -> float | pd.Series:
    """
    Given a list or series calculate the max dropping all NaNs

    :param x: any of a list, numpy array, pd.Series or pd. DataFrame
    :param name: optional name for the pd.Series result if x is a pd.DataFrame. If None will use default
    :return: float if x is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        return np.nanmax(x)

    name = "max" if name is None else name
    return dispatch_calc(x, _calc, name=name)


def stdev(
        x: list | NDArray | pd.DataFrame | pd.Series, annualize: bool = True, annual_periods: int = 252,
        name: str = None
) -> float | pd.Series:
    """
    Given a list or series calculate the standard deviation dropping all nans

    :param x: any of a list, numpy array, pd.Series or pd. DataFrame
    :param annualize: if True the annualize the results
    :param annual_periods: number of periods in the vector in a year for annualization, ie: 252 for a daily vector
    :param name: optional name for the pd.Series result if x is a pd.DataFrame. If None will use default
    :return: float if x is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        return np.nanstd(x)

    name = "standard_deviation" if name is None else name
    res = dispatch_calc(x, _calc, name=name)
    if annualize:
        res = res * sqrt(annual_periods)
    return res


def stdev_exp_weighted(
        x: list | NDArray | pd.DataFrame | pd.Series,
        window_length: int,
        half_life: int,
        annualize: bool = True,
        annual_periods: int = 252,
        name: str = None,
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
    :param name: optional name for the pd.Series result if x is a pd.DataFrame. If None will use default
    :return: float if x is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x, weights):
        x, weights = dropna(x, weights)
        average = np.average(x, weights=weights)
        variance = np.average((x - average) ** 2, weights=weights)
        return sqrt(variance)

    name = "standard_deviation_exponential_weighted" if name is None else name
    res = calc_exponential_function(x, window_length, half_life, _calc, name)
    if annualize:
        res = res * sqrt(annual_periods)
    return res


def downside_deviation(
        x: list | NDArray | pd.DataFrame | pd.Series, annualize: bool = True, annual_periods: int = 252,
        name: str = None
) -> float | pd.Series:
    """
    Downside semi-deviation. This is the standard deviation of all the observations less than the MAR, where the
    assumption is the MAR = 0. This excludes any values >= MAR, unlike other calculations which assign those a value
    of zero.

    :param x: any of a list, numpy array, pd.Series of pd.DataFrame
    :param annualize: if True the annualize the results
    :param annual_periods: number of periods in the vector in a year for annualization, ie: 252 for a daily vector
    :param name: optional name for the pd.Series result if x is a pd.DataFrame. If None will use default
    :return: float if x is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        x = dropna(x)
        x = x[x < 0]
        variance = np.average(x ** 2, weights=None)
        return sqrt(variance)

    name = "downside_deviation" if name is None else name
    res = dispatch_calc(x, _calc, name=name)
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
    :param annual_periods: number of periods in the vector in a year for annualization, ie: 252 for a daily vector.
        This is only used if the prices series does not have a DateTimeIndex, if so that is used.
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


def ttest_1sample_1side(x: list | NDArray | pd.DataFrame | pd.Series, name: str = None) -> float | pd.Series:
    """
    One-sided one-sample t-test p-value

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :param name: optional name for the pd.Series result if x is a pd.DataFrame. If None will use default
    :return: float if x is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        x = dropna(x)
        return ttest_1samp(x, 0, alternative="greater").pvalue

    name = "ttest_1sample_1side_pvalue" if name is None else name
    return dispatch_calc(x, _calc, name=name)


def ttest_2sample_2side(x_1: list | NDArray | pd.Series, x_2: list | NDArray | pd.Series) -> float | pd.Series:
    """
    Two-sided two-sample t-test p-value. If both x_1 and x_2 are series they will first be aligned on the superset
    of their index to do the test only on the values for index that exist in both series.

    :param x_1: any of a list, numpy array, pd.Series
    :param x_2: any of a list, numpy array, pd.Series
    :return: float if x is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """
    if isinstance(x_1, pd.Series) and isinstance(x_2, pd.Series):
        x_1 = reindex_superset(x_1, x_2)
        x_2 = reindex_superset(x_2, x_1)

    x_1 = x_1.values if isinstance(x_1, pd.Series) else x_1
    x_2 = x_2.values if isinstance(x_2, pd.Series) else x_2
    # remove the nan from both
    x_1, x_2 = dropna(x_1, x_2)
    x_2, x_1 = dropna(x_2, x_1)
    return ttest_ind(x_1, x_2, equal_var=False, alternative="two-sided").pvalue


def first_date(x: pd.Series | pd.DataFrame) -> pd.Series | Any:
    """
    For a given pd.Series of pd.DataFrame will return the index value of the first non NaN value

    :param x: pd.Series or pd.DataFrame
    :return: single value if x is pd.Series, pd.Series if x is a pd.DataFrame
    """
    if isinstance(x, pd.Series):
        first = x.first_valid_index()
        if first is None:
            return None
        return first

    elif isinstance(x, pd.DataFrame):
        res = pd.Series(name="first_date")
        for col in x.columns:
            res[col] = first_date(x[col])
        return res
    else:
        raise AttributeError("parameter x must be pd.Series or pd.DataFrame")


def last_date(x: pd.Series | pd.DataFrame) -> pd.Series | Any:
    """
    For a given pd.Series of pd.DataFrame will return the index value of the last non NaN value

    :param x: pd.Series or pd.DataFrame
    :return: single value if x is pd.Series, pd.Series if x is a pd.DataFrame
    """
    if isinstance(x, pd.Series):
        first = x.last_valid_index()
        if first is None:
            return None
        return first

    elif isinstance(x, pd.DataFrame):
        res = pd.Series(name="last_date")
        for col in x.columns:
            res[col] = last_date(x[col])
        return res
    else:
        raise AttributeError("parameter x must be pd.Series or pd.DataFrame")


def observations_count(x: list | NDArray | pd.DataFrame | pd.Series, name: str = None) -> float | pd.Series:
    """
    Given a list or series calculate the number of non NaN values

    :param x: any of a list, numpy array, pd.Series or pd. DataFrame
    :param name: optional name for the pd.Series result if x is a pd.DataFrame. If None will use default
    :return: float if x is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        return np.count_nonzero(~np.isnan(x))

    name = "observations_count" if name is None else name
    return dispatch_calc(x, _calc, name=name)


def percentage_above(
        x: list | NDArray | pd.DataFrame | pd.Series, threshold: int | float = 0, name: str = None
) -> float | pd.Series:
    """
    Given a list or series calculate the max dropping all NaNs

    :param x: any of a list, numpy array, pd.Series or pd. DataFrame
    :param name: optional name for the pd.Series result if x is a pd.DataFrame. If None will use default
    :param threshold: threshold value
    :return: float if x is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        x = np.asarray(x)
        x = dropna(x)
        return np.sum(x > threshold) / x.size

    name = "percentage_above" if name is None else name
    return dispatch_calc(x, _calc, name=name)
