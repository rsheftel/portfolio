"""
Financial math functions like Sharpe based on percentage return time series
"""

from math import sqrt
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from portfolio.math import transformation, statistics
from portfolio.math.base import dispatch_calc, dropna, prior_index
from portfolio.math.financial import common
from portfolio.math.financial.common import (
    sharpe,
    sharpe_exp_weighted,
    sortino,
    sortino_exp_weighted,
    conditional_sortino,
    conditional_sortino_exp_weighted,
    omega_ratio,
    robustness as common_robustness,
)
from portfolio.math.transformation import price_index


def __keep_imports():
    """
    this is a dummy function the sole purpose is to retain the imports so they are not removed by the Pycharm Optimize
    imports
    """
    assert sharpe
    assert sharpe_exp_weighted
    assert sortino
    assert sortino_exp_weighted
    assert conditional_sortino
    assert conditional_sortino_exp_weighted
    assert omega_ratio


def total_return(x: list | NDArray | pd.DataFrame | pd.Series) -> float | pd.Series:
    """
    Given a time series of periodic returns, calculate the total return for the entire series by compounding the
    periodic returns

    :param x: any of a list, numpy array, pd.Series or pd. DataFrame
    :return: float if x is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        # Calculate the compounded growth factor
        # Compute the cumulative product to get the compounded growth over time
        # The last element of compounded_growth will give us the total compounded return
        x = dropna(x)
        return np.cumprod(1 + x)[-1] - 1

    return dispatch_calc(x, _calc, name="total_return")


def r_squared(x: list | NDArray | pd.DataFrame | pd.Series) -> float | pd.Series:
    """
    The R-Squared of the VAMI of x (returns) as the response variable and an even sequence the length of x as
    the explanatory. NaN values are dropped and considered to not have been included in the series such that the
    spacing of the explanatory variables is constant. The VAMI line will be created by adding the starting index
    to the day one business day prior to the start of the series.

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :return: float if X is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        x = dropna(x)
        x = transformation.price_index(x, start_index=True)
        if len(x) < 1:
            return np.nan

        rhat = stats.linregress(np.arange(len(x)), x).rvalue
        return rhat ** 2

    return dispatch_calc(x, _calc, name="r_squared")


def k_ratio(x: list | NDArray | pd.DataFrame | pd.Series) -> float | pd.Series:
    """
    The K-Ratio using the original 1996 formulation by Lars Kestner. The x series is a return per period which is
    converted to a VAMI price index.
    https://www.dropbox.com/scl/fi/myfpukl78c2lgcrc1irf4/K-Ratio-in-Excel.pdf?rlkey=ly2z9cnp1swqdr8tm3e4iwitw&st=q7ulvlyb&dl=0

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :return: float if X is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        x = dropna(x)
        x = transformation.price_index(x, start_index=True)
        res = stats.linregress(np.arange(len(x)), x)
        return res.slope / (res.stderr * sqrt(len(x)))

    return dispatch_calc(x, _calc, name="k_ratio")


def robustness(x: list | NDArray | pd.DataFrame | pd.Series) -> float | pd.Series:
    """
    Robustness of the returns defined as the percentage of days, when ordered from the best to the worst, that can be
    eliminated and the total return would still be above zero. The higher the number, the more missed days the returns
    can withstand before turning negative.

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :return: float if X is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        # transform the returns to a pnl
        x = dropna(x)
        prices = transformation.price_index(x, start_index=True)
        pnl = np.diff(prices)
        return common_robustness(pnl)

    return dispatch_calc(x, _calc, name="robustness")


def drawdown_series(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Calculates the `drawdown <https://www.investopedia.com/terms/d/drawdown.asp>`_ series.

    This returns a series representing a drawdown. When the price is at all time highs, the drawdown is 0. However,
    when prices are below high water marks, the drawdown series = current / hwm - 1

    The max drawdown can be obtained by simply calling .min() on the result (since the drawdown series is negative)
    Method ignores all gaps of NaN's in the price series.

    """
    # make a copy so that we don't modify original data
    drawdown = prices.copy()

    # Fill NaN's with previous values
    drawdown = drawdown.ffill()

    # Ignore problems with NaN's in the beginning
    drawdown[np.isnan(drawdown)] = -np.inf

    # Rolling maximum
    if isinstance(drawdown, pd.DataFrame):
        roll_max = pd.DataFrame()
        for col in drawdown:
            roll_max[col] = np.maximum.accumulate(drawdown[col])
    else:
        roll_max = np.maximum.accumulate(drawdown)

    drawdown = drawdown / roll_max - 1.0
    return drawdown


def drawdown_details(returns: list | NDArray | pd.DataFrame | pd.Series) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """
    Given an input series or dataframe of returns, will return a dataframe, or dict of dataframes for an input
    dataframe, of the drawdowns with their start and end index, length and max drawdown amount.
    If the input has a DateTimeIndex the length of the drawdown returned is in business days, not accounting for
    holidays.

    :param returns: any of a list, numpy array, pd.Series or pd.DataFrame
    :return: pd.DataFrame if returns is a list, array or a pd.Series. A dict of pd.DataFrame if returns is a pd.DataFrame
    """
    if isinstance(returns, pd.DataFrame):
        res = {}
        for column in returns.columns:
            res[column] = drawdown_details(returns[column])
        return res

    # make sure the returns is pd.Series and determine the start_index
    if isinstance(returns, (list, np.ndarray)):
        returns = pd.Series(returns, index=range(1, len(returns) + 1))

    equity = price_index(returns, start_index=prior_index(returns))
    drawdown = drawdown_series(equity)
    return common.drawdown_details(drawdown)


def average_drawdown(x: list | NDArray | pd.DataFrame | pd.Series) -> float | pd.Series:
    """
    Average drawdown

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :return: float if X is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """
    return common.average_drawdown(x, drawdown_details)


def maximum_drawdown(x: list | NDArray | pd.DataFrame | pd.Series) -> float | pd.Series:
    """
    Maximum drawdown

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :return: float if X is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """
    return common.maximum_drawdown(x, drawdown_details)


def average_drawdown_time(x: list | NDArray | pd.DataFrame | pd.Series) -> float | pd.Series:
    """
    Average drawdown time length in days

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :return: float if X is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """
    return common.average_drawdown_time(x, drawdown_details)


def average_recovery_time(x: list | NDArray | pd.DataFrame | pd.Series) -> float | pd.Series:
    """
    Average recovery time of drawdowns

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :return: float if X is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """
    return common.average_recovery_time(x, drawdown_details)


def plunge_ratio(x: list | NDArray | pd.DataFrame | pd.Series) -> float | pd.Series:
    """
    Plunge ratio is the average percentage of time during a drawdown that is spent in recovery. If the number is -1.0
    that means that the maximum drawdown point was on the first day and the entire rest of the drawdown is spent in
    recovery. If the number is 0.0 that means that the drawdown was gradual reaching the maximum on the last day prior
    to recovering all the drawdown on the last day. The more negative the number, the sharper the losses, the closer
    to 0.0 the more grinding the losses.

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :return: float if X is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """
    return common.plunge_ratio(x, drawdown_details)


def plunge_ratio_exp_weighted(
        x: list | NDArray | pd.DataFrame | pd.Series,
        window_length: int,
        half_life: int,
) -> float | pd.Series:
    """
    Exponentially weighted Plunge Ratio.

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :param window_length: window for the exponential weighting. The window ignores any date index and assumes that
        each entry in the list/array/DataFrame/Series is equally spaced
    :param half_life: half life of the exponential decay
    :return: float if X is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """
    return common.plunge_ratio_exp_weighted(x, window_length, half_life, drawdown_details)


def calmar_ratio(x: list | NDArray | pd.DataFrame | pd.Series, annual_periods: int = 252) -> float | pd.Series:
    """
    Calmar ratio

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :param annual_periods: number of periods in the vector in a year for annualization, ie: 252 for a daily vector
        This is only used if the input x does not have a DateTimeIndex
    :return: float if X is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        x = dropna(x)
        avg_annual = statistics.cagr(price_index(x, start_index=prior_index(x)))
        max_drawdown = maximum_drawdown(x)
        return avg_annual / abs(max_drawdown)

    return dispatch_calc(x, _calc, name="calmar_ratio", as_series=True)


def _prep_factor_correlation(
        returns: pd.DataFrame,
        factor: pd.DataFrame,
        periods: int | None = None,
) -> pd.DataFrame:
    """
    prepare the DataFrames for correlation

    :param returns: pd.DataFrame of returns
    :param factor: pd.DataFrame of factor levels, if empty will be just the pnl correlations
    :param periods: if None will return a level-on-level correlation, if a number will return a change-on-change
        correlation
    :return: concatenated and differenced pd.DataFrame
    """
    # check the column names are unique
    assert (
            len(returns.columns.intersection(factor.columns)) == 0
    ), "duplicate columns in data and factors"

    # create the levels for both and combine
    equity = price_index(returns, start_index=prior_index(returns))
    factor_levels = factor.copy()
    combined = pd.concat([equity, factor_levels], axis=1)

    if periods:
        combined = combined.pct_change(periods, fill_method=None)
    return combined


def correlation(
        returns: pd.DataFrame,
        factor: pd.DataFrame = pd.DataFrame(),
        periods: int | None = None,
        method: Literal["pearson", "kendall", "spearman"] = "pearson",
) -> pd.DataFrame:
    """
    Returns a correlation matrix between the returns and the factors (if provided). If the periods is None it will be a
    levels correlation as the pnl are converted to index. If periods is a number then it will be a change-on-change
    correlation. To compute the correlations, the returns are converted to a price index assuming compounding of
    returns. If the factors are converted to change-on-change then it will be a percentage change to be consistent
    with the returns.

    :param returns: pd.DataFrame of returns
    :param factor: pd.DataFrame of factor levels, if empty will be just the pnl correlations
    :param periods: if None will return a level-on-level correlation, if a number will return a change-on-change
        correlation
    :param method: method to use for the correlation, default is "pearson" but can be "kendall" or "spearman"
    :return: pd.DataFrame of correlation matrix
    """
    combined = _prep_factor_correlation(returns, factor, periods)
    return combined.corr(method=method)


def correlation_pvalues(
        returns: pd.DataFrame,
        factor: pd.DataFrame,
        periods: int | None = None,
        method: Literal["pearson", "kendall", "spearman"] = "pearson",
) -> pd.DataFrame:
    """
    Returns a correlation p-values matrix between the pnls and the factors (if provided).
    If the periods is None it will be a levels correlation as the pnl are converted to index.
    If periods is a number then it will be a change-on-change correlation. To compute the correlations, the returns are
    converted to a price index assuming compounding of returns. If the factors are converted to change-on-change then
    they will be a percentage change to be consistent with the returns.

    :param returns: pd.DataFrame of returns
    :param factor: pd.DataFrame of factor levels, if empty will be just the pnl correlations
    :param periods: if None will return a level-on-level correlation, if a number will return a change-on-change
        correlation
    :param method: method to use for the correlation, default is "pearson" but can be "kendall" or "spearman"
    :return: pd.DataFrame of correlation matrix
    """
    combined = _prep_factor_correlation(returns, factor, periods)
    return statistics.correlation_pvalues(combined, method=method)


def add_factors_to_drawdown_details(
        drawdown_details_df: pd.DataFrame, factors: pd.DataFrame
) -> pd.DataFrame:
    """
    Adds factors information to the drawdown details dataframe and return the new drawdown details dataframe.
    The factors should be levels. The changes in the factors will be expressed as percentage changes.

    :param drawdown_details_df: pd.DataFrame containing drawdown details in the form returned by drawdown_details()
    :param factors: pd.DataFrame of time series of factor levels
    :param as_percent: if True will calculate the factor changes as percentages, otherwise will calculate as absolute
    :return: pd.DataFrame
    """
    return common.add_factors_to_drawdown_details(
        drawdown_details_df, factors, as_percent=True
    )
