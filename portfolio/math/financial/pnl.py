"""
Financial math functions like Sharpe based on PnL time series
"""

from math import sqrt

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

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
    robustness,
)
from portfolio.utils import match_index



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
    assert robustness



def risk_capital(
        downside_df: pd.Series | pd.DataFrame, target_risk_percent: float, rebalance_freq: str = "ME"
) -> pd.Series | pd.DataFrame:
    """
    Given a time series of downside risk measures, either a single series in a pd.Series or many series in a
    pd.DataFrame, and the target percent risk, calculate the risk capital ($) at every rebalance frequency. The downside
    risk measures are in $PnL and the target risk a percentage, such that the risk capital on that date is
    downside_risk_$pnl / target_risk_percent. The return Series of DataFrame will be populated on all the datetimes
    as the input Series or DataFrame, but will fill in the resulting DataFrame with a forward fill of the risk capital.
    If the data starts in the middle of the rebalance frequency, then the first date will be used for the initial
    stub period.

    :param downside_df: pd.Seires or pd.DataFrame of downside risk measures in PnL terms
    :param target_risk_percent: target percent risk that will determine the risk capital in $
    :param rebalance_freq: standard frequency of the rebalance, like "D" or "ME"
    :return: pd.Series or pd.DataFrame
    """

    capital = downside_df / target_risk_percent
    res = capital.resample(rebalance_freq).last()  # sample to the rebalance_freq
    res.loc[capital.index[0]] = capital.iloc[0]  # add the first item to fill in the values prior to first rebalance
    res = match_index(res, capital)
    return res


def total_pnl(x: list | NDArray | pd.DataFrame | pd.Series) -> float | pd.Series:
    """
    Given a time series of PnLs, calculate the cumulative PnL for the entire series

    :param x: any of a list, numpy array, pd.Series or pd. DataFrame
    :return: float if x is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """
    def _calc(x):
        return np.nansum(x)

    return dispatch_calc(x, _calc, name="total_pnl")


def r_squared(x: list | NDArray | pd.DataFrame | pd.Series) -> float | pd.Series:
    """
    The R-Squared of the cumulative values of x as the response variable and an even sequence the length of x as
    the explanatory. NaN values are dropped and considered to not have been included in the series such that the
    spacing of the explanatory variables is constant.

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :return: float if X is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        x = dropna(x)
        if len(x) < 1:
            return np.nan

        x = np.cumsum(x)
        rhat = stats.linregress(np.arange(len(x)), x).rvalue
        return rhat ** 2

    return dispatch_calc(x, _calc, name="r_squared")


def k_ratio(x: list | NDArray | pd.DataFrame | pd.Series) -> float | pd.Series:
    """
    The K-Ratio using the original 1996 formulation by Lars Kestner. The x series is a PnL per period.
    https://www.dropbox.com/scl/fi/myfpukl78c2lgcrc1irf4/K-Ratio-in-Excel.pdf?rlkey=ly2z9cnp1swqdr8tm3e4iwitw&st=q7ulvlyb&dl=0

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :return: float if X is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        x = dropna(x)
        x = np.insert(x, 0, 0)  # Add the anchor starting point at zero
        x = np.cumsum(x)
        res = stats.linregress(np.arange(len(x)), x)
        return res.slope / (res.stderr * sqrt(len(x)))

    return dispatch_calc(x, _calc, name="k_ratio")


def underwater_equity(pnl: list | NDArray | pd.DataFrame | pd.Series) -> pd.Series | pd.DataFrame:
    """
    For a given series of pnl will return the underwater equity of the input series.

    :param pnl: any of a list, numpy array, pd.Series or pd.DataFrame
    :return: pd.Series if pnl is a list, array or a pd.Series. A pd.DataFrame if pnl is a pd.DataFrame
    """
    if isinstance(pnl, pd.DataFrame):
        res = {}
        for column in pnl.columns:
            res[column] = underwater_equity(pnl[column])
        return pd.DataFrame(res)

    pnl = pd.Series(pnl)
    equity = pnl.cumsum()  # turn the PnL into equity curve
    equity = equity.ffill()  # fill forward the equity for NaN observations
    equity = equity.fillna(0)  # if the first instance is NaN this will make it zero

    equity = pd.concat([pd.Series(0, index=[prior_index(equity)]),
                        equity])  # prepend a zero as the first value to catch if the first pnl is negative
    high_water_mark = equity.cummax()
    high_water_mark = high_water_mark.iloc[1:]  # drop that prepended value
    equity = equity.iloc[1:]  # drop that prepended value
    drawdown = equity - high_water_mark

    return drawdown


def drawdown_details(pnl: list | NDArray | pd.DataFrame | pd.Series) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """
    Given an input series or dataframe of pnls, will return a dataframe, or dict of dataframes for an input dataframe,
    of the drawdowns with their start and end index, length and max drawdown amount. If the input has a DateTimeIndex
    the length of the drawdown returned is in business days, not accounting for holidays.

    :param pnl: any of a list, numpy array, pd.Series or pd.DataFrame
    :return: pd.DataFrame if pnl is a list, array or a pd.Series. A dict of pd.DataFrame if pnl is a pd.DataFrame
    """
    if isinstance(pnl, pd.DataFrame):
        res = {}
        for column in pnl.columns:
            res[column] = drawdown_details(pnl[column])
        return res

    drawdown = underwater_equity(pnl)
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
    :return: float if X is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        x = dropna(x)
        avg_annual = np.mean(x) * annual_periods
        max_drawdown = maximum_drawdown(x)
        return avg_annual / abs(max_drawdown)

    return dispatch_calc(x, _calc, name="calmar_ratio")
