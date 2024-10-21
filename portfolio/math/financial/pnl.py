"""
Financial math functions like Sharpe based on PnL time series
"""

from math import sqrt

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from portfolio.math.base import dispatch_calc, dropna
from portfolio.math.financial.common import (
    sharpe,
    sharpe_exp_weighted,
    sortino,
    sortino_exp_weighted,
    conditional_sortino,
    conditional_sortino_exp_weighted,
)
from portfolio.math.statistics import mean_exp_weighted


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


def pnl(x: list | NDArray | pd.DataFrame | pd.Series) -> float | pd.Series:
    """
    Given a time series of PnLs, calculate the cumulative PnL for the entire series

    :param x: any of a list, numpy array, pd.Series or pd. DataFrame
    :return: float if x is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        return np.nansum(x)

    return dispatch_calc(x, _calc, name="pnl")


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
        return rhat**2

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


def omega_ratio(x: list | NDArray | pd.DataFrame | pd.Series, threshold=0) -> float | pd.Series:
    """
    Calculate the Omega ratio for a series of pnl.
     See https://en.wikipedia.org/wiki/Omega_ratio for more details.

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :param threshold: the threshold value for dividing pnl into those above and below
    :return: float if X is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        x = dropna(x)
        # Calculate gains and losses relative to the threshold
        gains = x - threshold
        positive_gains = np.maximum(gains, 0)
        negative_gains = np.maximum(-gains, 0)

        # Calculate the integrals (sums in discrete case)
        integral_above = np.sum(positive_gains)
        integral_below = np.sum(negative_gains)

        # Avoid division by zero
        if integral_below == 0:
            return np.inf  # or a very large number if infinity isn't preferred

        # Calculate Omega ratio
        return integral_above / integral_below

    return dispatch_calc(x, _calc, name="omega_ratio")


def robustness(x: list | NDArray | pd.DataFrame | pd.Series) -> float | pd.Series:
    """
    Robustness of the pnl defined as the percentage of days, when ordered from the best to the worst, that can be
    eliminated and the total PnL would still be above zero. The higher the number, the more missed days the pnl
    can withstand before turning negative.

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :return: float if X is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        x = dropna(x)
        x_total = np.sum(x)
        if x_total <= 0.0:  # if Total PnL is negative the robustness is 0%
            return 0.0
        x_sorted = np.sort(x)[::-1]
        x_sorted_cum = x_sorted.cumsum()
        if x_sorted_cum.max() == x_total:  # If there are no down days, robustness is 100%
            return 1.0
        days_to_get_to_zero = np.argmax(x_sorted_cum >= x_total) + 1
        return days_to_get_to_zero / len(x)

    return dispatch_calc(x, _calc, name="robustness")


def underwater_equity(x: list | NDArray | pd.DataFrame | pd.Series) -> pd.Series | pd.DataFrame:
    """
    For a given input series will return the underwater equity of the input series.

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :return: pd.Series if X is a list, array or a pd.Series. A pd.DataFrame if x is a pd.DataFrame
    """
    if isinstance(x, pd.DataFrame):
        res = {}
        for column in x.columns:
            res[column] = underwater_equity(x[column])
        return pd.DataFrame(res)

    x = pd.Series(x)
    x = x.cumsum()  # turn the PnL into equity curve
    x = x.ffill()  # fill forward the equity for NaN observations
    x = x.fillna(0)  # if the first instance is NaN this will make it zero
    high_water_mark = x.cummax()
    drawdown = x - high_water_mark

    return drawdown


def drawdown_details(x: list | NDArray | pd.DataFrame | pd.Series) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """
    Given an input series or dataframe on pnls, will return a dataframe, or dict of dataframes for an input dataframe,
    of the drawdowns with their start and end index, length and max drawdown amount. If the input has a DateTimeIndex
    the length of the drawdown returned is in business days, not accounting for holidays.

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :return: pd.DataFrame if X is a list, array or a pd.Series. A dict of pd.DataFrame if x is a pd.DataFrame
    """
    if isinstance(x, pd.DataFrame):
        res = {}
        for column in x.columns:
            res[column] = drawdown_details(x[column])
        return res

    drawdown = underwater_equity(x)
    is_zero = drawdown == 0
    # find start dates (first day when dd is non-zero after a zero)
    start = ~is_zero & is_zero.shift(1)
    start = list(start[start == True].index)  # NOQA

    # find end dates (first day when dd is 0 after non-zero)
    end = is_zero & (~is_zero).shift(1)
    end = list(end[end == True].index)  # NOQA

    if len(start) == 0:  # start.empty
        return None

    # drawdown has no end (end period in dd)
    if len(end) == 0:  # end.empty
        end.append(drawdown.index[-1])

    # if the first drawdown start is larger than the first drawdown end it
    # means the drawdown series begins in a drawdown, and therefore we must add
    # the first index to the start series
    if start[0] > end[0]:
        start.insert(0, drawdown.index[0])

    # if the last start is greater than the end then we must add the last index
    # to the end series since the drawdown series must finish with a drawdown
    if start[-1] > end[-1]:
        end.append(drawdown.index[-1])

    result = pd.DataFrame(
        columns=("start", "end", "max_index", "length", "enter_length", "recovery_length", "drawdown"),
        index=range(0, len(start)),
    )

    for i in range(0, len(start)):
        # if the index of the Series is not a DateTimeIndex, and the start and end are the same, then need to set the
        # drawdown to the value on that single date. Pandas slicing works different for DateTimeIndex and not on how
        # inclusive it is of the end points
        if not isinstance(drawdown.index, pd.DatetimeIndex) and (start[i] == end[i]):
            dd = drawdown[end[i]]
        else:
            dd = drawdown[start[i] : end[i]].min()
        # find the index of the max drawdown, first instance
        if start[i] == end[i]:  # if the last drawdown is on the last day
            max_dd_index = start[i]
        else:
            max_dd_index = drawdown[start[i] : end[i]].idxmin()

        if isinstance(drawdown.index, pd.DatetimeIndex):
            if (i == len(start) - 1) and (
                drawdown[end[i]] != 0.0
            ):  # if this is the last drawdown and the series did not recover
                recovery_length = np.nan  # set the recovery length to pd.NaT
            else:
                recovery_length = np.busday_count(max_dd_index.date(), end[i].date())
            result.iloc[i] = (
                start[i],
                end[i],
                max_dd_index,
                np.busday_count(start[i].date(), end[i].date()),
                np.busday_count(start[i].date(), max_dd_index.date()),
                recovery_length,
                dd,
            )
        else:
            if (i == len(start) - 1) and (
                drawdown[end[i]] != 0.0
            ):  # if this is the last drawdown and the series did not recover
                recovery_length = np.nan  # set the recovery length to np.nan
            else:
                recovery_length = end[i] - max_dd_index
            result.iloc[i] = (
                start[i],
                end[i],
                max_dd_index,
                end[i] - start[i],
                max_dd_index - start[i],
                recovery_length,
                dd,
            )
    return result


def maximum_drawdown(x: list | NDArray | pd.DataFrame | pd.Series) -> float | pd.Series:
    """
    Maximum drawdown

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :return: float if X is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        details = drawdown_details(x)
        return details["drawdown"].min()

    return dispatch_calc(x, _calc, name="maximum_drawdown")


def average_drawdown(x: list | NDArray | pd.DataFrame | pd.Series) -> float | pd.Series:
    """
    Average drawdown

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :return: float if X is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        details = drawdown_details(x)
        return details["drawdown"].mean()

    return dispatch_calc(x, _calc, name="average_drawdown")


def average_drawdown_time(x: list | NDArray | pd.DataFrame | pd.Series) -> float | pd.Series:
    """
    Average drawdown time length in days

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :return: float if X is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        details = drawdown_details(x)
        return details["length"].mean()

    return dispatch_calc(x, _calc, name="average_drawdown_time", as_series=True)


def average_recovery_time(x: list | NDArray | pd.DataFrame | pd.Series) -> float | pd.Series:
    """
    Average recovery time of drawdowns

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :return: float if X is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        details = drawdown_details(x)
        return details["recovery_length"].mean()

    return dispatch_calc(x, _calc, name="average_recovery_time", as_series=True)


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

    def _calc(x):
        details = drawdown_details(x)
        lengths = details["length"]
        recovery_lengths = details["recovery_length"]
        recovery_lengths, lengths = dropna(recovery_lengths, lengths)  # filter out the last np.NaN if exists
        return -1 * (recovery_lengths / lengths).mean()

    return dispatch_calc(x, _calc, name="plunge_ratio")


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

    def _calc(x):
        details = drawdown_details(x)[["max_index", "length", "recovery_length"]]
        details = details.set_index("max_index")
        details = details.dropna()
        details["plunge"] = -1 * details["recovery_length"] / details["length"]
        plunges = details["plunge"].reindex(x.index)
        return mean_exp_weighted(plunges.values, window_length, half_life, annualize=False)

    return dispatch_calc(x, _calc, name="plunge_ratio_exponentially_weighted", as_series=True)


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
