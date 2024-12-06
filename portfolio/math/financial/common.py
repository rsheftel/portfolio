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


def omega_ratio(x: list | NDArray | pd.DataFrame | pd.Series, threshold=0) -> float | pd.Series:
    """
    Omega ratio
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


def drawdown_details(drawdown: pd.DataFrame | pd.Series) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """
    Given an input series or dataframe of drawdowns, will return a dataframe, or dict of dataframes for an input
    dataframe, of the drawdowns with their start and end index, length and max drawdown amount.
    If the input has a DateTimeIndex the length of the drawdown returned is in business days, not accounting for
    holidays.

    :param drawdown: pd.Series or pd.DataFrame
    :return: pd.DataFrame if drawdown is a pd.Series. A dict of pd.DataFrame if drawdown is a pd.DataFrame
    """
    if isinstance(drawdown, pd.DataFrame):
        res = {}
        for column in drawdown.columns:
            res[column] = drawdown_details(drawdown[column])
        return res

    is_zero = drawdown == 0
    # find start dates (first day when dd is non-zero after a zero)
    start = ~is_zero & is_zero.shift(1)
    start = list(start[start == True].index)  # NOQA

    # find end dates (first day when dd is 0 after non-zero)
    end = is_zero & (~is_zero).shift(1)
    end = list(end[end == True].index)  # NOQA

    if len(start) == 0:  # start.empty
        return pd.DataFrame(
            columns=("start", "end", "max_index", "length", "enter_length", "recovery_length", "drawdown"),
        )

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
            dd = drawdown[start[i]: end[i]].min()
        # find the index of the max drawdown, first instance
        if start[i] == end[i]:  # if the last drawdown is on the last day
            max_dd_index = start[i]
        else:
            max_dd_index = drawdown[start[i]: end[i]].idxmin()

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


def average_drawdown(x: list | NDArray | pd.DataFrame | pd.Series, drawdown_details) -> float | pd.Series:
    """
    Average drawdown

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :param drawdown_details: function to calculate drawdown details
    :return: float if X is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        details = drawdown_details(x)
        if len(details) == 0:
            return np.nan
        return details["drawdown"].mean()

    return dispatch_calc(x, _calc, name="average_drawdown")


def maximum_drawdown(x: list | NDArray | pd.DataFrame | pd.Series, drawdown_details) -> float | pd.Series:
    """
    Maximum drawdown

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :param drawdown_details: function to calculate drawdown details
    :return: float if X is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        details = drawdown_details(x)
        if len(details) == 0:
            return np.nan
        return details["drawdown"].min()

    return dispatch_calc(x, _calc, name="maximum_drawdown")


def average_drawdown_time(x: list | NDArray | pd.DataFrame | pd.Series, drawdown_details) -> float | pd.Series:
    """
    Average drawdown time length in days

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :param drawdown_details: function to calculate drawdown details
    :return: float if X is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        details = drawdown_details(x)
        if len(details) == 0:
            return np.nan
        return details["length"].mean()

    return dispatch_calc(x, _calc, name="average_drawdown_time", as_series=True)


def average_recovery_time(x: list | NDArray | pd.DataFrame | pd.Series, drawdown_details) -> float | pd.Series:
    """
    Average recovery time of drawdowns

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :param drawdown_details: function to calculate drawdown details
    :return: float if X is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        details = drawdown_details(x)
        if len(details) == 0:
            return np.nan
        return details["recovery_length"].mean()

    return dispatch_calc(x, _calc, name="average_recovery_time", as_series=True)


def plunge_ratio(x: list | NDArray | pd.DataFrame | pd.Series, drawdown_details) -> float | pd.Series:
    """
    Plunge ratio is the average percentage of time during a drawdown that is spent in recovery. If the number is -1.0
    that means that the maximum drawdown point was on the first day and the entire rest of the drawdown is spent in
    recovery. If the number is 0.0 that means that the drawdown was gradual reaching the maximum on the last day prior
    to recovering all the drawdown on the last day. The more negative the number, the sharper the losses, the closer
    to 0.0 the more grinding the losses.

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :param drawdown_details: function to calculate drawdown details
    :return: float if X is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        details = drawdown_details(x)
        if len(details) == 0:
            return np.nan
        lengths = details["length"]
        recovery_lengths = details["recovery_length"]
        recovery_lengths, lengths = dropna(recovery_lengths, lengths)  # filter out the last np.NaN if exists
        return -1 * (recovery_lengths / lengths).mean()

    return dispatch_calc(x, _calc, name="plunge_ratio")


def plunge_ratio_exp_weighted(
        x: list | NDArray | pd.DataFrame | pd.Series, window_length: int, half_life: int, drawdown_details
) -> float | pd.Series:
    """
    Exponentially weighted Plunge Ratio.

    :param x: any of a list, numpy array, pd.Series or pd.DataFrame
    :param window_length: window for the exponential weighting. The window ignores any date index and assumes that
        each entry in the list/array/DataFrame/Series is equally spaced
    :param half_life: half life of the exponential decay
    :param drawdown_details: function to calculate drawdown details
    :return: float if X is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        details = drawdown_details(x)[["max_index", "length", "recovery_length"]]
        details = details.set_index("max_index")
        details = details.dropna()
        if len(details) == 0:
            return np.nan
        details["plunge"] = -1 * details["recovery_length"] / details["length"]
        plunges = details["plunge"].reindex(x.index)
        return mean_exp_weighted(plunges.values, window_length, half_life, annualize=False)

    return dispatch_calc(x, _calc, name="plunge_ratio_exponential_weighted", as_series=True)
