"""
Transformation of time series
"""

import numpy as np
import pandas as pd
import scipy.stats
from numpy.typing import NDArray

from portfolio.utils import match_index, fillna


def pct_change_ln(prices: list | NDArray | pd.DataFrame | pd.Series) -> NDArray | pd.DataFrame | pd.Series:
    """
    Calculate the log-normal returns from a price series

    :param prices: either pandas DataFrame or Series, or a numpy array
    :return: same format as the prices parameter
    """
    if isinstance(prices, (pd.Series, pd.DataFrame)):
        # the default fillna in the pct_change method does not seem to work
        return np.log(prices.ffill().pct_change(fill_method=None) + 1)

    # for numpy
    return np.insert(np.diff(np.log(fillna(prices))), 0, np.nan)


def to_returns(prices: list | NDArray | pd.DataFrame | pd.Series) -> list | NDArray | pd.DataFrame | pd.Series:
    """
    Calculate the period returns from a price series. The length of the output is the same as the length of the input
    and the first value will be np.nan because there is no return on the first element.

    :param prices: either pandas DataFrame or Series, or a numpy array or list
    :return: same format as the prices parameter
    """
    if isinstance(prices, (pd.Series, pd.DataFrame)):
        # the default fillna in the pct_change method does not seem to work
        return prices.pct_change(fill_method=None)

    # for numpy and list
    return np.insert(np.diff(prices) / np.asarray(prices)[:-1], 0, np.nan)


def to_pnl(prices: list | NDArray | pd.DataFrame | pd.Series) -> list | NDArray | pd.DataFrame | pd.Series:
    """
    Calculate the pnl from a price series. The length of the output is the same as the length of the input
    and the first value will be np.nan because there is no return on the first element.

    :param prices: either pandas DataFrame or Series, or a numpy array or list
    :return: same format as the prices parameter
    """
    if isinstance(prices, (pd.Series, pd.DataFrame)):
        # the default fillna in the pct_change method does not seem to work
        return prices.diff()

    # for numpy and list
    return np.insert(np.diff(prices), 0, np.nan)


def price_index(
        returns: list | NDArray | pd.DataFrame | pd.Series, start_value: float | int = 100, start_index=None
) -> list | NDArray | pd.DataFrame | pd.Series:
    """
    Create a time series of price index from an array of period returns. The returns are compounded for each period.
    If the start_index is supplied then the start_value will be added at that index level. For numpy anything other
    than none in start_index will prepend the series with the start_value.
    NaNs in the input will be treated as zero values to calculate the index

    :param returns: either pandas DataFrame or Series, or a numpy array
    :param start_value: starting value of the series
    :param start_index: if provided then the index for the start_value in the returned list
    :return: same format as prices parameter
    """
    if isinstance(returns, pd.Series):
        # preserve what returns were nan to populate the result with nans
        nans = returns.isna()
        returns = returns.fillna(value=0)
        res = (1 + returns).expanding().apply(np.prod, raw=True) * start_value
        res.loc[nans] = np.nan
        if start_index is not None:
            assert start_index < res.index.min(), "provided start_index is not less first element of existing index"
            res.loc[start_index] = start_value
        return res.sort_index()

    if isinstance(returns, pd.DataFrame):
        res = {}
        for column in returns.columns:
            res[column] = price_index(returns[column], start_index=start_index, start_value=start_value)
        return pd.DataFrame(res)

    # for numpy and lists
    res = price_index(pd.Series(returns), start_value=start_value).values
    if start_index:
        res = np.insert(res, 0, start_value)
    if isinstance(returns, list):
        res = list(res)
    return res


def rebase(
        prices: list | NDArray | pd.DataFrame | pd.Series, start_value=100
) -> list | NDArray | pd.DataFrame | pd.Series:
    """
    Rebase a time series to a new number. This is the same as calculating returns and then cumulative TRI starting
    from the value.

    :param prices: either pandas DataFrame or Series, or a numpy array
    :param start_value: starting value of the series
    :return: same format as prices parameter
    """

    if isinstance(prices, (pd.Series, pd.DataFrame)):
        return prices / prices.iloc[0] * start_value

    # for numpy
    return prices / prices[0] * start_value


def winsorize(x: NDArray, limits: float = 0.01) -> NDArray:
    """
    Wrapper for the scipy winsorize function that removes NaNs from the evaluation. See scipy documentation here:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.mstats.winsorize.html

    :param x: numpy array
    :param limits: as defined in scipy documentation, the cutoff percentile
    :return: numpy array
    """
    if isinstance(x, pd.Series):
        if x.isna().all():  # if everything is NaN
            return x
        x[~np.isnan(x)] = scipy.stats.mstats.winsorize(x[~np.isnan(x)], limits=limits)  # noqa
    elif isinstance(x, np.ndarray):
        if np.isnan(x).all():
            return x
        x[~np.isnan(x)] = scipy.stats.mstats.winsorize(x[~np.isnan(x)], limits=limits)  # noqa
    else:
        raise ValueError("Object type for x not supported")
    return x


def pnl_to_returns(
    pnl: pd.Series | pd.DataFrame, capital: pd.Series | pd.DataFrame, dynamic_capital: str = None
) -> pd.Series | pd.DataFrame:
    """
    Given a pd.Series or pd.DataFrame of $PnL and capital, will return a matrix of %returns. The pnl and capital
    must have the same column names. The pnl and capital are end of period values, so the returns for a given
    day will be the pnl for today divided by the capital from the of the prior period. So if the frequency is daily
    then today's returns are today's pnl / yesterday's EOD capital.

    If pd.DataFrames are provided, then the columns names must be the same in the pnl and capital DataFrames

    If dynamic_capital is provided as a standard frequency string (ie: "D" or "ME") then the capital will be adjusted
    at that frequency based on the cumulative PnL over the life to date until that period end.

    :param pnl: pd.Series or pd.DataFrame
    :param capital: pd.Series or pd.DataFrame with columns matching pnl
    :param dynamic_capital: if a frequency string then add/subtract the pnl from the capital at the given frequency
    :return: pd.Series of pd.DataFrame
    """
    capital = capital.copy()
    if dynamic_capital:
        cumm_pnl = pnl.resample(dynamic_capital).sum()
        # If missing the first date, set to zero
        if capital.index[0] not in cumm_pnl.index:
            cumm_pnl.loc[capital.index[0]] = 0.0
        cumm_pnl = cumm_pnl.sort_index().cumsum()
        capital = capital + match_index(cumm_pnl, capital)
    capital_prior = match_index(capital.shift(1), pnl)
    returns = pnl / capital_prior
    return returns


def pnl_to_period(
        x: list | NDArray | pd.DataFrame | pd.Series,
        period: str,
        annual_periods: int = 252,
) -> float | pd.Series | pd.DataFrame:
    """
    Transform a PnL series to a given period, for example daily PnL to annual. If the x input has a DateTimeIndex then
    that will be used, otherwise the observations will be assumed to be at the annual_periods, for example a list
    of PnL with annual_periods = 252 would assume 252 observations in a year.

    :param x: any of a list, numpy array, pd.Series of pd.DataFrame
    :param period: target period to convert pnl to, for example "ME" for monthly, or "YE" for annual. Description of
        options: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    :param annual_periods: number of periods in the vector in a year for annualization, ie: 252 for a daily vector
    :return: same datatype and form as x
    """

    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.resample(period).sum()
    else:
        raise NotImplementedError("Datatypes other then pd.DataFrame and pd.Series not implemented")


def returns_to_period(
        x: list | NDArray | pd.DataFrame | pd.Series,
        period: str,
        annual_periods: int = 252,
) -> float | pd.Series | pd.DataFrame:
    """
    Transform a returns series to a given period, for example daily returns to annual. If the x input has a
    DateTimeIndex then that will be used, otherwise the observations will be assumed to be at the annual_periods,
    for example a list of returns with annual_periods = 252 would assume 252 observations in a year.

    :param x: any of a list, numpy array, pd.Series of pd.DataFrame
    :param period: target period to convert pnl to, for example "ME" for monthly, or "YE" for annual. Description of
        options: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    :param annual_periods: number of periods in the vector in a year for annualization, ie: 252 for a daily vector
    :return: same datatype and form as x
    """

    if isinstance(x, (pd.Series, pd.DataFrame)):
        return (1 + x).resample(period).prod() - 1
    else:
        raise NotImplementedError("Datatypes other then pd.DataFrame and pd.Series not implemented")


def returns_to_pnl(returns: pd.Series | pd.DataFrame, capital: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Given a pd.Series or pd.DataFrame of %returns and capital, will return a matrix of $PnL. The returns and capital
    must have the same column names. The returns and capital are end of period values, so the pnl for a given
    day will be the returns for today divided by the capital from the of the prior period. So if the frequency is daily
    then today's returns are today's returns / yesterday's EOD capital.

    If pd.DataFrames are provided, then the columns names must be the same in the returns and capital DataFrames

    :param returns: pd.Series or pd.DataFrame
    :param capital: pd.Series or pd.DataFrame with columns matching pnl
    :return: pd.Series of pd.DataFrame
    """
    capital = capital.copy()
    capital_prior = match_index(capital.shift(1), returns)
    pnl = returns * capital_prior
    return pnl


def volatility_match(x: list | NDArray | pd.DataFrame | pd.Series,
                     volatility_series: list | NDArray | pd.DataFrame | pd.Series) -> NDArray | pd.DataFrame | pd.Series:
    """
    Will return the x input with the values adjusted so that they match the volatility of the volatility_series input
    :param x: list, numpy array, pd.Series or pd.DataFrame
    :param volatility_series: list, numpy array or pd.Series
    :return: list, numpy array, pd.Series or pd.DataFrame
    """
    base_vol = np.nanstd(volatility_series)
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x * (base_vol / x.std(ddof=0))

    res = np.asarray(x) * (np.nanstd(x) / base_vol)
    if isinstance(x, list):
        return list(res)

    return res
