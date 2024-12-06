"""
Financial metrics reporting
"""

import pandas as pd

from portfolio.math import financial, statistics, transformation
import calendar

from pandas.io.formats.style import Styler

import portfolio.utils as styles
from portfolio.math import statistics
from portfolio.utils import column_types
from portfolio.utils import week_of_month
from portfolio.utils import average_of_cols, make_series


def all_metrics_pnl(pnl: pd.DataFrame, window_length=5 * 252, half_life=180) -> pd.DataFrame:
    """
    Returns a pd.DataFrame of all available metrics

    :param pnl: pd.DataFrame of pnl
    :param window_length: window for the exponential weighting. The window ignores any date index and assumes that
        each entry in the list/array/DataFrame/Series is equally spaced
    :param half_life: half life of the exponential decay
    :return: pd.DataFrame
    """
    metrics = [
        statistics.first_date(pnl),
        statistics.last_date(pnl),
        statistics.observations_count(pnl),
        financial.pnl.total_pnl(pnl),
        statistics.mean(pnl, name="average_daily_pnl"),
        statistics.mean_exp_weighted(pnl, window_length, half_life, name="average_daily_pnl_exponential_weighted"),
        statistics.stdev(pnl, annualize=False, name="daily_volatility"),
        statistics.stdev(pnl, annualize=True, name="annual_volatility"),
        statistics.stdev_exp_weighted(pnl, window_length, half_life, name="annual_volatility_exp_weighted"),
        statistics.downside_deviation(pnl, name="annual_downside_deviation"),
        statistics.downside_deviation_exp_weighted(pnl, window_length, half_life),
        financial.pnl.r_squared(pnl),
        financial.pnl.sharpe(pnl),
        financial.pnl.sharpe_exp_weighted(pnl, window_length, half_life),
        financial.pnl.sortino(pnl),
        financial.pnl.sortino_exp_weighted(pnl, window_length, half_life),
        financial.pnl.conditional_sortino(pnl),
        financial.pnl.conditional_sortino_exp_weighted(pnl, window_length, half_life),
        financial.pnl.k_ratio(pnl),
        financial.pnl.omega_ratio(pnl),
        financial.pnl.robustness(pnl),
        financial.pnl.average_drawdown(pnl),
        financial.pnl.maximum_drawdown(pnl),
        financial.pnl.average_drawdown_time(pnl),
        financial.pnl.average_recovery_time(pnl),
        financial.pnl.plunge_ratio(pnl),
        financial.pnl.plunge_ratio_exp_weighted(pnl, window_length, half_life),
        financial.pnl.calmar_ratio(pnl),
        statistics.ttest_1sample_1side(pnl, name="ttest_pvalue"),
    ]
    return pd.concat(metrics, axis=1).T


def all_metrics_returns(returns: pd.DataFrame, window_length=5 * 252, half_life=180) -> pd.DataFrame:
    """
    Returns a pd.DataFrame of all available metrics.

    :param returns: pd.DataFrame of returns
    :param window_length: window for the exponential weighting. The window ignores any date index and assumes that
        each entry in the list/array/DataFrame/Series is equally spaced
    :param half_life: half life of the exponential decay
    :return: pd.DataFrame
    """
    price_index = transformation.price_index(returns, 100, returns.index[0] - pd.offsets.BusinessDay(1))
    metrics = [
        statistics.first_date(returns),
        statistics.last_date(returns),
        statistics.observations_count(returns),
        financial.returns.total_return(returns),
        statistics.cagr(price_index),
        statistics.mean(returns, name="average_daily_return"),
        statistics.mean_exp_weighted(
            price_index, window_length, half_life, name="average_daily_pnl_exponential_weighted"
        ),
        statistics.stdev(returns, annualize=False, name="daily_volatility"),
        statistics.stdev(returns, annualize=True, name="annual_volatility"),
        statistics.stdev_exp_weighted(returns, window_length, half_life),
        statistics.downside_deviation(returns),
        statistics.downside_deviation_exp_weighted(returns, window_length, half_life),
        financial.returns.sharpe(returns),
        financial.returns.sharpe_exp_weighted(returns, window_length, half_life),
        financial.returns.sortino(returns),
        financial.returns.sortino_exp_weighted(returns, window_length, half_life),
        financial.returns.conditional_sortino(returns),
        financial.returns.conditional_sortino_exp_weighted(returns, window_length, half_life),
        financial.returns.k_ratio(returns),
        financial.returns.omega_ratio(returns),
        financial.returns.robustness(returns),
        financial.returns.average_drawdown(returns),
        financial.returns.maximum_drawdown(returns),
        financial.returns.average_drawdown_time(returns),
        financial.returns.average_recovery_time(returns),
        financial.returns.plunge_ratio(returns),
        financial.returns.plunge_ratio_exp_weighted(returns, window_length, half_life),
        financial.returns.calmar_ratio(returns),
        statistics.ttest_1sample_1side(returns, name="ttest_pvalue"),
    ]
    return pd.concat(metrics, axis=1).T


def month_year_table(ser: pd.Series | pd.DataFrame, fill_value=None, func=sum) -> pd.DataFrame:
    """
    Takes a series and returns a DataFrame in the standard performance reporting format with months for the columns
    years for the index, the values in the cell are the func applied of all values for that month-year.

    The YTD columns is the func applied to all the values in that year. The row at the bottom is the average of all
    years for a month or YTD.

    :param ser: pd.Series or one column pd.DataFrame with Datetime index
    :param fill_value: value for the cells with no data in the series
    :param func: function to apply to the data for aggregation
    :return: pd.DataFrame
    """
    ser = make_series(ser)
    # aggregate to yearly
    ytd = ser.resample("YE").agg(func)
    ytd.index = ytd.index.year
    # aggregate to monthly
    ser = ser.resample("ME").agg(func)
    # fill out the full month-year index of all possible months and expand the original series to fill it
    min_year = ser.index.year.min()
    max_year = ser.index.year.max()
    align_index = pd.DatetimeIndex(
        pd.date_range(
            pd.Timestamp(year=min_year, month=1, day=1), pd.Timestamp(year=max_year, month=12, day=31), freq="ME"
        )
    )
    ser, _ = ser.align(pd.Series(index=align_index), fill_value=fill_value)

    df = pd.DataFrame(ser)
    if len(df.columns) > 1:
        raise AttributeError(f"Only one column allowed if x is a DataFrame: {len(df.columns)}")
    df.columns = ["x"]
    df["year"] = df.index.year
    df["month"] = df.index.month

    # reshape to the correct format & add YTD column
    res = df.pivot(index="year", columns="month", values="x")
    res.index.name = None
    res.columns.name = None
    res["YTD"] = ytd
    # append the average row at the bottom
    res = average_of_cols(res)
    res = res.rename(columns=dict(enumerate(calendar.month_abbr)))
    return res


def year_func_table(ser: pd.Series | pd.DataFrame, funcs: dict = None) -> pd.DataFrame:
    """
    Takes a series and a dictionary of functions and returns a DataFrame years for the index, the columns as the
    name of the function from the dictionary key, and the values in the cell are the func applied of all values for
    that year. The LTD columns is the func applied to all the values.

    :param ser: pd.Series or one column pd.DataFrame with Datetime index
    :param funcs: dict of key as the display name, and value as the function
    :return: pd.DataFrame
    """
    ser = make_series(ser)

    df = ser.resample("YE").agg(funcs.values())
    df.columns = funcs.keys()
    df.index = df.index.year
    for name, func in funcs.items():
        df.at["LTD", name] = func(ser)
    return df.T


def metric_table(metrics: pd.DataFrame) -> Styler:
    """
    Takes in a metric pd.DataFrame of the type returned by the all_metrics_pnl() or all_metrics_returns() functions
    and returns a Styler dataframe with the cells formatted.

    In this version no rows are formatted as percent, all numbers are assumed to be decimal or integer.

    :param metrics: pd.DataFrame with the metric names in the index, time series in the column names, and metric values
        in the cells
    :return: pd.Styler
    """
    # x = functional.invert_iterable(subset)
    # df = metrics[x[0]].T
    # names = functional.replace_none(x[1], df.columns)
    # df.columns = names

    df = metrics.copy()
    row_types = column_types(df.T)
    df = styles.date_format(df, rows=row_types["date"])
    df = styles.datetime_format(df, rows=row_types["datetime"])
    df = styles.number_format(df, rows=row_types["large_number"], precision=0)
    df = styles.number_format(df, rows=row_types["small_number"] + row_types["percent"], precision=3)
    df = styles.number_format(df, rows=row_types["integer"], precision=0)
    return df


def best_worst_table(x: pd.DataFrame, period: str = None) -> pd.DataFrame:
    """
    Returns a pd.DataFrame of the best, avg, worst and percentage of positive over a given time aggregation. If the
    period is None then no aggregation is performed.

    :param x: pd.DataFrame
    :param period: target period to convert x to, for example "ME" for monthly, or "YE" for annual. Description of
        options: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    :return: pd.DataFrame
    """
    if period:
        x = x.resample(period).sum()
    metrics = [
        statistics.max(x, name="best"),
        statistics.mean(x[x > 0], "average up"),
        statistics.mean(x, "average"),
        statistics.mean(x[x < 0], "average down"),
        statistics.min(x, name="worst"),
        statistics.percentage_above(x, name="percent up"),
    ]
    return pd.concat(metrics, axis=1).T


def seasonal_table(x: pd.Series | pd.DataFrame, period: str = None) -> pd.DataFrame:
    """

    :param x:
    :param period:
    :return:
    """
    x = x.copy()
    if isinstance(x, pd.Series):
        x = pd.DataFrame(x)

    match period:
        case _ if period.upper() in ['MONTH', 'MONTHLY', "M", "ME"]:
            x['_group'] = x.index.month
            res = x.groupby('_group').mean()
            res = res.rename(index=dict(enumerate(calendar.month_abbr)))
            # res = res.sort_index()
        case _ if period.upper() in ['DAY', 'WEEKDAY', "D", "B"]:
            x['_group'] = x.index.dayofweek
            res = x.groupby('_group').mean()
            res = res.rename(index=dict(enumerate(calendar.day_abbr)))
            # res = res.sort_index()
        case _ if period.upper() in ['WEEKOFMONTH', "WOM"]:
            x['_group'] = x.index.to_series().apply(week_of_month)
            res = x.groupby('_group').mean()
            res = res.sort_index()
        case _:
            raise AttributeError(f"Period {period} not recognized")

    res.index.name = None
    return res
