"""
Unit tests for financial functions on Returns time series
"""

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
from pytest import approx

from portfolio.math import financial
from portfolio.math.testing import mock_time_series

time_series_equity: dict = {}
time_series_returns: dict = {}


def setup_module():
    global time_series_equity, time_series_returns
    prices = mock_time_series(101, 0.10, auto_regress=0.01, drift=0.02, seed=5678, round_decimals=2)
    time_series_equity = {"numpy": prices, "series": pd.Series(prices, index=pd.bdate_range("2024-04-01", periods=101))}

    time_series_returns = {
        "numpy": np.diff(time_series_equity["numpy"]) / time_series_equity["numpy"][:-1],
        "series": time_series_equity["series"].pct_change().iloc[1:],
    }
    rets_nan = time_series_returns["series"].copy()
    rets_nan.iloc[[-5, -15, -25, -35]] = np.nan
    time_series_returns["dataframe"] = pd.DataFrame({"px": time_series_returns["series"], "nans": rets_nan})


def test_total_return():
    assert financial.returns.total_return(time_series_returns["series"]) == approx(0.0268)
    assert financial.returns.total_return(time_series_returns["dataframe"]["nans"]) == approx(0.039309236)


def test_sharpe():
    assert financial.returns.sharpe(time_series_returns["series"]) == approx(0.557332321)
    actual = financial.returns.sharpe(time_series_returns["dataframe"])
    expected = pd.Series({"px": 0.557332321, "nans": 0.800113387}, name="sharpe")
    assert_series_equal(actual, expected)


def test_sharpe_exponential_weighted():
    assert financial.returns.sharpe_exp_weighted(time_series_returns["series"], 50, 15, annualize=False) == approx(
        0.074725546
    )
    actual = financial.returns.sharpe_exp_weighted(time_series_returns["dataframe"], 100, 75)
    expected = pd.Series({"px": 0.63899256, "nans": 0.95164053}, name="sharpe_exponential_weighted")
    assert_series_equal(actual, expected)


def test_sortino():
    assert financial.returns.sortino(time_series_returns["series"]) == approx(1.024980961)
    actual = financial.returns.sortino(time_series_returns["dataframe"])
    expected = pd.Series({"px": 1.024980961, "nans": 1.469608032}, name="sortino")
    assert_series_equal(actual, expected)


def test_sortino_exponential_weighted():
    assert financial.returns.sortino_exp_weighted(time_series_returns["series"], 50, 50, annualize=False) == approx(
        0.106283132
    )
    actual = financial.returns.sortino_exp_weighted(time_series_returns["dataframe"], 50, 25)
    expected = pd.Series({"px": 1.919548824, "nans": 3.0444}, name="sortino_exponential_weighted")
    assert_series_equal(actual, expected)


def test_conditional_sortino():
    assert financial.returns.conditional_sortino(time_series_returns["series"]) == approx(0.60491944)
    actual = financial.returns.conditional_sortino(time_series_returns["dataframe"])
    expected = pd.Series({"px": 0.60491944, "nans": 0.854541665}, name="conditional_sortino")
    assert_series_equal(actual, expected)


def test_conditional_sortino_exp_weighted():
    assert financial.returns.conditional_sortino_exp_weighted(time_series_returns["series"], 80, 40) == approx(
        0.852769333)
    actual = financial.returns.conditional_sortino_exp_weighted(time_series_returns["dataframe"], 50, 10)
    expected = pd.Series({"px": 1.768665718, "nans": 2.351750037}, name="conditional_sortino_exponential_weighted")
    assert_series_equal(actual, expected)
