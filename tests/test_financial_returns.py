"""
Unit tests for financial functions on Returns time series
"""

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal
from pytest import approx

from portfolio.math import financial
from portfolio.math.transformation import price_index, to_returns
from portfolio.testing import mock_time_series

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
        0.852769333
    )
    actual = financial.returns.conditional_sortino_exp_weighted(time_series_returns["dataframe"], 50, 10)
    expected = pd.Series({"px": 1.768665718, "nans": 2.351750037}, name="conditional_sortino_exponential_weighted")
    assert_series_equal(actual, expected)


def test_r_squared():
    assert financial.returns.r_squared(time_series_returns["numpy"]) == approx(0.53572506)
    assert financial.returns.r_squared(time_series_returns["series"]) == approx(0.53572506)
    assert_series_equal(
        financial.returns.r_squared(time_series_returns["dataframe"]),
        pd.Series({"px": 0.53572506, "nans": 0.69283529}, name="r_squared"),
    )


def test_k_ratio():
    assert financial.returns.k_ratio(time_series_returns["numpy"]) == approx(1.06350680)
    assert financial.returns.k_ratio(time_series_returns["series"]) == approx(1.06350680)
    assert_series_equal(
        financial.returns.k_ratio(time_series_returns["dataframe"]),
        pd.Series({"px": 1.06350680, "nans": 1.48629589}, name="k_ratio"),
    )


def test_omega_ratio():
    assert financial.returns.omega_ratio(time_series_returns["numpy"]) == approx(1.091964937)
    assert financial.returns.omega_ratio(time_series_returns["series"]) == approx(1.091964937)
    assert_series_equal(
        financial.returns.omega_ratio(time_series_returns["dataframe"]),
        pd.Series({"px": 1.091964937, "nans": 1.133791535}, name="omega_ratio"),
    )


def test_robustness():
    assert financial.returns.robustness(time_series_returns["numpy"]) == approx(0.02)
    assert financial.returns.robustness(time_series_returns["series"]) == approx(0.02)
    assert_series_equal(
        financial.returns.robustness(time_series_returns["dataframe"]),
        pd.Series({"px": 0.02, "nans": 0.02083333}, name="robustness"),
    )


def test_drawdown_series():
    x = pd.Series(time_series_returns["numpy"], index=range(1, len(time_series_returns["numpy"]) + 1))
    x = price_index(x, start_index=0)
    expected = pd.Series([0, -0.0043, 0, 0, -0.016058249, -0.01037303, 0], index=range(7))
    actual = financial.returns.drawdown_series(x)
    assert_series_equal(actual[:7], expected)

    x = time_series_returns["series"]
    x = price_index(x, start_index=pd.Timestamp("2024-04-01"))
    expected = pd.Series(
        [-0.016922778, -0.005154639, -0.005446411, -0.010698308, -0.001556117, -0.006516242, -0.001361603],
        index=pd.bdate_range(start="2024-08-09", periods=7),
    )
    actual = financial.returns.drawdown_series(x)
    assert_series_equal(actual[-7:], expected, check_freq=False)

    x = time_series_returns["dataframe"]
    x = price_index(x, start_index=pd.Timestamp("2024-04-01"))
    expected = pd.DataFrame(
        {
            "px": [-0.016922778, -0.005154639, -0.005446411, -0.010698308, -0.001556117, -0.006516242, -0.001361603],
            "nans": [-0.011771712, 0, 0, -0.005280657, 0, -0.004967855, 0],
        },
        index=pd.bdate_range(start="2024-08-09", periods=7),
    )
    actual = financial.returns.drawdown_series(x)
    assert_frame_equal(actual[-7:], expected, check_freq=False)


def test_drawdown_details():
    # raw numpy confirm the start and end are index
    expected = pd.DataFrame(
        {
            "start": [1.0, 4, 7, 42, 48, 53, 62, 66, 69],
            "end": [2.0, 6, 41, 47, 52, 61, 65, 68, 100],
            "drawdown": [
                -0.004300000,
                -0.016058249,
                -0.020820999,
                -0.017559436,
                -0.018031333,
                -0.013452474,
                -0.018600098,
                -0.010565447,
                -0.018770667,
            ],
        }
    )
    actual = financial.returns.drawdown_details(time_series_returns["numpy"])
    assert_frame_equal(actual[["start", "end", "drawdown"]], expected, check_dtype=False)

    # with dates on series, confirm length is business days
    expected = pd.DataFrame(
        {
            "start": [
                pd.Timestamp(d)
                for d in [
                    "2024-04-02",
                    "2024-04-05",
                    "2024-04-10",
                    "2024-05-29",
                    "2024-06-06",
                    "2024-06-13",
                    "2024-06-26",
                    "2024-07-02",
                    "2024-07-05",
                ]
            ],
            "end": [
                pd.Timestamp(d)
                for d in [
                    "2024-04-03",
                    "2024-04-09",
                    "2024-05-28",
                    "2024-06-05",
                    "2024-06-12",
                    "2024-06-25",
                    "2024-07-01",
                    "2024-07-04",
                    "2024-08-19",
                ]
            ],
            "max_index": [
                pd.Timestamp(d)
                for d in [
                    "2024-04-02",
                    "2024-04-05",
                    "2024-05-17",
                    "2024-05-31",
                    "2024-06-11",
                    "2024-06-21",
                    "2024-06-26",
                    "2024-07-03",
                    "2024-07-30",
                ]
            ],
            "length": [
                1.00,
                2.00,
                34.00,
                5.00,
                4.00,
                8.00,
                3.00,
                2.00,
                31.00,
            ],
            "enter_length": [
                0.0,
                0.0,
                27.0,
                2.0,
                3.0,
                6.0,
                0.0,
                1.0,
                17.0,
            ],
            "recovery_length": [1.0, 2.0, 7.0, 3.0, 1.0, 2.0, 3.0, 1.0, np.nan],
            "drawdown": [
                -0.004300000,
                -0.016058249,
                -0.020820999,
                -0.017559436,
                -0.018031333,
                -0.013452474,
                -0.018600098,
                -0.010565447,
                -0.018770667,
            ],
        },
    )
    actual = financial.returns.drawdown_details(time_series_returns["series"])
    assert_frame_equal(actual, expected, check_dtype=False)

    # DataFrame
    expected = pd.DataFrame(
        {
            "start": [
                pd.Timestamp(d)
                for d in [
                    "2024-04-02",
                    "2024-04-05",
                    "2024-04-10",
                    "2024-05-29",
                    "2024-06-06",
                    "2024-06-13",
                    "2024-06-26",
                    "2024-07-03",
                    "2024-07-05",
                    "2024-07-29",
                    "2024-08-14",
                    "2024-08-16",
                ]
            ],
            "end": [
                pd.Timestamp(d)
                for d in [
                    "2024-04-03",
                    "2024-04-09",
                    "2024-05-28",
                    "2024-06-05",
                    "2024-06-12",
                    "2024-06-25",
                    "2024-07-01",
                    "2024-07-04",
                    "2024-07-25",
                    "2024-08-12",
                    "2024-08-15",
                    "2024-08-19",
                ]
            ],
            "max_index": [
                pd.Timestamp(d)
                for d in [
                    "2024-04-02",
                    "2024-04-05",
                    "2024-05-17",
                    "2024-05-31",
                    "2024-06-11",
                    "2024-06-21",
                    "2024-06-26",
                    "2024-07-03",
                    "2024-07-17",
                    "2024-07-29",
                    "2024-08-14",
                    "2024-08-16",
                ]
            ],
            "length": [1.00, 2.00, 34.00, 5.00, 4.00, 8.00, 3.00, 1.00, 14.00, 10, 1, 1],
            "enter_length": [0.0, 0.0, 27.0, 2.0, 3.0, 6.0, 0.0, 0, 8.0, 0, 0, 0],
            "recovery_length": [1.0, 2.0, 7.0, 3.0, 1.0, 2.0, 3.0, 1.0, 6, 10, 1, 1],
            "drawdown": [
                -0.004300000,
                -0.016058249,
                -0.020820999,
                -0.017559436,
                -0.018031333,
                -0.013452474,
                -0.018600098,
                -0.006092767,
                -0.015185526,
                -0.013629283,
                -0.005280657,
                -0.004967855,
            ],
        },
    )
    actual = financial.returns.drawdown_details(time_series_returns["dataframe"])
    assert_frame_equal(actual['nans'], expected, check_dtype=False)


def test_drawdown_details_empty():
    # no drawdown data
    x = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    expected = pd.DataFrame(
        columns=("start", "end", "max_index", "length", "enter_length", "recovery_length", "drawdown"),
    )
    actual = financial.returns.drawdown_details(x)
    assert_frame_equal(actual, expected)

    assert financial.returns.average_drawdown(x) is np.nan
    assert financial.returns.maximum_drawdown(x) is np.nan
    assert financial.returns.average_drawdown_time(x) is np.nan
    assert financial.returns.average_recovery_time(x) is np.nan


def test_average_drawdown():
    expected = pd.Series({"px": -0.01535097, "nans": -0.01283156}, name="average_drawdown")
    actual = financial.returns.average_drawdown(time_series_returns['dataframe'])
    assert_series_equal(actual, expected)


def test_maximum_drawdown():
    expected = pd.Series({"px": -0.020820999, "nans": -0.020820999}, name="maximum_drawdown")
    actual = financial.returns.maximum_drawdown(time_series_returns['dataframe'])
    assert_series_equal(actual, expected)


def test_average_drawdown_time():
    expected = pd.Series({"px": 10.0, "nans": 7.0}, name="average_drawdown_time")
    actual = financial.returns.average_drawdown_time(time_series_returns['dataframe'])
    assert_series_equal(actual, expected)


def test_average_recovery_time():
    expected = pd.Series({"px": 2.5, "nans": 3.166667}, name="average_recovery_time")
    actual = financial.returns.average_recovery_time(time_series_returns['dataframe'])
    assert_series_equal(actual, expected)


def test_plunge_ratio():
    expected = pd.Series({"px": -0.600735, "nans": -0.727871}, name="plunge_ratio")
    actual = financial.returns.plunge_ratio(time_series_returns['dataframe'])
    assert_series_equal(actual, expected)


def test_plunge_ratio_exp_weighted():
    expected = pd.Series({"px": -0.502269030, "nans": -0.801793373}, name="plunge_ratio_exponential_weighted")
    actual = financial.returns.plunge_ratio_exp_weighted(time_series_returns['dataframe'], 75, 25)
    assert_series_equal(actual, expected)


def test_calmar_ratio():
    expected = pd.Series({"px": 3.430906491, "nans": 5.082554087}, name="calmar_ratio")
    actual = financial.returns.calmar_ratio(time_series_returns['dataframe'])
    assert_series_equal(actual, expected)


def test_factor_correlation():
    x_1 = mock_time_series(500, 0.1, drift=0.1, seed=123)
    x_2 = mock_time_series(500, 0.15, drift=0.2, auto_regress=0.1, seed=456)
    levels = pd.DataFrame({"x1": x_1, "x2": x_2}, pd.bdate_range("2024-01-01", periods=500))
    returns = to_returns(levels)

    x_1 = mock_time_series(500, 0.05, drift=0.15, auto_regress=0.5, seed=123)
    x_2 = mock_time_series(500, 0.15, drift=0.02, auto_regress=0.1, seed=456)
    factor = pd.DataFrame({"factor1": x_1, "factor2": x_2}, pd.bdate_range("2024-01-01", periods=500))

    # levels correlation
    names = ["x1", "x2", "factor1", "factor2"]
    expected = pd.DataFrame(
        [
            [1, 0.998550, 0.999541, 0.946753],
            [0.998550, 1, 0.999317, 0.957257],
            [0.999541, 0.999317, 1, 0.947426],
            [0.946753, 0.957257, 0.947426, 1],
        ],
        columns=names,
        index=names,
    )
    actual = financial.returns.correlation(returns, factor)
    assert_frame_equal(actual, expected)

    # 5 day change on change
    names = ["x1", "x2", "factor1", "factor2"]
    expected = pd.DataFrame(
        [
            [1, 0.090731849, 0.857191888, 0.073698409],
            [0.090731849, 1, 0.107814111, 0.975452303],
            [0.857191888, 0.107814111, 1, 0.069018057],
            [0.073698409, 0.975452303, 0.069018057, 1],
        ],
        columns=names,
        index=names,
    )
    actual = financial.returns.correlation(returns, factor, periods=5)
    assert_frame_equal(actual, expected)


def test_correlation_pvalues():
    x_1 = mock_time_series(500, 0.1, drift=0.1, seed=123)
    x_2 = mock_time_series(500, 0.15, drift=0.2, auto_regress=0.1, seed=456)
    levels = pd.DataFrame({"x1": x_1, "x2": x_2}, pd.bdate_range("2024-01-01", periods=500))
    returns = to_returns(levels)

    x_1 = mock_time_series(500, 0.05, drift=0.15, auto_regress=0.5, seed=123)
    x_2 = mock_time_series(500, 0.15, drift=0.02, auto_regress=0.1, seed=456)
    factor = pd.DataFrame({"factor1": x_1, "factor2": x_2}, pd.bdate_range("2024-01-01", periods=500))

    # 5 day change on change
    names = ["x1", "x2", "factor1", "factor2"]
    expected = pd.DataFrame(
        [
            [0, 0.043620851, 0, 0.101817641],
            [0.043620851, 0, 0.016520498, 0],
            [0, 0.016520498, 0, 0.125152288],
            [0.101817641, 0, 0.125152288, 0],
        ],
        columns=names,
        index=names,
    )
    actual = financial.returns.correlation_pvalues(returns, factor, periods=5)
    assert_frame_equal(actual, expected)


def test_add_factors_to_drawdown_details():
    x = pd.Series(
        mock_time_series(100, 0.20, auto_regress=0.1, drift=0.10, seed=405),
        index=pd.bdate_range("2024-01-01", periods=100),
    )
    drawdowns = financial.pnl.drawdown_details(x.diff())

    factors = pd.DataFrame(
        {
            "fac1": mock_time_series(50, 0.20, auto_regress=0.1, drift=0.10, seed=505),
            "fac2": mock_time_series(50, 0.10, auto_regress=-0.1, drift=-0.10, seed=505, nans=20),
        },
        index=pd.bdate_range("2024-02-01", periods=50),
    )

    actual = financial.returns.add_factors_to_drawdown_details(drawdowns, factors)

    expected = pd.DataFrame(
        [
            [-0.010632, -0.007516, 0.007620, -0.016044],
            [0.017721, 0.000309, 0.002180, 0.000000],
            [-0.000885, -0.002264, -0.004588, -0.008562],
            [0.004002, -0.011383, 0.015737, 0.001493],
            [-0.000570, -0.006133, -0.002512, -0.001981],
            [0.005938, 0.001894, 0.024073, 0.009368],
            [0.001508, -0.002436, 0.012726, 0.003124],
        ],
        index=list(range(5, 12)),
    )
    columns = pd.MultiIndex.from_tuples(
        [
            ("factor_change_start_to_max", "fac1"),
            ("factor_change_start_to_max", "fac2"),
            ("factor_change_max_to_end", "fac1"),
            ("factor_change_max_to_end", "fac2"),
        ],
        names=["factor_change", 0],
    )
    expected.columns = columns

    assert_frame_equal(actual.loc[list(range(5, 12)), columns], expected, atol=0.000001)
