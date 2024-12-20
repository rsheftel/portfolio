"""
Unit test for financial functions with PnL time series
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal, assert_frame_equal
from pytest import approx

from portfolio.math import base, financial
from portfolio.testing import mock_time_series

# GLOBAL VARIABLES

# Time series without nan
x = mock_time_series(100, 0.25, auto_regress=0.01, drift=0.05, seed=1234, round_decimals=2)
dates = pd.date_range("2024-01-01", periods=100, freq="1D")
time_series_01_series = pd.Series(x, index=dates)
time_series_01_df = pd.DataFrame({"px1": x, "px2": x * 2}, index=pd.date_range("2024-01-01", periods=100, freq="1D"))
time_series_01_list = [
    (list(x), "float"),
    (np.array(x), "float"),
    (time_series_01_series, "float"),
    (time_series_01_df, "pd.Series"),
]

# Time series with nans
x_nan = np.array(x.copy())
x_nan2 = x_nan.copy()
x_nan[[-10, -20, -30, -40]] = np.nan
x_nan2[[-5, -15, -25, -35]] = np.nan
time_series_nan_series = pd.Series(x_nan, index=dates)
time_series_nan_df = pd.DataFrame(
    {"px1": x_nan, "px2": x_nan2}, index=pd.date_range("2024-01-01", periods=100, freq="1D")
)
time_series_nan_list = [
    (list(x_nan), "float"),
    (np.array(x_nan), "float"),
    (time_series_nan_series, "float"),
    (time_series_nan_df, "pd.Series"),
]

# time series with trend and nan
time_series_trend_nan = mock_time_series(100, 0.05, auto_regress=0.01, drift=0.15, seed=1234, round_decimals=2)
time_series_trend_nan[[-5, -15, -25, -35]] = np.nan
time_series_trend_nan_pnl = np.diff(time_series_trend_nan)


# END GLOBAL VARIABLES


def test_risk_capital():
    # Series, daily freq
    ser = time_series_01_series.diff()[1:]
    risk = base.rolling(ser, np.std, window=20, min_periods=1)
    expected = pd.Series(
        [28.596228, 26.850219, 26.195138, 25.881337, 25.737076], index=pd.date_range("2024-04-05", periods=5)
    )
    actual = financial.pnl.risk_capital(risk, 0.10, rebalance_freq="D")
    assert_series_equal(actual[-5:], expected)

    # DataFrame, monthly freq
    df = time_series_01_df.diff()[1:]
    risk = base.rolling(df, np.std, window=20, min_periods=2).dropna()
    # strip out the actual month ends to check the dates on the return
    risk.loc["2024-01-31"] = np.nan
    risk.loc["2024-02-29"] = np.nan
    risk.loc["2024-03-31"] = np.nan
    risk = risk.dropna()

    px1 = [33.5] * 28
    px1.extend([51.847826] * 28)
    px1.extend([38.797118] * 30)
    px1.extend([57.450166] * 9)
    dates = pd.date_range("2024-01-03", "2024-04-09")
    dates = dates.drop(["2024-01-31", "2024-02-29", "2024-03-31"])
    expected = pd.DataFrame({"px1": px1}, index=dates)

    actual = financial.pnl.risk_capital(risk, 0.05, rebalance_freq="ME")
    assert_frame_equal(actual[["px1"]], expected)


@pytest.mark.parametrize("x, return_type", time_series_01_list)
def test_pnl(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return_type=float")
        assert financial.pnl.total_pnl(x) == 10253.92

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        actual = financial.pnl.total_pnl(x)
        expected = pd.Series({"px1": 10253.92, "px2": 20507.84}, name="total_pnl")
        assert_series_equal(actual, expected)


@pytest.mark.parametrize("x, return_type", time_series_01_list)
def test_sharpe(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return type = float")
        x = np.diff(x)  # make it a change series from levels
        assert financial.pnl.sharpe(x) == approx(0.214757)

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        x = x.diff(1)  # make it a change series from levels
        actual = financial.pnl.sharpe(x)
        expected = pd.Series({"px1": 0.214757, "px2": 0.214757}, name="sharpe")
        assert_series_equal(actual, expected)


@pytest.mark.parametrize("x, return_type", time_series_nan_list)
def test_sharpe_nan(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return type = float")
        x = np.diff(x)  # make it a change series from levels
        assert financial.pnl.sharpe(x) == approx(0.518755872)

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        x = x.diff(1)  # make it a change series from levels
        actual = financial.pnl.sharpe(x)
        expected = pd.Series({"px1": 0.518755872, "px2": 0.298978202}, name="sharpe")
        assert_series_equal(actual, expected)


@pytest.mark.parametrize("x, return_type", time_series_01_list)
def test_sharpe_exp_weighted(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return type = float")
        x = np.diff(x)  # make it a change series from levels
        assert financial.pnl.sharpe_exp_weighted(x, 50, 10) == approx(-0.514405)
        assert financial.pnl.sharpe_exp_weighted(x, 90, 20) == approx(-0.084543393)

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        x = x.diff(1)  # make it a change series from levels
        actual = financial.pnl.sharpe_exp_weighted(x, 90, 20)
        expected = pd.Series({"px1": -0.084543393, "px2": -0.084543393}, name="sharpe_exponential_weighted")
        assert_series_equal(actual, expected)


@pytest.mark.parametrize("x, return_type", time_series_nan_list)
def test_sharpe_exp_weighted_nan(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return type = float")
        x = np.diff(x)  # make it a change series from levels
        assert financial.pnl.sharpe_exp_weighted(x, 50, 10) == approx(-0.26723111)
        assert financial.pnl.sharpe_exp_weighted(x, 90, 20) == approx(0.27486970)

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        x = x.diff(1)  # make it a change series from levels
        actual = financial.pnl.sharpe_exp_weighted(x, 50, 15)
        expected = pd.Series({"px1": 0.05743569, "px2": -0.50406652}, name="sharpe_exponential_weighted")
        assert_series_equal(actual, expected)


@pytest.mark.parametrize("x, return_type", time_series_nan_list)
def test_sortino_nan(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return_type = float")
        x = np.diff(x)  # make it a change series from levels
        assert financial.pnl.sortino(x, annualize=False) == approx(0.055343276)

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        # as a list - window equal x length
        x = x.diff(1)  # make it a change series from levels
        actual = financial.pnl.sortino(x)
        expected = pd.Series({"px1": 0.878547274, "px2": 0.503027598}, name="sortino")
        assert_series_equal(actual, expected)


@pytest.mark.parametrize("x, return_type", time_series_nan_list)
def test_sortino_exp_weighted_nan(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return type = float")
        x = np.diff(x)  # make it a change series from levels
        assert financial.pnl.sortino_exp_weighted(x, 50, 10) == approx(-0.487259045)
        assert financial.pnl.sortino_exp_weighted(x, 90, 20) == approx(0.487349929)

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        x = x.diff(1)  # make it a change series from levels
        actual = financial.pnl.sortino_exp_weighted(x, 50, 15)
        expected = pd.Series({"px1": 0.104071185, "px2": -0.894089054}, name="sortino_exponential_weighted")
        assert_series_equal(actual, expected)


@pytest.mark.parametrize("x, return_type", time_series_01_list)
def test_conditional_sortino(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return_type = float")
        x = np.diff(x)  # make it a change series from levels
        assert financial.pnl.conditional_sortino(x, annualize=False) == approx(0.012932139)

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        # as a list - window equal x length
        x = x.diff(1)  # make it a change series from levels
        actual = financial.pnl.conditional_sortino(x)
        expected = pd.Series({"px1": 0.205291, "px2": 0.205291}, name="conditional_sortino")
        assert_series_equal(actual, expected)


@pytest.mark.parametrize("x, return_type", time_series_nan_list)
def test_conditional_sortino_nan(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return_type = float")
        x = np.diff(x)  # make it a change series from levels
        assert financial.pnl.conditional_sortino(x, annualize=False) == approx(0.031612902)

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        # as a list - window equal x length
        x = x.diff(1)  # make it a change series from levels
        actual = financial.pnl.conditional_sortino(x)
        expected = pd.Series({"px1": 0.501839, "px2": 0.286071}, name="conditional_sortino")
        assert_series_equal(actual, expected)


@pytest.mark.parametrize("x, return_type", time_series_nan_list)
def test_conditional_sortino_exp_weighted_nan(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return type = float")
        x = np.diff(x)  # make it a change series from levels
        assert financial.pnl.conditional_sortino_exp_weighted(x, 50, 10) == approx(-0.298499034)
        assert financial.pnl.conditional_sortino_exp_weighted(x, 90, 20) == approx(0.290072348)

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        x = x.diff(1)  # make it a change series from levels
        actual = financial.pnl.conditional_sortino_exp_weighted(x, 50, 15)
        expected = pd.Series(
            {"px1": 0.063263295, "px2": -0.5391126485}, name="conditional_sortino_exponential_weighted"
        )
        assert_series_equal(actual, expected)


@pytest.mark.parametrize("x, return_type", time_series_nan_list)
def test_r_square_nan(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return_type = float")
        x = np.diff(x)  # make it a change series from levels
        assert financial.pnl.r_squared(x) == approx(0.69287427)

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        x = x.diff(1)  # make it a change series from levels
        actual = financial.pnl.r_squared(x)
        expected = pd.Series({"px1": 0.69287427, "px2": 0.54591477}, name="r_squared")
        assert_series_equal(actual, expected)


@pytest.mark.parametrize("x, return_type", time_series_nan_list)
def test_k_ratio_nan(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return_type = float")
        x = np.diff(x)  # make it a change series from levels
        assert financial.pnl.k_ratio(x) == approx(1.49362289)

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        x = x.diff(1)  # make it a change series from levels
        actual = financial.pnl.k_ratio(x)
        expected = pd.Series({"px1": 1.49362289, "px2": 1.09688972}, name="k_ratio")
        assert_series_equal(actual, expected)


@pytest.mark.parametrize("x, return_type", time_series_nan_list)
def test_omega_ratio_nan(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return_type = float")
        x = np.diff(x)  # make it a change series from levels
        assert financial.pnl.omega_ratio(x, threshold=1) == approx(0.36467907)

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        x = x.diff(1)  # make it a change series from levels
        actual = financial.pnl.omega_ratio(x)
        expected = pd.Series({"px1": 1.08967954, "px2": 1.05054107}, name="omega_ratio")
        assert_series_equal(actual, expected)


def test_robustness():
    x = time_series_trend_nan_pnl
    assert financial.pnl.robustness(x) == approx(0.1978022)

    # edge case of no down days
    x = np.arange(20)
    assert financial.pnl.robustness(x) == 1.0

    # negative total PnL
    x = np.array([1.0, -2.0, 1.0, -2.0, 1.0, -2.0])
    assert financial.pnl.robustness(x) == 0.0


def test_underwater_equity():
    expected_start = pd.Series([0, -0.37, 0, -0.39, -0.38, 0, 0, -0.32, 0, -0.56, 0, 0, 0, -0.78, -0.11])
    expected_end = pd.Series(
        [
            -0.20,
            -0.20,
            0.0,
            0.0,
            -0.07,
            0.0,
            -0.26,
            -0.27,
            -0.31,
            -0.03,
            -0.03,
            -0.03,
            0.0,
            0.0,
            -0.36,
        ],
        pd.RangeIndex(start=84, stop=99, step=1),
    )

    # as a numpy array
    x = time_series_trend_nan_pnl
    assert_series_equal(financial.pnl.underwater_equity(x)[:15], expected_start)
    assert_series_equal(financial.pnl.underwater_equity(x)[-15:], expected_end)

    # as a series
    x = pd.Series(x)
    assert_series_equal(financial.pnl.underwater_equity(x)[:15], expected_start)
    assert_series_equal(financial.pnl.underwater_equity(x)[-15:], expected_end)

    # as a dataframe
    df = pd.DataFrame({"px1": x, "px2": x})
    expected_start.name = "px1"
    expected_end.name = "px1"
    assert_series_equal(financial.pnl.underwater_equity(df).iloc[:15, 0], expected_start)
    assert_series_equal(financial.pnl.underwater_equity(df).iloc[-15:, 0], expected_end)
    expected_start.name = "px2"
    expected_end.name = "px2"
    assert_series_equal(financial.pnl.underwater_equity(df).iloc[:15, 1], expected_start)
    assert_series_equal(financial.pnl.underwater_equity(df).iloc[-15:, 1], expected_end)

    # nan as first element
    x = [np.nan, 1.2, -0.2, -0.3, 0.4, 2.0]
    actual = financial.pnl.underwater_equity(x)
    expected = pd.Series([0.0, 0.0, -0.2, -0.5, -0.1, 0])
    assert_series_equal(actual, expected)

    # negative first number
    x = [-1, 2, 3, -6, -4, -5]
    actual = financial.pnl.underwater_equity(x)
    expected = pd.Series([-1, 0, 0, -6, -10, -15])
    assert_series_equal(actual, expected)

    x = [-1, -2, 3, -6, -4, -5]
    actual = financial.pnl.underwater_equity(x)
    expected = pd.Series([-1, -3, 0, -6, -10, -15])
    assert_series_equal(actual, expected)


def test_drawdown_details():
    x = time_series_trend_nan_pnl
    expected = pd.DataFrame(
        {
            "start": [90, 98],
            "end": [96, 98],
            "max_index": [92, 98],
            "length": [6, 0],
            "enter_length": [2, 0],
            "recovery_length": [4, np.nan],
            "drawdown": [-0.31, -0.36],
        },
        index=pd.RangeIndex(23, 25),
    )
    actual = financial.pnl.drawdown_details(x)
    assert_frame_equal(actual[-2:], expected, check_dtype=False)

    # with dates on series, confirm length is business days
    x = pd.Series(data=time_series_trend_nan_pnl, index=pd.bdate_range("2024-04-01", periods=99))
    expected = pd.DataFrame(
        {
            "start": [pd.Timestamp(year=2024, month=8, day=d) for d in [5, 15]],
            "end": [pd.Timestamp(year=2024, month=8, day=d) for d in [13, 15]],
            "max_index": [pd.Timestamp(year=2024, month=8, day=d) for d in [7, 15]],
            "length": [6, 0],
            "enter_length": [2, 0],
            "recovery_length": [4, np.nan],
            "drawdown": [-0.31, -0.36],
        },
        index=pd.RangeIndex(23, 25),
    )
    actual = financial.pnl.drawdown_details(x)
    assert_frame_equal(actual[-2:], expected, check_dtype=False)

    # as a dataframe
    df = pd.DataFrame({"px1": x_nan, "px2": x_nan2}, index=pd.bdate_range("2024-03-29", periods=100))
    df = df.diff()
    expected1 = pd.DataFrame(
        {
            "start": [
                pd.Timestamp("2024-05-27"),
                pd.Timestamp("2024-06-27"),
                pd.Timestamp("2024-07-01"),
                pd.Timestamp("2024-07-03"),
                pd.Timestamp("2024-07-16"),
            ],
            "end": [
                pd.Timestamp("2024-06-26"),
                pd.Timestamp("2024-06-28"),
                pd.Timestamp("2024-07-02"),
                pd.Timestamp("2024-07-15"),
                pd.Timestamp("2024-08-15"),
            ],
            "max_index": [
                pd.Timestamp("2024-05-30"),
                pd.Timestamp("2024-06-27"),
                pd.Timestamp("2024-07-01"),
                pd.Timestamp("2024-07-03"),
                pd.Timestamp("2024-07-23"),
            ],
            "length": [22, 1, 1, 8, 22],
            "enter_length": [3, 0, 0, 0, 5],
            "recovery_length": [19, 1, 1, 8, np.nan],
            "drawdown": [-6.85, -0.75, -5.33, -4.88, -8.30],
        },
        index=pd.RangeIndex(5, 10),
    )
    expected2 = pd.DataFrame(
        {
            "start": [
                pd.Timestamp("2024-04-26"),
                pd.Timestamp("2024-05-06"),
                pd.Timestamp("2024-05-17"),
                pd.Timestamp("2024-05-27"),
                pd.Timestamp("2024-07-03"),
            ],
            "end": [
                pd.Timestamp("2024-05-03"),
                pd.Timestamp("2024-05-16"),
                pd.Timestamp("2024-05-24"),
                pd.Timestamp("2024-07-02"),
                pd.Timestamp("2024-08-15"),
            ],
            "max_index": [
                pd.Timestamp("2024-04-26"),
                pd.Timestamp("2024-05-08"),
                pd.Timestamp("2024-05-22"),
                pd.Timestamp("2024-05-30"),
                pd.Timestamp("2024-07-23"),
            ],
            "length": [5, 8, 5, 26, 31],
            "enter_length": [0, 2, 3, 3, 14],
            "recovery_length": [5, 6, 2, 23, np.nan],
            "drawdown": [-4.46, -4.78, -1.97, -6.85, -10.42],
        },
        index=pd.RangeIndex(2, 7),
    )
    actual = financial.pnl.drawdown_details(df)
    assert_frame_equal(actual["px1"][-5:], expected1, check_dtype=False)
    assert_frame_equal(actual["px2"][-5:], expected2, check_dtype=False)


def test_drawdown_details_empty():
    # no drawdown data
    x = pd.Series([1, 2, 3, 4, 5])
    expected = pd.DataFrame(
        columns=("start", "end", "max_index", "length", "enter_length", "recovery_length", "drawdown"),
    )
    actual = financial.pnl.drawdown_details(x)
    assert_frame_equal(actual, expected)

    assert financial.pnl.average_drawdown(x) is np.nan
    assert financial.pnl.maximum_drawdown(x) is np.nan
    assert financial.pnl.average_drawdown_time(x) is np.nan
    assert financial.pnl.average_recovery_time(x) is np.nan


def test_average_drawdown():
    x = pd.DataFrame({"px1": x_nan, "px2": x_nan2}, index=pd.bdate_range("2024-03-29", periods=100))
    x = x.diff()

    expected = pd.Series({"px1": -4.53, "px2": -5.20857142}, name="average_drawdown")
    actual = financial.pnl.average_drawdown(x)
    assert_series_equal(actual, expected)


def test_max_drawdown():
    x = pd.DataFrame({"px1": x_nan, "px2": x_nan2}, index=pd.bdate_range("2024-03-29", periods=100))
    x = x.diff()

    expected = pd.Series({"px1": -8.30, "px2": -10.42}, name="maximum_drawdown")
    actual = financial.pnl.maximum_drawdown(x)
    assert_series_equal(actual, expected)


def test_avg_drawdown_time():
    x = pd.DataFrame({"px1": x_nan, "px2": x_nan2}, index=pd.bdate_range("2024-03-29", periods=100))
    x = x.diff()

    expected = pd.Series({"px1": 8.8, "px2": 13.0}, name="average_drawdown_time")
    actual = financial.pnl.average_drawdown_time(x)
    assert_series_equal(actual, expected)


def test_avg_recovery_time():
    x = pd.DataFrame({"px1": x_nan, "px2": x_nan2}, index=pd.bdate_range("2024-03-29", periods=100))
    x = x.diff()

    expected = pd.Series({"px1": 5.777778, "px2": 7.6666667}, name="average_recovery_time")
    actual = financial.pnl.average_recovery_time(x)
    assert_series_equal(actual, expected)


def test_plunge_ratio():
    x = pd.DataFrame({"px1": x_nan, "px2": x_nan2}, index=pd.bdate_range("2024-03-29", periods=100))
    x = x.diff()

    expected = pd.Series({"px1": -0.845960, "px2": -0.772436}, name="plunge_ratio")
    actual = financial.pnl.plunge_ratio(x)
    assert_series_equal(actual, expected)


def test_plunge_ratio_exp_weighted():
    x = pd.DataFrame({"px1": x_nan, "px2": x_nan2}, index=pd.bdate_range("2024-03-29", periods=100))
    x = x.diff()

    expected = pd.Series({"px1": -0.854753, "px2": -0.766261}, name="plunge_ratio_exponential_weighted")
    actual = financial.pnl.plunge_ratio_exp_weighted(x, 512, 120)
    assert_series_equal(actual, expected)


@pytest.mark.parametrize("x, return_type", time_series_nan_list)
def test_calmar_ratio_nan(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return_type = float")
        x = np.diff(x)  # make it a change series from levels
        assert financial.pnl.calmar_ratio(x) == approx(2.4556070)

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        x = x.diff(1)  # make it a change series from levels
        actual = financial.pnl.calmar_ratio(x)
        expected = pd.Series({"px1": 2.4556070, "px2": 1.092278163}, name="calmar_ratio")
        assert_series_equal(actual, expected)


def test_factor_correlation():
    x_1 = mock_time_series(500, 0.1, drift=0.1, seed=123)
    x_2 = mock_time_series(500, 0.15, drift=0.2, auto_regress=0.1, seed=456)
    levels = pd.DataFrame({"x1": x_1, "x2": x_2}, pd.bdate_range("2024-01-01", periods=500))
    pnl = levels.diff()

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
    actual = financial.pnl.correlation(pnl, factor)
    assert_frame_equal(actual, expected)

    # 5 day change on change
    names = ["x1", "x2", "factor1", "factor2"]
    expected = pd.DataFrame(
        [
            [1, 0.065119, 0.866212, 0.065887],
            [0.065119, 1, 0.065462, 1],
            [0.866212, 0.065462, 1, 0.063972],
            [0.065887, 1, 0.063972, 1],
        ],
        columns=names,
        index=names,
    )
    actual = financial.pnl.correlation(pnl, factor, periods=5)
    assert_frame_equal(actual, expected)


def test_correlation_pvalues():
    x_1 = mock_time_series(500, 0.1, drift=0.1, seed=123)
    x_2 = mock_time_series(500, 0.15, drift=0.2, auto_regress=0.1, seed=456)
    levels = pd.DataFrame({"x1": x_1, "x2": x_2}, pd.bdate_range("2024-01-01", periods=500))
    pnl = levels.diff()

    x_1 = mock_time_series(500, 0.05, drift=0.15, auto_regress=0.5, seed=123)
    x_2 = mock_time_series(500, 0.15, drift=0.02, auto_regress=0.1, seed=456)
    factor = pd.DataFrame({"factor1": x_1, "factor2": x_2}, pd.bdate_range("2024-01-01", periods=500))

    # 5 day change on change
    names = ["x1", "x2", "factor1", "factor2"]
    expected = pd.DataFrame(
        [
            [0, 0.147984, 0, 0.143659],
            [0.147984, 0, 0.146271, 0],
            [0, 0.146271, 0, 0.155278],
            [0.143659, 0, 0.155278, 0],
        ],
        columns=names,
        index=names,
    )
    actual = financial.pnl.correlation_pvalues(pnl, factor, periods=5)
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

    actual = financial.pnl.add_factors_to_drawdown_details(drawdowns, factors)

    expected = pd.DataFrame(
        [
            [
                0.000000,
                0.000000,
                1.123367,
                0.319755,
            ],
            [
                0.483822,
                0.331674,
                0.762334,
                -1.597426,
            ],
            [
                0.600429,
                -0.222345,
                0.223687,
                0.000000,
            ],
            [
                0.000000,
                0.000000,
                -0.471285,
                -0.839049,
            ],
            [
                1.564399,
                -1.115518,
                1.615720,
                0.144655,
            ],
            [
                -0.922802,
                -0.803806,
                -0.261760,
                -0.190993,
            ],
            [
                2.008487,
                0.182250,
                2.517570,
                0.903297,
            ],
            [
                1.609416,
                0.384067,
                1.336126,
                0.298627,
            ],
        ],
        index=list(range(4, 12)),
    )
    columns = pd.MultiIndex.from_tuples([('factor_change_start_to_max', 'fac1'), ('factor_change_start_to_max', 'fac2'),
                                         ('factor_change_max_to_end', 'fac1'), ('factor_change_max_to_end', 'fac2')],
                                        names=['factor_change', 0])
    expected.columns = columns

    assert_frame_equal(actual.loc[list(range(4, 12)), columns], expected)
