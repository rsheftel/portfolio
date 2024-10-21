"""
Unit test for the transformation module
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from pandas import to_datetime
from pandas.testing import assert_frame_equal, assert_series_equal

import portfolio.math.base
import portfolio.math.financial as financial
import portfolio.math.transformation as transformation
from portfolio.math.testing import mock_time_series

# global variables
data = {}
index = pd.DatetimeIndex([])
time_series_equity = {}
time_series_pnl = {}
time_series_returns: dict
bdates: pd.DatetimeIndex


def setup_module():
    global data, index, time_series_equity, time_series_pnl, time_series_returns, bdates
    data_numpy = {
        "basic": np.array([100.0, 101, 99, 102, 100, 103, 104, 102]),
        "many_nan": np.array([np.nan, 101.0, 99, np.nan, 100, 103, 104, np.nan]),
        "one_nan": np.array([100.0, 101, 99, np.nan, 100, 103, 104, 102]),
        "all_nan": np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
    }
    index = pd.date_range("2010-01-01", periods=8)
    data = {
        "numpy": data_numpy,
        "series": {k: pd.Series(data_numpy[k], index=index) for k in data_numpy},
        "dataframe": {k: pd.DataFrame({"data": data_numpy[k]}, index=index) for k in data_numpy},
    }
    bdates = pd.bdate_range("2024-03-29", periods=100)
    time_series_1 = mock_time_series(100, 0.05, auto_regress=0.01, drift=0.15, seed=1234, round_decimals=2)
    time_series_1[[-5, -15, -25, -35]] = np.nan
    time_series_2 = mock_time_series(100, 0.25, auto_regress=0.01, drift=0.05, seed=1234, round_decimals=2)
    time_series_3 = time_series_2.copy()
    time_series_2[[-10, -20, -30, -40]] = np.nan
    time_series_3[[-5, -15, -25, -35]] = np.nan
    time_series_equity = {
        "numpy": time_series_1,
        "series": pd.Series(time_series_1, index=bdates),
        "dataframe": pd.DataFrame({"px1": time_series_2, "px2": time_series_3}, index=bdates),
    }
    time_series_pnl = {
        "numpy": np.diff(time_series_equity["numpy"]),
        "series": time_series_equity["series"].diff(),
        "dataframe": time_series_equity["dataframe"].diff(),
    }
    time_series_returns = {
        "numpy": np.diff(time_series_equity["numpy"]) / time_series_equity["numpy"][:-1],
        "series": time_series_equity["series"].pct_change(fill_method=None).iloc[1:],  # drop the first row
        "dataframe": time_series_equity["dataframe"].pct_change(fill_method=None).iloc[1:, :],  # drop the first row
    }


def assert_equality(method, actual, expected_values):
    if method == "numpy":
        assert_almost_equal(actual, expected_values, 5)
    elif method == "series":
        assert_series_equal(actual, pd.Series(expected_values, index=index))
    elif method == "dataframe":
        assert_frame_equal(actual, pd.DataFrame({"data": expected_values}, index=index))
    else:
        assert False


@pytest.mark.parametrize("method", ["numpy", "series", "dataframe"])
@pytest.mark.parametrize(
    "key, expected",
    [
        ("basic", [np.nan, 0.01, -0.01980198, 0.03030303, -0.019607843, 0.03, 0.009708738, -0.019230769]),
        ("many_nan", [np.nan, np.nan, -0.01980198, np.nan, np.nan, 0.03, 0.009708738, np.nan]),
    ],
)
def test_to_returns(method, key, expected):
    returns = transformation.to_returns(data[method][key])
    assert_equality(method, returns, expected)


def test_price_index():
    expected = [100.30, 99.93, 100.90, 100.51, 100.52, 101.18, 101.32, 101.00, 101.35, 100.79]
    actual = transformation.price_index(time_series_returns["numpy"][:10], start_value=100)
    assert_almost_equal(actual, expected, 5)

    expected = [100, 100.30, 99.93, 100.90, 100.51, 100.52, 101.18, 101.32, 101.00, 101.35, 100.79]
    actual = transformation.price_index(time_series_returns["numpy"][:10], start_value=100, start_index=True)
    assert_almost_equal(actual, expected, 5)

    expected = pd.Series(
        [100.30, 99.93, 100.90, 100.51, 100.52, 101.18, 101.32, 101.00, 101.35, 100.79], index=bdates[1:11]
    )
    actual = transformation.price_index(time_series_returns["series"][:10], start_value=100)
    assert_series_equal(actual, expected)

    expected = pd.Series(
        [100, 100.30, 99.93, 100.90, 100.51, 100.52, 101.18, 101.32, 101.00, 101.35, 100.79], index=bdates[:11]
    )
    expected[pd.Timestamp("2024-03-29")] = 100
    actual = transformation.price_index(time_series_returns["series"][:10], start_value=100,
                                        start_index=pd.Timestamp("2024-03-29"))
    assert_series_equal(actual, expected, check_freq=False)

    expected = pd.DataFrame(
        {
            "px1": [100.79, 98.23, 102.39, 99.73, 99.11, 101.69, 101.72, 99.41, 100.46, 96.97],
            "px2": [100.79, 98.23, 102.39, 99.73, 99.11, 101.69, 101.72, 99.41, 100.46, 96.97],
        },
        index=bdates[1:11],
    )
    actual = transformation.price_index(time_series_returns["dataframe"][:10], start_value=100)
    assert_frame_equal(actual, expected, check_freq=False)

    expected = pd.DataFrame(
        {
            "px1": [100, 100.79, 98.23, 102.39, 99.73, 99.11, 101.69, 101.72, 99.41, 100.46, 96.97],
            "px2": [100, 100.79, 98.23, 102.39, 99.73, 99.11, 101.69, 101.72, 99.41, 100.46, 96.97],
        },
        index=bdates[0:11],
    )
    actual = transformation.price_index(
        time_series_returns["dataframe"][:10], start_value=100, start_index=pd.Timestamp("2024-03-29")
    )
    assert_frame_equal(actual, expected, check_freq=False)


@pytest.mark.parametrize("method", ["numpy", "series", "dataframe"])
@pytest.mark.parametrize(
    "key, expected",
    [
        ("basic", [50, 50.5, 49.5, 51, 50, 51.5, 52, 51]),
        ("many_nan", [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
        ("one_nan", [50, 50.5, 49.5, np.nan, 50, 51.5, 52, 51]),
    ],
)
def test_rebase(method, key, expected):
    new_prices = transformation.rebase(data[method][key], start_value=50)
    assert_equality(method, new_prices, expected)


@pytest.mark.parametrize("method", ["numpy", "series"])
@pytest.mark.parametrize(
    "key, expected",
    [
        ("basic", [100.0, 101, 100, 102, 100, 102, 102, 102]),
        ("many_nan", [np.nan, 101.0, 100, np.nan, 100, 103, 103, np.nan]),
        ("one_nan", [100.0, 101, 100, np.nan, 100, 103, 103, 102]),
        ("all_nan", [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
    ],
)
def test_winsorize(method, key, expected):
    new_prices = transformation.winsorize(data[method][key], limits=0.25)
    assert_equality(method, new_prices, expected)


def test_pnl_to_returns():
    x = mock_time_series(100, 0.25, auto_regress=0.01, drift=0.05, seed=1234, round_decimals=2)
    dates = pd.date_range("2024-01-01", periods=100, freq="1D")
    ser = pd.Series(x, index=dates).diff().dropna()
    df = pd.DataFrame({"pnl1": x, "pnl2": x * 2}, index=dates).diff().dropna()

    risk_ser = portfolio.math.base.rolling(ser, np.std, window=20, min_periods=2)
    capital_ser = financial.pnl.risk_capital(risk_ser[1:], 0.10, rebalance_freq="D")

    expected = pd.Series(
        [np.nan, np.nan, 0.248358, -0.096959, -0.022079, 0.102328, 0.001190],
        index=pd.date_range("2024-01-02", periods=7),
    )
    actual = transformation.pnl_to_returns(ser, capital_ser)
    assert_series_equal(actual[:7], expected, atol=1e-6, rtol=1e-6)

    # dates in pnl missing in capital
    risk = portfolio.math.base.rolling(df, np.std, window=20, min_periods=2).dropna()
    # strip out the actual month ends to check the dates on the return
    risk.loc["2024-01-31"] = np.nan
    risk.loc["2024-02-29"] = np.nan
    risk.loc["2024-03-31"] = np.nan
    risk = risk.dropna()
    capital_df = financial.pnl.risk_capital(risk, 0.05, rebalance_freq="ME")

    expected = pd.DataFrame(
        {"pnl1": [-0.051550, -0.012707, -0.016362, 0.012707, 0.039513, -0.023499, -0.006440, 0.024717, -0.042820]},
        index=pd.date_range("2024-04-01", periods=9),
    )
    expected["pnl2"] = expected["pnl1"]
    actual = transformation.pnl_to_returns(df, capital_df)
    assert_frame_equal(actual[-9:], expected, atol=1e-6, rtol=1e-6)

    # dates in capital missing in pnl
    df_missing = df.copy().drop(["2024-04-02", "2024-04-08"])
    expected = pd.DataFrame(
        {"pnl1": [-0.051550, -0.012707, -0.016362, 0.012707, 0.039513, -0.023499, -0.006440, 0.024717, -0.042820]},
        index=pd.date_range("2024-04-01", periods=9),
    )
    expected = expected.drop(["2024-04-02", "2024-04-08"])
    expected["pnl2"] = expected["pnl1"]
    actual = transformation.pnl_to_returns(df_missing, capital_df)
    assert_frame_equal(actual[-7:], expected, atol=1e-6, rtol=1e-6)


def test_pnl_to_returns_dynamic_capital():
    x = mock_time_series(100, 0.25, auto_regress=0.01, drift=0.05, seed=1234, round_decimals=2)
    dates = pd.date_range("2024-01-01", periods=100, freq="1D")
    df = pd.DataFrame({"pnl1": x, "pnl2": x * 2}, index=dates).diff().dropna()
    # dates in pnl missing in capital
    risk = portfolio.math.base.rolling(df, np.std, window=20, min_periods=2).dropna()
    # strip out the actual month ends to check the dates on the return
    risk.loc["2024-01-31"] = np.nan
    risk.loc["2024-02-29"] = np.nan
    risk.loc["2024-03-31"] = np.nan
    risk = risk.dropna()
    capital_df = financial.pnl.risk_capital(risk, 0.05, rebalance_freq="ME")

    expected = pd.DataFrame(
        {
            "pnl1": [
                -0.046907485,
                -0.011379550,
                -0.014653119,
                0.011379550,
                0.035385723,
                -0.021044373,
                -0.005767717,
                0.022135562,
                -0.038347523,
            ]
        },
        index=pd.date_range("2024-04-01", periods=9),
    )
    expected["pnl2"] = expected["pnl1"]

    actual = transformation.pnl_to_returns(df, capital_df, dynamic_capital="ME")
    assert_frame_equal(actual[-9:], expected, atol=1e-6, rtol=1e-6)

    # dynamic capital, first date in the pnl is month end
    df2 = df["2024-01-31":]
    expected = pd.DataFrame(
        {"pnl1": [0.0605970, -0.0662687, 0.0220870, 0.0219014, 0.0009280]},
        index=pd.date_range("2024-01-31", periods=5),
    )
    expected["pnl2"] = expected["pnl1"]

    actual = transformation.pnl_to_returns(df2, capital_df, dynamic_capital="ME")
    assert_frame_equal(actual[:5], expected, atol=1e-6, rtol=1e-6)


def test_pnl_to_period():
    # series YE
    expected = pd.Series([13.55], [pd.Timestamp("2024-12-31")])
    actual = transformation.pnl_to_period(time_series_pnl["series"], "YE")
    assert_series_equal(actual, expected, check_freq=False)

    # series ME
    expected = pd.Series(
        [0, 3.09, 3.73, 2.72, 3.48, 0.53],
        pd.to_datetime(["2024-03-31", "2024-04-30", "2024-05-31", "2024-06-30", "2024-07-31", "2024-08-31"]),
    )
    actual = transformation.pnl_to_period(time_series_pnl["series"], "ME")
    assert_series_equal(actual, expected, check_freq=False)

    # dataframe YE
    expected = pd.DataFrame({"px1": [7.36], "px2": [4.11]}, [pd.Timestamp("2024-12-31")])
    actual = transformation.pnl_to_period(time_series_pnl["dataframe"], "YE")
    assert_frame_equal(actual, expected, check_freq=False)

    # dataframe ME
    expected = pd.DataFrame(
        {
            "px1": [0, 0.06, 2.55, 4.56, 2.69, -2.5],
            "px2": [0, 0.06, 2.55, 0.27, 4.84, -3.61],
        },
        pd.to_datetime(["2024-03-31", "2024-04-30", "2024-05-31", "2024-06-30", "2024-07-31", "2024-08-31"]),
    )
    actual = transformation.pnl_to_period(time_series_pnl["dataframe"], "ME")
    assert_frame_equal(actual, expected, check_freq=False)


def test_returns_to_period():
    actual = transformation.returns_to_period(time_series_returns["dataframe"], "ME")
    expected = pd.Series(
        [0.0006, 0.0254847, 0.045175211, 0.02617929, -0.0236183],
        index=to_datetime(["4/30/24", "5/31/24", "6/30/24", "7/31/24", "8/31/24"]),
    )
    expected.name = "px1"
    assert_series_equal(actual["px1"], expected, check_freq=False)
