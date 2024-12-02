"""
Unit tests for grids
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from portfolio import report as grids
from portfolio.math import transformation
from testing import mock_time_series


def test_month_year_table():
    ser = pd.Series(list(range(100)), index=pd.date_range("2023-11-15", periods=100))
    expected = pd.DataFrame(
        {
            "Jan": [0.0, 1922.0, 1922.0 / 2],
            "Feb": [0.0, 1947, 1947 / 2],
            "Mar": [0.0, 0, 0],
            "Apr": [0.0, 0, 0],
            "May": [0.0, 0, 0],
            "Jun": [0.0, 0, 0],
            "Jul": [0.0, 0, 0],
            "Aug": [0.0, 0, 0],
            "Sep": [0.0, 0, 0],
            "Oct": [0.0, 0, 0],
            "Nov": [120.0, 0, 120 / 2],
            "Dec": [961.0, 0, 961 / 2],
            "YTD": [120.0 + 961, 1922 + 1947, (120 + 961 + 1922 + 1947) / 2],
        },
        index=[2023, 2024, "average"],
    )

    actual = grids.month_year_table(ser, fill_value=0)
    assert_frame_equal(actual, expected)

    # one column dataframe works
    df = pd.DataFrame(ser)
    df.columns = ["px"]
    actual = grids.month_year_table(df, fill_value=0)
    assert_frame_equal(actual, expected)

    # more than one column dataframe fails
    bad_df = df.copy()
    bad_df["bad"] = bad_df["px"]
    with pytest.raises(AttributeError):
        grids.month_year_table(bad_df)

    # na as fill value
    expected = pd.DataFrame(
        {
            "Jan": [np.nan, 1922.0, 1922.0],
            "Feb": [np.nan, 1947, 1947],
            "Mar": [np.nan, np.nan, np.nan],
            "Apr": [np.nan, np.nan, np.nan],
            "May": [np.nan, np.nan, np.nan],
            "Jun": [np.nan, np.nan, np.nan],
            "Jul": [np.nan, np.nan, np.nan],
            "Aug": [np.nan, np.nan, np.nan],
            "Sep": [np.nan, np.nan, np.nan],
            "Oct": [np.nan, np.nan, np.nan],
            "Nov": [120.0, np.nan, 120],
            "Dec": [961.0, np.nan, 961],
            "YTD": [120.0 + 961, 1922 + 1947, (120 + 961 + 1922 + 1947) / 2],
        },
        index=[2023, 2024, "average"],
    )

    actual = grids.month_year_table(ser)
    assert_frame_equal(actual, expected)

    # use average as the func
    expected = pd.DataFrame(
        {
            "Jan": [0.0, 62, 62.0 / 2],
            "Feb": [0.0, 88.5, 88.5 / 2],
            "Mar": [0.0, 0, 0],
            "Apr": [0.0, 0, 0],
            "May": [0.0, 0, 0],
            "Jun": [0.0, 0, 0],
            "Jul": [0.0, 0, 0],
            "Aug": [0.0, 0, 0],
            "Sep": [0.0, 0, 0],
            "Oct": [0.0, 0, 0],
            "Nov": [7.5, 0, 7.5 / 2],
            "Dec": [31.0, 0, 31.0 / 2],
            "YTD": [23, 73, (23 + 73) / 2],
        },
        index=[2023, 2024, "average"],
    )

    ser = pd.Series(list(range(100)), index=pd.date_range("2023-11-15", periods=100))
    actual = grids.month_year_table(ser, fill_value=0, func=np.mean)
    assert_frame_equal(actual, expected)


def test_year_func_table():
    ser = pd.Series(list(range(100)), index=pd.date_range("2023-11-15", periods=100))

    expected = pd.DataFrame(
        {2023: [1081, 23, 13.71130920], 2024: [3869, 73, 15.44344521], "LTD": [4950, 49.5, 28.86607005]},
        index=["total", "average", "st_dev"],
    )

    actual = grids.year_func_table(ser, {"total": np.sum, "average": np.average, "st_dev": np.std})
    assert_frame_equal(actual, expected)


def test_best_worst_table():
    x1 = mock_time_series(100, 0.25, auto_regress=0.01, drift=0.05, seed=4321, nans=5, round_decimals=2)
    x2 = mock_time_series(100, 0.25, auto_regress=0.01, drift=0.05, seed=1234, nans=5, round_decimals=2)
    df = pd.DataFrame({"px1": x1, "px2": x2}, index=pd.bdate_range("2023-11-15", periods=100))
    df = transformation.to_pnl(df)

    # no period, no aggregation
    expected = pd.DataFrame(
        {
            "px1": [5.96, 1.789333333, -0.006853933, -1.843863636, -5.58, 0.505617978],
            "px2": [7.67, 1.826458333, 0.026777778, -2.03, -6.28, 0.533333333],
        },
        index=[
            "best",
            "average up",
            "average",
            "average down",
            "worst",
            "percent up",
        ],
    )
    actual = grids.best_worst_table(df)
    assert_frame_equal(actual, expected)

    # Aggregating a daily series to D or B will fill NaNs with zeros and change the results
    expected = pd.DataFrame(
        {
            "px1": [5.96, 1.789333333, -0.0061, -1.843863636, -5.58, 0.45],
            "px2": [7.67, 1.826458333, 0.0241, -2.03, -6.28, 0.48],
        },
        index=[
            "best",
            "average up",
            "average",
            "average down",
            "worst",
            "percent up",
        ],
    )
    actual = grids.best_worst_table(df, period="B")
    assert_frame_equal(actual, expected)

    # Monthly aggregation
    expected = pd.DataFrame(
        {
            "px1": [2.08, 1.505, -0.10166666667, -3.315, -5.32, 0.6666667],
            "px2": [3.54, 2.1766667, 0.40166667, -2.06, -2.64, 0.5],
        },
        index=[
            "best",
            "average up",
            "average",
            "average down",
            "worst",
            "percent up",
        ],
    )
    actual = grids.best_worst_table(df, period="ME")
    assert_frame_equal(actual, expected)


def test_seasonal_table():
    x1 = mock_time_series(100, 0.25, auto_regress=0.01, drift=0.05, seed=4321, nans=5, round_decimals=2)
    x2 = mock_time_series(100, 0.25, auto_regress=0.01, drift=0.05, seed=1234, nans=5, round_decimals=2)
    df = pd.DataFrame({"px1": x1, "px2": x2}, index=pd.bdate_range("2023-11-15", periods=100))
    df = transformation.to_pnl(df)

    # Monthly
    expected = pd.DataFrame(
        {
            "px1": [0.075454545, 0.11, 0.017647059, -2.66, -0.145555556, 0.099047619],
            "px2": [-0.07047619, 0.186315789, -0.125714286, np.nan, 0.211818182, 0.036666667],
        },
        index=["Jan", "Feb", "Mar", "Apr", "Nov", "Dec"],
    )

    actual = grids.seasonal_table(df, "ME")
    assert_frame_equal(actual, expected)

    # Day of Week
    expected = pd.DataFrame(
        {
            "px1": [0.67, -1.131764706, -0.3125, -0.292222222, 0.8415],
            "px2": [0.136111111, 0.061176471, 0.339444444, -0.383888889, -0.014736842],
        },
        index=["Mon", "Tue", "Wed", "Thu", "Fri"],
    )

    actual = grids.seasonal_table(df, "day")
    assert_frame_equal(actual, expected)

    # Week of Month
    expected = pd.DataFrame(
        {
            "px1": [-0.733, -0.201666667, 0.429444444, -0.1704, 0.382222222],
            "px2": [-0.421428571, -0.496111111, 0.377272727, 0.052272727, 0.23047619],
        },
        index=[1, 2, 3, 4, 5],
    )

    actual = grids.seasonal_table(df, "WeekofMonth")
    assert_frame_equal(actual, expected)
