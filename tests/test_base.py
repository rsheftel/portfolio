"""
Unit tests for the base functions
"""
import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal, assert_equal
from pandas.testing import assert_series_equal, assert_frame_equal

from portfolio.math import base, statistics
from testing import mock_time_series

x = mock_time_series(100, 0.25, auto_regress=0.01, drift=0.05, seed=1234, round_decimals=2)
dates = pd.date_range("2024-01-01", periods=100, freq="1D")
time_series_01_series = pd.Series(x, index=dates)
time_series_01_df = pd.DataFrame({"px1": x, "px2": x * 2}, index=pd.date_range("2024-01-01", periods=100, freq="1D"))


def test_exp_weights():
    expected = [
        0.120683934,
        0.105061466,
        0.091461319,
        0.079621703,
        0.069314718,
        0.060341967,
        0.052530733,
        0.045730659,
        0.039810851,
        0.034657359,
    ]

    actual = base.exponential_weights(window_length=10, half_life=5)
    assert_almost_equal(actual, expected)


def test_dropna():
    # lists
    x = [1.1, 2, 3, None]
    other = [5, 6, 7, 8]

    actual_x, actual_other = base.dropna(x, other)
    assert actual_x == [1.1, 2, 3]
    assert actual_other == [5, 6, 7]

    # numpy array
    x = np.array([1.1, np.nan, 3, None])
    other = np.array([1, 2, 3, 4])

    actual_x, actual_other = base.dropna(x, other)
    assert_equal(actual_x, np.array([1.1, 3]))
    assert_equal(actual_other, np.array([1, 3]))

    # pandas Series
    x = pd.Series([1.1, np.nan, 3, None])
    other = pd.Series([1, 2, 3, 4])

    actual_x, actual_other = base.dropna(x, other)
    assert_series_equal(actual_x, x[[0, 2]])
    assert_series_equal(actual_other, other[[0, 2]])

    # no other provided
    x = [1.1, 2, 3, None]
    assert base.dropna(x) == [1.1, 2, 3]

    x = np.array([1.1, np.nan, 3, None])
    assert_equal(base.dropna(x), np.array([1.1, 3]))

    x = pd.Series([1.1, np.nan, 3, None])
    assert_series_equal(base.dropna(x), x[[0, 2]])


def test_rolling():
    expected = pd.DataFrame(
        {
            "px1": [
                100,
                200.79,
                299.02,
                401.41,
                501.14,
                600.25,
                701.94,
                803.66,
                903.07,
                1003.53,
                1000.5,
                1002.04,
                1005.99,
                1005.77,
                1003.57,
                1004.65,
                1003.76,
                1003.53,
                1005.48,
                1008.06,
            ]
        },
        index=pd.date_range("2024-01-01", periods=20),
    )
    expected["px2"] = expected["px1"] * 2
    actual = base.rolling(time_series_01_df, sum, 10)[:20]
    assert_frame_equal(actual, expected)

    # test with kwargs
    def summer(x, mult):
        return sum(x) * mult

    expected = expected * 10
    actual = base.rolling(time_series_01_df, summer, 10, mult=10)[:20]
    assert_frame_equal(actual, expected)


def test_rolling_exp_weighted():
    # test with an exponentially weighted function
    expected = pd.Series(
        [
            100,
            100.408684178417,
            99.631577259431,
            100.394450016327,
            100.242530384842,
            100.019626480249,
            100.310604186792,
            100.532342326041,
            100.370399298099,
            100.382399880024,
        ],
        index=pd.date_range("2024-01-01", periods=10),
        name="px1",
    )

    actual = base.rolling(
        time_series_01_df["px1"],
        statistics.mean_exp_weighted,
        window=50,
        min_periods=1,
        window_length=50,
        half_life=10,
        annualize=False,
    )
    assert_series_equal(actual[:10], expected)

    expected = pd.Series(
        [
            0,
            0.394762894990199,
            1.09060897318624,
            1.54364147732881,
            1.38415432759397,
            1.31968162545865,
            1.35630186915417,
            1.34665859026705,
            1.3066562438438,
            1.21639198867236,
        ],
        index=pd.date_range("2024-01-01", periods=10),
        name="px1",
    )

    actual = base.rolling(
        time_series_01_df["px1"],
        statistics.stdev_exp_weighted,
        window=50,
        min_periods=1,
        window_length=50,
        half_life=10,
        annualize=False,
    )
    assert_series_equal(actual[:10], expected)


def test_prior_index():
    assert base.prior_index([9, 2, 3]) == 0
    assert base.prior_index(np.array([9, 2, 3])) == 0
    assert base.prior_index(pd.Series([9, 2, 3], index=[11, 12, 13])) == 10
    assert base.prior_index(pd.Series([9, 2, 3], index=[11.5, 12, 13])) == 10.5
    assert base.prior_index(pd.Series([9, 2, 3],
                                      index=pd.date_range("2024-11-11", periods=3))) == pd.Timestamp("2024-11-08"
                                                                                                     )
