"""
Tests for the statistics module
"""

from math import sqrt

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from pandas.testing import assert_series_equal
from pytest import approx

import portfolio.math.statistics as statutils
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


# END GLOBAL VARIABLES


@pytest.mark.parametrize("x, return_type", time_series_nan_list)
def test_mean(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return_type = float")
        actual = statutils.mean(x)
        assert actual == approx(102.42812500)

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        actual = statutils.mean(x)
        expected = pd.Series({"px1": 102.42812500, "px2": 102.460520833}, name="mean")
        assert_series_equal(actual, expected)


@pytest.mark.parametrize("x, return_type", time_series_01_list)
def test_mean_exp_weighted(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return_type = float")
        # as a list - window equal x length
        actual = statutils.mean_exp_weighted(x, 100, 50, annualize=False)
        assert actual == approx(103.0904464)

        # as a list - window greater than x length
        actual = statutils.mean_exp_weighted(x, 200, 25)
        assert actual == approx(103.5476904 * 252)

        # as a list - window less than x length
        actual = statutils.mean_exp_weighted(x, 50, 10)
        assert actual == approx(104.3199690 * 252)

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        # as a list - window equal x length
        actual = statutils.mean_exp_weighted(x, 100, 50)
        expected = pd.Series({"px1": 103.0904464 * 252, "px2": 103.0904464 * 2 * 252}, name="mean_exponential_weighted")
        assert_series_equal(actual, expected)

        # as a list - window greater than x length
        actual = statutils.mean_exp_weighted(x, 200, 25, annualize=False)
        expected = pd.Series({"px1": 103.5476904, "px2": 103.5476904 * 2}, name="mean_exponential_weighted")
        assert_series_equal(actual, expected)

        # as a list - window less than x length
        actual = statutils.mean_exp_weighted(x, 50, 10, annualize=False)
        expected = pd.Series({"px1": 104.3199690, "px2": 104.3199690 * 2}, name="mean_exponential_weighted")
        assert_series_equal(actual, expected)


@pytest.mark.parametrize("x, return_type", time_series_nan_list)
def test_mean_exp_weighted_with_nan(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return_type = float")
        # as a list - window equal x length
        actual = statutils.mean_exp_weighted(x, 50, 25, annualize=False)
        assert actual == approx(103.9143056)

        # as a list - window greater than x length
        actual = statutils.mean_exp_weighted(x, 200, 25, annualize=False)
        assert actual == approx(103.4268676)

        # as a list - window less than x length
        actual = statutils.mean_exp_weighted(x, 50, 10, annualize=False)
        assert actual == approx(104.2217369)

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        # as a list - window equal x length
        actual = statutils.mean_exp_weighted(x, 50, 10, annualize=False)
        expected = pd.Series({"px1": 104.2217369, "px2": 104.2404477}, name="mean_exponential_weighted")
        assert_series_equal(actual, expected)


@pytest.mark.parametrize("x, return_type", time_series_nan_list)
def test_min(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return_type = float")
        actual = statutils.min(x)
        assert actual == approx(96.97)

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        actual = statutils.min(x)
        expected = pd.Series({"px1": 96.97, "px2": 96.97}, name="min")
        assert_series_equal(actual, expected)


@pytest.mark.parametrize("x, return_type", time_series_nan_list)
def test_max(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return_type = float")
        actual = statutils.max(x)
        assert actual == approx(106.95)

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        actual = statutils.max(x)
        expected = pd.Series({"px1": 106.95, "px2": 106.95}, name="max")
        assert_series_equal(actual, expected)


@pytest.mark.parametrize("x, return_type", time_series_nan_list)
def test_percent_above(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return_type = float")
        actual = statutils.percentage_above(x, 100)
        assert actual == approx(0.875)

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        actual = statutils.percentage_above(x, 102.45, name='percent_greater')
        expected = pd.Series({"px1": 0.479166667, "px2": 0.489583333}, name="percent_greater")
        assert_series_equal(actual, expected)


@pytest.mark.parametrize("x, return_type", time_series_nan_list)
def test_stdev(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return_type = float")
        actual = statutils.stdev(x, annualize=False)
        assert actual == approx(2.077825041)

        actual = statutils.stdev(x)
        assert actual == approx(2.077825041 * sqrt(252))

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        actual = statutils.stdev(x, annualize=False)
        expected = pd.Series({"px1": 2.077825041, "px2": 2.106179583}, name="standard_deviation")
        assert_series_equal(actual, expected)

        actual = statutils.stdev(x)
        expected = pd.Series(
            {"px1": 2.077825041 * sqrt(252), "px2": 2.106179583 * sqrt(252)}, name="standard_deviation"
        )
        assert_series_equal(actual, expected)


@pytest.mark.parametrize("x, return_type", time_series_01_list)
def test_stdev_exp_weighted(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return_type = float")
        # as a list - window equal x length
        actual = statutils.stdev_exp_weighted(x, 100, 50, annualize=False)
        assert actual == approx(2.068694916)

        # as a list - window greater than x length
        actual = statutils.stdev_exp_weighted(x, 200, 25)
        assert actual == approx(1.954001321 * sqrt(252))

        # as a list - window less than x length
        actual = statutils.stdev_exp_weighted(x, 50, 10)
        assert actual == approx(1.54679696 * sqrt(252))

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        # as a list - window equal x length
        actual = statutils.stdev_exp_weighted(x, 100, 50, annualize=False)
        expected = pd.Series(
            {"px1": 2.068694916, "px2": 2.068694916 * 2}, name="standard_deviation_exponential_weighted"
        )
        assert_series_equal(actual, expected)

        # as a list - window greater than x length
        actual = statutils.stdev_exp_weighted(x, 200, 25)
        expected = pd.Series(
            {"px1": 1.954001321 * sqrt(252), "px2": 1.954001321 * 2 * sqrt(252)},
            name="standard_deviation_exponential_weighted",
        )
        assert_series_equal(actual, expected)

        # as a list - window less than x length
        actual = statutils.stdev_exp_weighted(x, 50, 10, annualize=False)
        expected = pd.Series({"px1": 1.54679696, "px2": 1.54679696 * 2}, name="standard_deviation_exponential_weighted")
        assert_series_equal(actual, expected)


@pytest.mark.parametrize("x, return_type", time_series_nan_list)
def test_stdev_exp_weighted_with_nan(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return_type = float")
        # as a list - window equal x length
        actual = statutils.stdev_exp_weighted(x, 50, 25, annualize=False)
        assert actual == approx(1.71049145)

        # as a list - window greater than x length
        actual = statutils.stdev_exp_weighted(x, 200, 25, annualize=False)
        assert actual == approx(1.939138144)

        # as a list - window less than x length
        actual = statutils.stdev_exp_weighted(x, 50, 10, annualize=False)
        assert actual == approx(1.525678244)

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        # as a list - window equal x length
        actual = statutils.stdev_exp_weighted(x, 50, 10, annualize=False)
        expected = pd.Series({"px1": 1.525678244, "px2": 1.553862322}, name="standard_deviation_exponential_weighted")
        assert_series_equal(actual, expected)


@pytest.mark.parametrize("x, return_type", time_series_01_list)
def test_downside_deviation(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return_type = float")
        x = np.diff(x)  # make it a change series from levels
        assert statutils.downside_deviation(x, annualize=False) == approx(2.55412520)

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        # as a list - window equal x length
        x = x.diff(1)  # make it a change series from levels
        actual = statutils.downside_deviation(x)
        expected = pd.Series(
            {"px1": 2.55412520 * np.sqrt(252), "px2": 2.55412520 * 2 * np.sqrt(252)}, name="downside_deviation"
        )
        assert_series_equal(actual, expected)


@pytest.mark.parametrize("x, return_type", time_series_nan_list)
def test_downside_deviation_nan(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return_type = float")
        x = np.diff(x)  # make it a change series from levels
        assert statutils.downside_deviation(x, annualize=False) == approx(2.55842129)

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        # as a list - window equal x length
        x = x.diff(1)  # make it a change series from levels
        actual = statutils.downside_deviation(x)
        expected = pd.Series(
            {"px1": 2.55842129 * np.sqrt(252), "px2": 2.50626824 * np.sqrt(252)}, name="downside_deviation"
        )
        assert_series_equal(actual, expected)


@pytest.mark.parametrize("x, return_type", time_series_01_list)
def test_downside_deviation_exp_weighted(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return_type = float")
        x = np.diff(x)  # make it a change series from levels
        assert statutils.downside_deviation_exp_weighted(x, 50, 5, annualize=False) == approx(1.85189701)

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        # as a list - window equal x length
        x = x.diff(1)  # make it a change series from levels
        actual = statutils.downside_deviation_exp_weighted(x, 200, 10)
        expected = pd.Series(
            {"px1": 2.16698091 * np.sqrt(252), "px2": 2.16698091 * 2 * np.sqrt(252)},
            name="downside_deviation_exponential_weighted",
        )
        assert_series_equal(actual, expected)


@pytest.mark.parametrize("x, return_type", time_series_nan_list)
def test_downside_deviation_exp_weighted_nan(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return_type = float")
        x = np.diff(x)  # make it a change series from levels
        assert statutils.downside_deviation_exp_weighted(x, 50, 20, annualize=False) == approx(2.42372900)

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        # as a list - window equal x length
        x = x.diff(1)  # make it a change series from levels
        actual = statutils.downside_deviation_exp_weighted(x, 200, 25)
        expected = pd.Series(
            {"px1": 2.46713128 * np.sqrt(252), "px2": 2.42687766 * np.sqrt(252)},
            name="downside_deviation_exponential_weighted",
        )
        assert_series_equal(actual, expected)


@pytest.mark.parametrize("x, return_type", time_series_01_list)
def test_cagr(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return_type = float")
        assert statutils.cagr(x, annual_periods=365.25) == approx(0.12604629)

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        actual = statutils.cagr(x)
        expected = pd.Series(
            {"px1": 0.12604629, "px2": 0.12604629},
            name="cagr",
        )
        assert_series_equal(actual, expected)


def test_odr():
    y = [53.0, 58, 60, 65, 68, 72, 73, 75, 78, 79, 80, 81, 84, 85, 92]
    x = [48.0, 29, 25, 23, 14, 35, 4, 19, 23, 26, 5, 11, 8, 17, 4]

    expected = (-0.83702332755446796, 89.771586489367337)
    actual = statutils.orthogonal_distance_regression(x, y)
    assert_almost_equal(actual, expected, decimal=4)

    reverse_actual = statutils.orthogonal_distance_regression(y, x)
    assert reverse_actual[0] == approx(1 / actual[0], 0.001, 0.001)


def test_tls():
    y = [53.0, 58, 60, 65, 68, 72, 73, 75, 78, 79, 80, 81, 84, 85, 92]
    x = [48.0, 29, 25, 23, 14, 35, 4, 19, 23, 26, 5, 11, 8, 17, 4]

    expected = (-0.837050285, 89.772108862)
    actual = statutils.total_least_squares(x, y)
    assert_almost_equal(actual, expected)

    reverse_actual = statutils.total_least_squares(y, x)
    assert reverse_actual[0] == approx(1 / actual[0])


def test_weighted_odr():
    x = [x * 1.0 for x in range(1, 31)]
    y = [
        1.1,
        1.1,
        2.2,
        2.2,
        4.3,
        5.7,
        6.4,
        7.6,
        8.5,
        9.5,  # Left bucket
        10.2,
        9.7,
        10.2,
        9.8,
        10.2,
        9.9,
        10.2,
        9,
        10.2,
        9.2,  # Middle bucket
        9.5,
        8.5,
        7.6,
        6.4,
        5.7,
        4.3,
        2.2,
        2.2,
        1.1,
        1.1,
    ]  # Right bucket

    # Run the entire selection with 1.0 weight confirm same as non-weighted
    noweight = statutils.orthogonal_distance_regression(x, y)
    w = [1.0] * 30
    weighted = statutils.orthogonal_distance_regression(x, y, w)

    expected = (-0.002785287, 6.569838612)
    assert_almost_equal(weighted, expected, decimal=3)
    assert_almost_equal(weighted, noweight, decimal=4)

    # Confirm that weighting each bucket at 1.0 and non-bucket at 0.0 equals the un-weighted bucket only
    noweight = statutils.orthogonal_distance_regression(x[0:10], y[0:10])
    w = [1.0] * 10 + [0.0] * 10 + [0.0] * 10
    weighted = statutils.orthogonal_distance_regression(x, y, w)

    expected = (1.033831466, -0.826073064)
    assert_almost_equal(weighted, expected, decimal=5)
    assert_almost_equal(weighted, noweight, decimal=5)

    noweight = statutils.orthogonal_distance_regression(x[10:20], y[10:20])
    w = [0.0] * 10 + [1.0] * 10 + [0.0] * 10
    weighted = statutils.orthogonal_distance_regression(x, y, w)

    expected = (-0.065381632, 10.873415295)
    assert_almost_equal(weighted, expected, decimal=5)
    assert_almost_equal(weighted, noweight, decimal=5)

    noweight = statutils.orthogonal_distance_regression(x[20:30], y[20:30])
    w = [0.0] * 10 + [0.0] * 10 + [1.0] * 10
    weighted = statutils.orthogonal_distance_regression(x, y, w)

    expected = (-1.033831466, 31.222702388)
    assert_almost_equal(weighted, expected, decimal=5)
    assert_almost_equal(weighted, noweight, decimal=5)

    # Now weight the entire with ascending weights
    w = [(x + 1.0) / 10.0 for x in range(30)]
    weighted = statutils.orthogonal_distance_regression(x, y, w)

    expected = (-0.30792614475046898, 12.755303812599847)
    assert_almost_equal(weighted, expected, decimal=5)


def test_weighted_tls():
    x = [x * 1.0 for x in range(1, 31)]
    y = [
        1.1,
        1.1,
        2.2,
        2.2,
        4.3,
        5.7,
        6.4,
        7.6,
        8.5,
        9.5,  # Left bucket
        10.2,
        9.7,
        10.2,
        9.8,
        10.2,
        9.9,
        10.2,
        9,
        10.2,
        9.2,  # Middle bucket
        9.5,
        8.5,
        7.6,
        6.4,
        5.7,
        4.3,
        2.2,
        2.2,
        1.1,
        1.1,
    ]  # Right bucket

    # Run the entire selection with 1.0 weight confirm same as non-weighted
    noweight = statutils.total_least_squares(x, y)
    w = [1.0] * 30
    weighted = statutils.total_least_squares(x, y, w)

    expected = (-0.002785287, 6.569838612)
    assert_almost_equal(weighted, expected)
    assert_almost_equal(weighted, noweight)

    # Confirm that weighting each bucket at 1.0 and non-bucket at 0.0 equals the un-weighted bucket only
    noweight = statutils.total_least_squares(x[0:10], y[0:10])
    w = [1.0] * 10 + [0.0] * 10 + [0.0] * 10
    weighted = statutils.total_least_squares(x, y, w)

    expected = (1.033831466, -0.826073064)
    assert_almost_equal(weighted, expected)
    assert_almost_equal(weighted, noweight)

    noweight = statutils.total_least_squares(x[10:20], y[10:20])
    w = [0.0] * 10 + [1.0] * 10 + [0.0] * 10
    weighted = statutils.total_least_squares(x, y, w)

    expected = (-0.065381632, 10.873415295)
    assert_almost_equal(weighted, expected)
    assert_almost_equal(weighted, noweight)

    noweight = statutils.total_least_squares(x[20:30], y[20:30])
    w = [0.0] * 10 + [0.0] * 10 + [1.0] * 10
    weighted = statutils.total_least_squares(x, y, w)

    expected = (-1.033831466, 31.222702388)
    assert_almost_equal(weighted, expected)
    assert_almost_equal(weighted, noweight)

    # Now weight the entire with ascending weights
    w = [(x + 1.0) / 10.0 for x in range(30)]
    weighted = statutils.total_least_squares(x, y, w)

    expected = (-0.305357531, 12.297008485)
    assert_almost_equal(weighted, expected)


@pytest.mark.parametrize("x, return_type", time_series_01_list)
def test_ttest_1sample_1side(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return_type = float")
        x = np.diff(x)
        assert statutils.ttest_1sample_1side(x) == approx(0.4468684)

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        x = x.diff(1)  # make it a change series from levels
        actual = statutils.ttest_1sample_1side(x)
        expected = pd.Series(
            {"px1": 0.4468684, "px2": 0.4468684},
            name="ttest_1sample_1side_pvalue",
        )
        assert_series_equal(actual, expected)


def test_ttest_2sample_2side():
    x_1 = [-3.27, 2.18, -2.52, 0.86, 2.84, -0.45, -0.48, 2.13, -2.18, -4.5, -2.86, 2.29, 0.91, -1.23, 3.26]
    x_2 = [-0.89, -1.98, -0.62, -2.89, -3.39, 1.9, 1.12, 0.88, 3.21, 2.41, 4.94, 2.85, 1.94, -1.67, -2.5]

    assert statutils.ttest_2sample_2side(x_1, x_2) == approx(0.549562559)

    x_1 = pd.Series(x_1)
    assert statutils.ttest_2sample_2side(x_1, x_2) == approx(0.549562559)

    x_2 = pd.Series(x_2)
    assert statutils.ttest_2sample_2side(x_1, x_2) == approx(0.549562559)


def test_first_date():
    ser = pd.Series([1, 2, 3], pd.date_range("20130101", periods=3))
    assert statutils.first_date(ser) == pd.Timestamp("20130101")

    df = pd.DataFrame({"first": [np.nan, "b", "c"], "second": [1, np.nan, np.nan]}, index=["aa", "bb", "cc"])
    assert_series_equal(statutils.first_date(df), pd.Series(["bb", "aa"], index=["first", "second"], name="first_date"))


def test_last_date():
    ser = pd.Series([1, 2, 3], pd.date_range("20130101", periods=3))
    assert statutils.last_date(ser) == pd.Timestamp("20130103")

    df = pd.DataFrame({"first": [np.nan, "b", "c"], "second": [1, np.nan, np.nan]}, index=["aa", "bb", "cc"])
    assert_series_equal(statutils.last_date(df), pd.Series(["cc", "aa"], index=["first", "second"], name="last_date"))


def test_observations_count():
    ser = pd.Series([1, 2, 3], pd.date_range("20130101", periods=3))
    assert statutils.observations_count(ser) == 3

    df = pd.DataFrame({"first": [np.nan, 3, 4.0], "second": [1, np.nan, np.nan]}, index=["aa", "bb", "cc"])
    expected = pd.Series([2, 1], index=["first", "second"], name="observations_count")
    actual = statutils.observations_count(df)
    assert_series_equal(actual, expected)
