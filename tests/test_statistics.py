"""
Tests for the statistics module
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from pandas.testing import assert_series_equal, assert_frame_equal
from pytest import approx

import portfolio.math.statistics as statutils
from math import sqrt
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


@pytest.mark.parametrize(
    "x, expected",
    [
        ([1, 2, 0, 3], 75.0),  # 3 out of 4 values are non-zero
        ([0, 0, 0, 0], 0.0),  # no non-zero values
        ([1, 2, 3, 4], 100.0),  # all values are non-zero
    ],
)
def test_percentage_non_zero_list(x, expected):
    actual = statutils.percentage_non_zero(x)
    assert_almost_equal(actual * 100, expected)


@pytest.mark.parametrize(
    "x, expected",
    [
        (np.array([1, 2, 0.00000000001, 3]), 75.0),  # 3 out of 4 values are non-zero
        (np.array([0, 0, 0, 0]), 0.0),  # no non-zero values
        (np.array([1, 2, 3, 4]), 100.0),  # all values are non-zero
    ],
)
def test_percentage_non_zero_ndarray(x, expected):
    actual = statutils.percentage_non_zero(x)
    assert_almost_equal(actual * 100, expected)


@pytest.mark.parametrize(
    "x, expected",
    [
        (pd.Series([1, 2, 0, 3]), 75.0),  # 3 out of 4 values are non-zero
        (pd.Series([0, 0, 0, 0]), 0.0),  # no non-zero values
        (pd.Series([1, 2, 3, 4]), 100.0),  # all values are non-zero
    ],
)
def test_percentage_non_zero_series(x, expected):
    actual = statutils.percentage_non_zero(x)
    assert_almost_equal(actual * 100, expected)


def test_percentage_non_zero_dataframe():
    df = pd.DataFrame({"col1": [1, 2, 0, 3], "col2": [0, 0, 3, 4]})
    actual = statutils.percentage_non_zero(df, name="custom_name")
    expected = pd.Series({"col1": 75.0, "col2": 50.0}, name="custom_name")
    assert_series_equal(actual * 100, expected)


@pytest.mark.parametrize(
    "x, expected",
    [
        ([1, 2, 0, np.nan], 66.67),  # ignores NaN, 2 out of 3 are non-zero
        (np.array([0, 0, np.nan, 0]), 0.0),  # all zeros after ignoring NaN
        (pd.Series([np.nan, 0, 3, np.nan]), 50.0),  # 1 out of 2 is non-zero after ignoring NaN
    ],
)
def test_percentage_non_zero_with_nans(x, expected):
    actual = statutils.percentage_non_zero(x)
    assert_almost_equal(actual * 100, expected, decimal=2)


@pytest.mark.parametrize(
    "number, digits, base, expected",
    [
        (123.456, 2, 10, 123.45),
        (123.456, 1, 10, 123.4),
        (123.456, 0, 10, 123.0),
        (129.456, -1, 10, 120.0),
        (120.456, -2, 10, 100.0),
        (-123.456, 2, 10, -123.46),
        (-123.456, 1, 10, -123.5),
        (-123.456, 0, 10, -124.0),
        (-123.456, -1, 10, -130.0),
        (-123.456, -2, 10, -200.0),
        (0.0, 2, 10, 0.0),
        (0.0, -2, 10, 0.0),
        (30, -1, 25, 25),
        (5, -1, 25, 0),
        (-5, -1, 25, -25),
    ],
)
def test_round_down(number, digits, base, expected):
    actual = statutils.round_down(number, digits, base)
    assert actual == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize(
    "number, digits, base, expected",
    [
        (123.456, 2, 10, 123.46),
        (123.456, 1, 10, 123.5),
        (123.456, 0, 10, 124.0),
        (129.456, -1, 10, 130.0),
        (120.456, -2, 10, 200.0),
        (-123.456, 2, 10, -123.45),
        (-123.456, 1, 10, -123.4),
        (-123.456, 0, 10, -123.0),
        (-123.456, -1, 10, -120.0),
        (-123.456, -2, 10, -100.0),
        (0.0, 2, 10, 0.0),
        (0.0, -2, 10, 0.0),
        (30, -1, 25, 50),
        (5, -1, 25, 25),
        (-5, -1, 25, 0),
    ],
)
def test_round_up(number, digits, base, expected):
    actual = statutils.round_up(number, digits, base)
    assert actual == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize(
    "min_value, max_value, bins, digits, bin_size, expected",
    [
        (0, 10, [0, 2, 4, 6, 8, 10], None, None, [0, 2, 4, 6, 8, 10]),
        (0, 10, 5, None, None, 5),
        (0, 10, None, 0, None, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        (
                1.5,
                2.5,
                None,
                1,
                None,
                [
                    1.5,
                    1.6,
                    1.7,
                    1.8,
                    1.9,
                    2.0,
                    2.1,
                    2.2,
                    2.3,
                    2.4,
                    2.5,
                ],
        ),
        (2, 8, None, -1, None, [0, 10]),
        (25, 85, None, -1, None, [20, 30, 40, 50, 60, 70, 80, 90]),
        (6, 125, None, None, 25, [0, 25, 50, 75, 100, 125]),
    ],
)
def test_bin_list(min_value, max_value, bins, digits, bin_size, expected):
    actual = statutils.bin_list(min_value=min_value, max_value=max_value, bins=bins, bin_size=bin_size, digits=digits)
    assert_array_equal(actual, expected)


def test_bin_list_errors():
    with pytest.raises(AssertionError, match="min_value=.* must be less than or equal to max_value=.*"):
        statutils.bin_list(min_value=10, max_value=0, bins=5)
    with pytest.raises(AssertionError, match="only supply one of bins, digits or bin_size"):
        statutils.bin_list(min_value=0, max_value=10, bins=5, digits=2)


@pytest.mark.parametrize(
    "x, bins, normalize, expected",
    [
        (
                [1, 2, 2, 3, 3, 3],
                3,
                True,
                pd.Series(
                    [0.1667, 0.3333, 0.5],
                    index=pd.CategoricalIndex(
                        pd.IntervalIndex.from_tuples([(0.997, 1.667), (1.667, 2.333), (2.333, 3.0)]), ordered=True
                    ),
                ),
        ),
        (
                [1, 2, 3, 4, 5],
                [0, 2, 4, 6],
                False,
                pd.Series(
                    [2, 2, 1],
                    index=pd.CategoricalIndex(pd.IntervalIndex.from_tuples([(-0.001, 2), (2, 4), (4, 6)]),
                                              ordered=True),
                ),
        ),
    ],
)
def test_histogram_list(x, bins, normalize, expected):
    actual = statutils.histogram(x, bins=bins, normalize=normalize)
    assert_series_equal(actual, expected, atol=0.0001, check_dtype=False)


@pytest.mark.parametrize(
    "x, bins, normalize, expected",
    [
        (
                pd.Series([1, 2, 2, 3, 3, 3]),
                3,
                True,
                pd.Series(
                    [0.1667, 0.3333, 0.5],
                    index=pd.CategoricalIndex(
                        pd.IntervalIndex.from_tuples([(0.997, 1.667), (1.667, 2.333), (2.333, 3.0)]), ordered=True
                    ),
                ),
        ),
        (
                pd.Series([1, 2, 3, 4]),
                [0, 2, 4],
                False,
                pd.Series(
                    [2, 2],
                    index=pd.CategoricalIndex(pd.IntervalIndex.from_tuples([(-0.001, 2), (2, 4)]), ordered=True),
                ),
        ),
    ],
)
def test_histogram_series(x, bins, normalize, expected):
    actual = statutils.histogram(x, bins=bins, normalize=normalize)
    assert_series_equal(actual, expected, atol=0.0001, check_dtype=False)


@pytest.mark.parametrize(
    "x, bins, normalize, expected",
    [
        (
                pd.DataFrame({"A": [1, 2, 3], "B": [2, 2, 4]}),
                [0, 2, 4],
                True,
                pd.DataFrame(
                    {
                        "A": pd.Series(
                            [0.6667, 0.3333],
                            index=pd.CategoricalIndex(pd.IntervalIndex.from_tuples([(-0.001, 2), (2, 4)]),
                                                      ordered=True),
                        ),
                        "B": pd.Series(
                            [0.6667, 0.3333],
                            index=pd.CategoricalIndex(pd.IntervalIndex.from_tuples([(-0.001, 2), (2, 4)]),
                                                      ordered=True),
                        ),
                    }
                ),
        )
    ],
)
def test_histogram_dataframe(x, bins, normalize, expected):
    actual = statutils.histogram(x, bins=bins, normalize=normalize)
    assert_frame_equal(actual, expected, atol=0.0001, check_dtype=False)


@pytest.mark.parametrize(
    "x, bin_size, normalize, expected",
    [
        (
                pd.DataFrame({"A": [1, 20, 30, 25], "B": [21, 27, 40, 50]}),
                25,
                True,
                pd.DataFrame(
                    {
                        "A": pd.Series(
                            [0.75, 0.25],
                            index=pd.CategoricalIndex(pd.IntervalIndex.from_tuples([(-0.001, 25), (25, 50)]),
                                                      ordered=True),
                        ),
                        "B": pd.Series(
                            [0.25, 0.75],
                            index=pd.CategoricalIndex(pd.IntervalIndex.from_tuples([(-0.001, 25), (25, 50)]),
                                                      ordered=True),
                        ),
                    }
                ),
        )
    ],
)
def test_histogram_bin_size(x, bin_size, normalize, expected):
    actual = statutils.histogram(x, bin_size=bin_size, normalize=normalize)
    assert_frame_equal(actual, expected, atol=0.0001, check_dtype=False)
