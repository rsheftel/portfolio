"""
Tests for the statistics module
"""

from math import sqrt

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from pandas.testing import assert_series_equal, assert_frame_equal
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
    {"px1": x_nan, "px2": x_nan2},
    index=pd.date_range("2024-01-01", periods=100, freq="1D"),
)
time_series_all_nan_df = pd.DataFrame(
    {"nan": [np.nan] * 5, "None": [None] * 5}, index=pd.date_range("2024-01-01", periods=5)
)
time_series_all_zero_series = pd.Series([0.0] * 5, index=pd.date_range("2024-01-01", periods=5))
time_series_nan_list = [
    (list(x_nan), "float"),
    (np.array(x_nan), "float"),
    (time_series_nan_series, "float"),
    (time_series_nan_df, "pd.Series"),
    (time_series_all_nan_df, "nan"),
    (time_series_all_zero_series, "zeros"),
]


# END GLOBAL VARIABLES


@pytest.mark.parametrize("x, return_type", time_series_nan_list)
def test_mean(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return_type = float")
        actual = statutils.mean(x)
        assert actual == approx(102.42812500)
        actual = statutils.mean(x, annualize=True)
        assert actual == approx(102.42812500 * 252)

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        expected = pd.Series({"px1": 102.42812500, "px2": 102.460520833}, name="mean")
        actual = statutils.mean(x)
        assert_series_equal(actual, expected)
        actual = statutils.mean(x, annualize=True)
        assert_series_equal(actual, expected * 252)

    if return_type == "nan":
        print("testing return_type = nan")
        expected = pd.Series({"nan": np.nan, "None": np.nan}, name="mean")
        actual = statutils.mean(x)
        assert_series_equal(actual, expected)

    if return_type == "zeros":
        print("testing return_type = zeros")
        actual = statutils.mean(x)
        assert actual == 0.0


def test_mean_all_nans():
    assert np.isnan(statutils.mean([None, None, None]))
    assert np.isnan(statutils.mean([np.nan, np.nan, np.nan]))
    assert np.isnan(statutils.mean([]))


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
        expected = pd.Series(
            {"px1": 103.0904464 * 252, "px2": 103.0904464 * 2 * 252},
            name="mean_exponential_weighted",
        )
        assert_series_equal(actual, expected)

        # as a list - window greater than x length
        actual = statutils.mean_exp_weighted(x, 200, 25, annualize=False)
        expected = pd.Series(
            {"px1": 103.5476904, "px2": 103.5476904 * 2},
            name="mean_exponential_weighted",
        )
        assert_series_equal(actual, expected)

        # as a list - window less than x length
        actual = statutils.mean_exp_weighted(x, 50, 10, annualize=False)
        expected = pd.Series(
            {"px1": 104.3199690, "px2": 104.3199690 * 2},
            name="mean_exponential_weighted",
        )
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

    if return_type == "nan":
        print("testing return_type = nan")
        expected = pd.Series({"nan": np.nan, "None": np.nan}, name="mean_exponential_weighted")
        actual = statutils.mean_exp_weighted(x, 5, 2)
        assert_series_equal(actual, expected)

    if return_type == "zeros":
        print("testing return_type = zeros")
        actual = statutils.mean_exp_weighted(x, 5, 2)
        assert actual == 0.0


def test_mean_exp_weighted_all_nan():
    # if there are no values, returns np.nan
    actual = statutils.mean_exp_weighted([np.nan] * 50, 50, 10)
    assert np.isnan(actual)


@pytest.mark.parametrize("x, return_type", time_series_nan_list)
def test_median(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return_type = float")
        actual = statutils.median(x)
        assert actual == approx(102.4)
        actual = statutils.median(x, annualize=True)
        assert actual == approx(102.4 * 252)

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        expected = pd.Series({"px1": 102.4, "px2": 102.42}, name="median")
        actual = statutils.median(x)
        assert_series_equal(actual, expected)
        actual = statutils.median(x, annualize=True)
        assert_series_equal(actual, expected * 252)

    if return_type == "nan":
        print("testing return_type = nan")
        expected = pd.Series({"nan": np.nan, "None": np.nan}, name="median")
        actual = statutils.median(x)
        assert_series_equal(actual, expected)

    if return_type == "zeros":
        print("testing return_type = zeros")
        actual = statutils.median(x)
        assert actual == 0.0


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

    if return_type == "nan":
        print("testing return_type = nan")
        expected = pd.Series({"nan": np.nan, "None": np.nan}, name="min")
        actual = statutils.min(x)
        assert_series_equal(actual, expected)

    if return_type == "zeros":
        print("testing return_type = zeros")
        actual = statutils.min(x)
        assert actual == 0.0


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

    if return_type == "nan":
        print("testing return_type = nan")
        expected = pd.Series({"nan": np.nan, "None": np.nan}, name="max")
        actual = statutils.max(x)
        assert_series_equal(actual, expected)

    if return_type == "zeros":
        print("testing return_type = zeros")
        actual = statutils.max(x)
        assert actual == 0.0


@pytest.mark.parametrize("x, return_type", time_series_nan_list)
def test_percent_above(x, return_type):
    print(f"testing - {type(x)}")
    if return_type == "float":
        print("testing return_type = float")
        actual = statutils.percentage_above(x, 100)
        assert actual == approx(0.875)

    if return_type == "pd.Series":
        print("testing return_type = pd.Series")
        actual = statutils.percentage_above(x, 102.45, name="percent_greater")
        expected = pd.Series({"px1": 0.479166667, "px2": 0.489583333}, name="percent_greater")
        assert_series_equal(actual, expected)

    if return_type == "nan":
        print("testing return_type = nan")
        expected = pd.Series({"nan": np.nan, "None": np.nan}, name="percentage_above")
        actual = statutils.percentage_above(x)
        assert_series_equal(actual, expected)

    if return_type == "zeros":
        print("testing return_type = zeros")
        actual = statutils.percentage_above(x, exclude_zeros=True)
        assert np.isnan(actual)
        actual = statutils.percentage_above(x, exclude_zeros=False)
        assert actual == 0.0


def test_percent_above_with_zeros():
    ser = pd.Series([1, -2, -3, 0, 0, 0])
    assert statutils.percentage_above(ser) == 1 / 6
    assert statutils.percentage_above(ser, exclude_zeros=True) == 1 / 3

    ser = pd.Series([-1, -2, -3, 0, 0, 0])
    assert statutils.percentage_above(ser) == 0
    assert statutils.percentage_above(ser, exclude_zeros=True) == 0


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
            {"px1": 2.077825041 * sqrt(252), "px2": 2.106179583 * sqrt(252)},
            name="standard_deviation",
        )
        assert_series_equal(actual, expected)

    if return_type == "nan":
        print("testing return_type = nan")
        expected = pd.Series({"nan": np.nan, "None": np.nan}, name="standard_deviation")
        actual = statutils.stdev(x)
        assert_series_equal(actual, expected)

    if return_type == "zeros":
        print("testing return_type = zeros")
        actual = statutils.stdev(x)
        assert actual == 0.0


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
            {"px1": 2.068694916, "px2": 2.068694916 * 2},
            name="standard_deviation_exponential_weighted",
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
        expected = pd.Series(
            {"px1": 1.54679696, "px2": 1.54679696 * 2},
            name="standard_deviation_exponential_weighted",
        )
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
        expected = pd.Series(
            {"px1": 1.525678244, "px2": 1.553862322},
            name="standard_deviation_exponential_weighted",
        )
        assert_series_equal(actual, expected)

    if return_type == "nan":
        print("testing return_type = nan")
        expected = pd.Series({"nan": np.nan, "None": np.nan}, name="standard_deviation_exponential_weighted")
        actual = statutils.stdev_exp_weighted(x, 5, 2)
        assert_series_equal(actual, expected)

    if return_type == "zeros":
        print("testing return_type = zeros")
        actual = statutils.stdev_exp_weighted(x, 5, 2)
        assert actual == 0.0


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
            {"px1": 2.55412520 * np.sqrt(252), "px2": 2.55412520 * 2 * np.sqrt(252)},
            name="downside_deviation",
        )
        assert_series_equal(actual, expected)


def test_downside_deviation_all_up():
    assert statutils.downside_deviation([1, 2, 3, 4]) == 0
    assert statutils.downside_deviation_exp_weighted([1, 2, 3, 4], 5, 2) == 0


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
            {"px1": 2.55842129 * np.sqrt(252), "px2": 2.50626824 * np.sqrt(252)},
            name="downside_deviation",
        )
        assert_series_equal(actual, expected)

    if return_type == "nan":
        print("testing return_type = nan")
        expected = pd.Series({"nan": np.nan, "None": np.nan}, name="downside_deviation")
        actual = statutils.downside_deviation(x)
        assert_series_equal(actual, expected)

    if return_type == "zeros":
        print("testing return_type = zeros")
        actual = statutils.downside_deviation(x)
        assert actual == 0.0


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

    if return_type == "nan":
        print("testing return_type = nan")
        expected = pd.Series({"nan": np.nan, "None": np.nan}, name="downside_deviation_exponential_weighted")
        actual = statutils.downside_deviation_exp_weighted(x, 5, 2)
        assert_series_equal(actual, expected)

    if return_type == "zeros":
        print("testing return_type = zeros")
        actual = statutils.downside_deviation_exp_weighted(x, 5, 2)
        assert actual == 0.0


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

    df = pd.DataFrame(
        {"first": [np.nan, "b", "c"], "second": [1, np.nan, np.nan]},
        index=["aa", "bb", "cc"],
    )
    assert_series_equal(
        statutils.first_date(df),
        pd.Series(["bb", "aa"], index=["first", "second"], name="first_date"),
    )

    # test MultiIndex
    columns = pd.MultiIndex.from_tuples([("A", "1"), ("B", "2")])
    df = pd.DataFrame([[np.nan, 1], ["b", np.nan], ["c", np.nan]], index=["aa", "bb", "cc"], columns=columns)
    assert_series_equal(statutils.first_date(df), pd.Series(["bb", "aa"], index=columns, name="first_date"))

    # with DateTimeIndex
    df = pd.DataFrame(
        {"first": [np.nan, "b", "c"], "second": [1, np.nan, np.nan]},
        index=pd.date_range("2024-04-05", periods=3),
    )
    expected = pd.Series(
        [pd.Timestamp("2024-04-06"), pd.Timestamp("2024-04-05")], index=["first", "second"], name="first_date"
    )
    actual = statutils.first_date(df)
    assert_series_equal(actual, expected)


def test_last_date():
    ser = pd.Series([1, 2, 3], pd.date_range("20130101", periods=3))
    assert statutils.last_date(ser) == pd.Timestamp("20130103")

    df = pd.DataFrame(
        {"first": [np.nan, "b", "c"], "second": [1, np.nan, np.nan]},
        index=["aa", "bb", "cc"],
    )
    assert_series_equal(
        statutils.last_date(df),
        pd.Series(["cc", "aa"], index=["first", "second"], name="last_date"),
    )

    # test MultiIndex
    columns = pd.MultiIndex.from_tuples([("A", "1"), ("B", "2")])
    df = pd.DataFrame([[np.nan, 1], ["b", np.nan], ["c", np.nan]], index=["aa", "bb", "cc"], columns=columns)
    assert_series_equal(
        statutils.last_date(df),
        pd.Series(["cc", "aa"], index=columns, name="last_date"),
    )
    # with DateTimeIndex
    df = pd.DataFrame(
        {"first": [np.nan, "b", "c"], "second": [1, np.nan, np.nan]},
        index=pd.date_range("2024-04-05", periods=3),
    )
    expected = pd.Series(
        [pd.Timestamp("2024-04-07"), pd.Timestamp("2024-04-05")], index=["first", "second"], name="last_date"
    )
    actual = statutils.last_date(df)
    assert_series_equal(actual, expected)


def test_observations_count():
    ser = pd.Series([1, 2, 3], pd.date_range("20130101", periods=3))
    assert statutils.observations_count(ser) == 3

    df = pd.DataFrame(
        {"first": [np.nan, 3, 4.0], "second": [1, np.nan, np.nan]},
        index=["aa", "bb", "cc"],
    )
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
        (
                pd.Series([np.nan, 0, 3, np.nan]),
                50.0,
        ),  # 1 out of 2 is non-zero after ignoring NaN
        ([None, np.nan, None], np.nan),  # all nans
        ([0.0, 0.0, 0.0], 0.0)  # all zeros
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
                    2.5
                ],
        ),
        (2, 8, None, -1, None, [0, 10]),
        (25, 85, None, -1, None, [20, 30, 40, 50, 60, 70, 80, 90]),
        (6, 125, None, None, 25, [0, 25, 50, 75, 100, 125]),
    ],
)
def test_bin_list(min_value, max_value, bins, digits, bin_size, expected):
    actual = statutils.bin_list(
        min_value=min_value,
        max_value=max_value,
        bins=bins,
        bin_size=bin_size,
        digits=digits,
    )
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
                        pd.IntervalIndex.from_tuples([(0.997, 1.667), (1.667, 2.333), (2.333, 3.0)]),
                        ordered=True,
                    ),
                ),
        ),
        (
                [1, 2, 3, 4, 5],
                [0, 2, 4, 6],
                False,
                pd.Series(
                    [2, 2, 1],
                    index=pd.CategoricalIndex(
                        pd.IntervalIndex.from_tuples([(-0.001, 2), (2, 4), (4, 6)]),
                        ordered=True,
                    ),
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
                        pd.IntervalIndex.from_tuples([(0.997, 1.667), (1.667, 2.333), (2.333, 3.0)]),
                        ordered=True,
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
                            index=pd.CategoricalIndex(
                                pd.IntervalIndex.from_tuples([(-0.001, 2), (2, 4)]),
                                ordered=True,
                            ),
                        ),
                        "B": pd.Series(
                            [0.6667, 0.3333],
                            index=pd.CategoricalIndex(
                                pd.IntervalIndex.from_tuples([(-0.001, 2), (2, 4)]),
                                ordered=True,
                            ),
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
                            index=pd.CategoricalIndex(
                                pd.IntervalIndex.from_tuples([(-0.001, 25), (25, 50)]),
                                ordered=True,
                            ),
                        ),
                        "B": pd.Series(
                            [0.25, 0.75],
                            index=pd.CategoricalIndex(
                                pd.IntervalIndex.from_tuples([(-0.001, 25), (25, 50)]),
                                ordered=True,
                            ),
                        ),
                    }
                ),
        )
    ],
)
def test_histogram_bin_size(x, bin_size, normalize, expected):
    actual = statutils.histogram(x, bin_size=bin_size, normalize=normalize)
    assert_frame_equal(actual, expected, atol=0.0001, check_dtype=False)
