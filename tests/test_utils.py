import pandas as pd
from portfolio import utils

import pytest
from pandas.testing import assert_series_equal
import numpy as np
from io import StringIO
from pandera.errors import SchemaError
from pandas.testing import assert_frame_equal


def test_asof_next():
    # exact match
    series = pd.Series([10, 20, 30], index=[1, 2, 3])
    result = utils.asof_next(series, 2)
    assert result == 20

    # test_asof_next_series_next_value():
    series = pd.Series([10, 20, 30], index=[1, 2, 3])
    result = utils.asof_next(series, 2.5)
    assert result == 30

    # with nan:
    series = pd.Series([10, np.nan, 30], index=[1, 2, 3])
    result = utils.asof_next(series, 2)
    assert result == 30

    # test_asof_next_series_beyond_end():
    series = pd.Series([10, 20, 30], index=[1, 2, 3])
    result = utils.asof_next(series, 4)
    assert pd.isna(result)

    # test_asof_next_series_unsorted_index():
    series = pd.Series([10, 20, 30], index=[3, 1, 2])
    with pytest.raises(ValueError, match="asof_next requires a sorted index"):
        utils.asof_next(series, 2)

    # test_asof_next_frame_exact_match():
    frame = pd.DataFrame({"A": [10, 20, 30], "B": [11, 21, 31]}, index=[1, 2, 3])
    result = utils.asof_next(frame, 2)
    assert_series_equal(result, pd.Series({"A": 20, "B": 21}, name=2))

    # test_asof_next_frame_next_value():
    frame = pd.DataFrame({"A": [10, 20, 30], "B": [11, 21, 31]}, index=[1, 2, 3])
    result = utils.asof_next(frame, 2.5)
    assert_series_equal(result, pd.Series({"A": 30, "B": 31}, name=2.5))

    # with nans:
    frame = pd.DataFrame({"A": [10, np.nan, 30.0], "B": [11, 21, 31]}, index=[1, 2, 3])
    result = utils.asof_next(frame, 1.5)
    assert_series_equal(result, pd.Series({"A": 30.0, "B": 21}, name=1.5))

    # test_asof_next_frame_beyond_end():
    frame = pd.DataFrame({"A": [10, 20, 30], "B": [11, 21, 31]}, index=[1, 2, 3])
    result = utils.asof_next(frame, 4)
    expected = pd.Series({"A": np.nan, "B": np.nan})
    assert_series_equal(result, expected)

    # test_asof_next_frame_unsorted_index():
    frame = pd.DataFrame({"A": [10, 20, 30]}, index=[3, 1, 2])
    with pytest.raises(ValueError, match="asof_next requires a sorted index"):
        utils.asof_next(frame, 2)


def test_asof_prior():
    # exact match
    series = pd.Series([10, 20, 30], index=[1, 2, 3])
    result = utils.asof_prior(series, 2)
    assert result == 20

    series = pd.Series([10, 20, 30], index=[1, 2, 3])
    result = utils.asof_prior(series, 2.5)
    assert result == 20

    series = pd.Series([10, np.nan, 30], index=[1, 2, 3])
    result = utils.asof_prior(series, 2)
    assert result == 10

    series = pd.Series([10, 20, 30], index=[1, 2, 3])
    result = utils.asof_prior(series, 0)
    assert pd.isna(result)

    # test_asof_prior_series_unsorted_index():
    series = pd.Series([10, 20, 30], index=[3, 1, 2])
    with pytest.raises(ValueError, match="asof requires a sorted index"):
        utils.asof_prior(series, 2)

    # test_asof_prior_frame_exact_match():
    frame = pd.DataFrame({"A": [10, 20, 30], "B": [11, 21, 31]}, index=[1, 2, 3])
    result = utils.asof_prior(frame, 2)
    assert_series_equal(result, pd.Series({"A": 20, "B": 21}, name=2))

    frame = pd.DataFrame({"A": [10, 20, 30], "B": [11, 21, 31]}, index=[1, 2, 3])
    result = utils.asof_prior(frame, 2.5)
    assert_series_equal(result, pd.Series({"A": 20, "B": 21}, name=2.5))

    # with nans:
    frame = pd.DataFrame({"A": [10, np.nan, 30.0], "B": [11, 21, 31]}, index=[1, 2, 3])
    result = utils.asof_prior(frame, 2.5)
    assert_series_equal(result, pd.Series({"A": 10.0, "B": 21}, name=2.5))

    frame = pd.DataFrame({"A": [10, np.nan, 30.0], "B": [11, np.nan, 31]}, index=[1, 2, 3])
    result = utils.asof_prior(frame, 2.5)
    assert_series_equal(result, pd.Series({"A": 10.0, "B": 11.0}, name=2.5))

    frame = pd.DataFrame({"A": [10, 20, 30], "B": [11, 21, 31]}, index=[1, 2, 3])
    result = utils.asof_prior(frame, 0)
    expected = pd.Series({"A": np.nan, "B": np.nan}, name=0)
    assert_series_equal(result, expected)

    # test_asof_prior_frame_unsorted_index():
    frame = pd.DataFrame({"A": [10, 20, 30]}, index=[3, 1, 2])
    with pytest.raises(ValueError, match="asof requires a sorted index"):
        utils.asof_prior(frame, 2)


def test_read_csv_time_series_numeric():
    # float values
    buffer = StringIO("datetime,value1,value2\n2023-01-01,1.0,2.0\n2023-01-02,3.0,4.0")
    expected = pd.DataFrame(
        {"value1": [1.0, 3.0], "value2": [2.0, 4.0]},
        index=pd.DatetimeIndex(["2023-01-01", "2023-01-02"], name="datetime"),
    )
    actual = utils.read_csv_time_series(buffer)
    assert_frame_equal(actual, expected)

    # int values
    buffer = StringIO("datetime,value1,value2\n2023-01-01,1,2\n2023-01-02,3,4")
    expected = pd.DataFrame(
        {"value1": [1, 3], "value2": [2, 4]},
        index=pd.DatetimeIndex(["2023-01-01", "2023-01-02"], name="datetime"),
    )
    actual = utils.read_csv_time_series(buffer)
    assert_frame_equal(actual, expected)

    # if all_numeric = False can be any values
    buffer = StringIO("datetime,value1,value2\n2023-01-01,1.0,2.0\n2023-01-02,3,abc")
    expected = pd.DataFrame(
        {"value1": [1.0, 3.0], "value2": ["2.0", "abc"]},
        index=pd.DatetimeIndex(["2023-01-01", "2023-01-02"], name="datetime"),
    )
    actual = utils.read_csv_time_series(buffer, all_numeric=False)
    assert_frame_equal(actual, expected)

    # bad value in data converts to nan if all_numeric = true
    buffer = StringIO("datetime,value1,value2\n2023-01-01,1.0,2.0\n2023-01-02,3,abc")
    expected = pd.DataFrame(
        {"value1": [1.0, 3.0], "value2": [2.0, np.nan]},
        index=pd.DatetimeIndex(["2023-01-01", "2023-01-02"], name="datetime"),
    )
    actual = utils.read_csv_time_series(buffer, all_numeric=True)
    assert_frame_equal(actual, expected)

    # bad value in date
    buffer = StringIO("datetime,value1,value2\n2023-01-01,1.0,2.0\n1.0,3.0,4.0")
    with pytest.raises(SchemaError):
        utils.read_csv_time_series(buffer)
