"""
Unit tests of the testing module
"""

import numpy as np
from pytest import approx

from portfolio import testing


def assert_results(values, min_value, max_value, sum_value):
    assert min(values) == approx(min_value)
    assert max(values) == approx(max_value)
    assert sum(values) == approx(sum_value)


def test_mock_price_series():
    # just noise no rounding
    res = testing.mock_time_series(100, 0.10)
    assert len(res) == 100
    assert min(res) == approx(98.391767583, 0.00000001)
    assert max(res) == approx(101.429811017, 0.00000001)
    assert sum(res) == approx(10003.514321349, 0.00000001)

    # noise with rounding
    res = testing.mock_time_series(100, 0.10, round_decimals=4)
    assert_results(res, 98.3918, 101.4298, 10003.5142)

    # new seed
    res = testing.mock_time_series(100, 0.10, round_decimals=4, seed=100)
    assert_results(res, 98.8402, 101.2817, 9995.0048)

    # -1 multiplier
    res = testing.mock_time_series(100, 0.10, round_decimals=4, multiplier=-1)
    assert_results(res, 98.5702, 101.6082, 9996.4858)

    # mean reversion
    res = testing.mock_time_series(100, 0.10, auto_regress=0.8, round_decimals=4)
    assert_results(res, 97.7341, 103.2846, 10010.4306)

    # trend
    res = testing.mock_time_series(100, 0.10, auto_regress=1.0, round_decimals=4)
    assert_results(res, 97.6199, 108.7050, 10344.1365)

    # drift
    res = testing.mock_time_series(100, 0.10, drift=-0.1, round_decimals=4)
    assert_results(res, 90.1799, 101.0116, 9508.5142)

    # mean reversion with drift
    res = testing.mock_time_series(100, 0.10, auto_regress=0.5, drift=0.1, round_decimals=4)
    assert_results(res, 100.0, 111.1221, 10501.2881)

    # nans inserted
    res = testing.mock_time_series(100, 0.10, round_decimals=4, nans=5)
    assert sum(np.isnan(res)) == 5
    assert np.nansum(res) == approx(9501.7288)
