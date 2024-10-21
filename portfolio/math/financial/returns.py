"""
Financial math functions like Sharpe based on percentage return time series
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from portfolio.math.base import dispatch_calc, dropna
from portfolio.math.financial.common import (
    sharpe,
    sharpe_exp_weighted,
    sortino,
    sortino_exp_weighted,
    conditional_sortino,
    conditional_sortino_exp_weighted,
)


def __keep_imports():
    """
    this is a dummy function the sole purpose is to retain the imports so they are not removed by the Pycharm Optimize
    imports
    """
    assert sharpe
    assert sharpe_exp_weighted
    assert sortino
    assert sortino_exp_weighted
    assert conditional_sortino
    assert conditional_sortino_exp_weighted


def total_return(x: list | NDArray | pd.DataFrame | pd.Series) -> float | pd.Series:
    """
    Given a time series of periodic returns, calculate the total return for the entire series by compounding the
    periodic returns

    :param x: any of a list, numpy array, pd.Series or pd. DataFrame
    :return: float if x is a list, array or a pd.Series. A pd.Series if x is a pd.DataFrame
    """

    def _calc(x):
        # Calculate the compounded growth factor
        # Compute the cumulative product to get the compounded growth over time
        # The last element of compounded_growth will give us the total compounded return
        x = dropna(x)
        return np.cumprod(1 + x)[-1] - 1

    return dispatch_calc(x, _calc, name="total_return")
