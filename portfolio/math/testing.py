import numpy as np
import numpy.random as nprandom


def mock_time_series(
    length: int,
    volatility: float,
    annualization=252,
    auto_regress=0,
    drift=0,
    base=100,
    multiplier=1,
    nans=0,
    seed=0,
    round_decimals=False,
):
    """
    Generate mock time series from a random brownian motion with a given seed value for the random numbers.

    :param length: length of the resulting array
    :param volatility: annual volatility of the process in decimal (ie: 0.10 for 10% annual vol)
    :param annualization: the number of units in a year for scaling the annual volatility (ie: 252 for daily data)
    :param auto_regress: the auto regression coefficient. The change in step i will equal the prior step * this
        auto_regress parameter + noise
    :param drift: drift coefficient. The changes will be adjusted to this coefficient * the index number
    :param base: base price of the levels series
    :param multiplier: multiply the entire changes by this. For example to create a down market mirror of an up market
        set this to -1
    :param nans: if > 0 then nans number of elements will be randomly replaced with nan
    :param seed: seed value for the numpy random number generator
    :param round_decimals: if an int then all values will be rounded to this number of decimal plaes
    :return: numpy array of prices
    """

    # seed the random numbers
    nprandom.seed(seed)

    # generate an array of length of normally distributed numbers for the changes
    changes = nprandom.normal(0, volatility / annualization**0.5, length - 1)
    changes = np.insert(changes, 0, 0.0)  # insert a zero change as the first element
    changes = changes * base * multiplier  # scale by the base and multiplier

    if auto_regress:
        for i in range(1, length):
            changes[i] = auto_regress * changes[i - 1] + changes[i]

    if drift:
        for i in range(1, length):
            changes[i] = changes[i] + drift * i

    if nans:  # add a certain number of randomly placed None
        locations = nprandom.randint(0, length - 1, nans)
        changes[locations] = None

    if round_decimals:
        changes = changes.round(round_decimals)

    return base + changes
