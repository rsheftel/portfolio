import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pandera as pa
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_datetime64_dtype,
    is_datetime64_ns_dtype,
    is_datetime64tz_dtype,
)
from pandas.io.formats.style import Styler


def make_iterable(variable, as_list: bool = False, none_returns_none: bool = False) -> Iterable | None:
    """
    Checks if the variable is an iterable, but not a string. If SO then it returns the variable, if not then it
    makes the variable a single element list. Optionally can convert the variable to a list regardless.
    :param variable: the variable to convert
    :param as_list: will force a conversion to a list
    :param none_returns_none: if the variable is None then will return None
    :return: either the variable if it is an iterable, or a list (variable)
    """
    if none_returns_none and variable is None:
        return None
    if isinstance(variable, Iterable) and not isinstance(variable, (str, pd.DataFrame, pd.Series)):
        return list(variable) if as_list else variable
    else:
        return [variable]


def reindex_superset(
        source: pd.Series | pd.DataFrame,
        target: pd.Index | pd.Series | pd.DataFrame,
        fill_value: Any = None,
        fill_method: str | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Given a source series or DataFrame will reindex to the superset of the index values in the source and the
    index values in the target series, DataFrame or DataTimeIndex.
    The resulting index of the source will be the superset union of the index values in source and target.
    The fill method options are either None to ignore and do not fill, or ffill or bfill or any other valid option for
    pandas.fillna()

    :param source: Series or DataFrame to align and match with the with Target
    :param target: index to match in the return Series or DataFrame
    :param fill_value: value to use to fill new entries
    :param fill_method: None to ignore and leave missing values np.nan, ffill or bill
    :return: pd.Series or pd.DataFrame
    """
    source = source.copy()
    if isinstance(target, (pd.Series, pd.DataFrame)):
        target = target.index
    all_index = source.index.union(target)
    return source.reindex(all_index, fill_value=fill_value, method=fill_method)


def fillna(array, value=None, method="ffill", limit=None):
    """
    Fills the na in a numpy array using the same logic as pandas

    :param array: numpy array
    :param value: as described here http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.fillna.html
    :param method: as described here http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.fillna.html
    :param limit: as described here http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.fillna.html
    :return: numpy array
    """

    series = pd.Series(array)
    if value is not None:
        method = None
    if value is not None:
        return series.fillna(value=value, limit=limit).to_numpy()
    elif method == "ffill":
        return series.ffill(limit=limit).to_numpy()
    elif method == "bfill":
        return series.bfill(limit=limit).to_numpy()
    else:
        raise RuntimeError("unable to find a fill method for the combination of function parameters")


def match_index(
        source: pd.Series | pd.DataFrame,
        target: pd.Series | pd.DataFrame | pd.DatetimeIndex,
        fill_method: str | None = "ffill",
) -> pd.Series | pd.DataFrame:
    """
    Given a source series or DataFrame will align and match the index in the target series, DataFrame or DataTimeIndex.
    The resulting index of the source will match the index in the target. Rows in the source index not in the target
    will be dropped, and rows in the target not in the source will be filled with the fill_method. The fill method
    options are either None to ignore and do not fill, or ffill or bfill or any other valid option for pandas.fillna()

    :param source: Series or DataFrame to align and match with the with Target
    :param target: index to match in the return Series or DataFrame
    :param fill_method: None to ignore and leave missing values np.nan, ffill or bill
    :return: pd.Series or pd.DataFrame
    """

    source = source.copy().sort_index()
    target = target.copy().sort_index()

    res, _ = source.align(target)  # fill out all possible dates in source and target
    if fill_method == "ffill":
        res = res.ffill()
    elif fill_method == "bfill":
        res = res.bfill()
    res, _ = res.align(target, join="right")  # align to the dates only in target
    if not res.index.freq:
        res.index.freq = res.index.inferred_freq
    return res


def column_types(
        df: pd.DataFrame,
        small_number_cutoff: float | int = 1000,
        date_columns: list = None,
        datetime_columns: list = None,
        boolean_columns: list = None,
        large_number_columns: list = None,
        small_number_columns: list = None,
        integer_columns: list = None,
        percent_columns: list = None,
) -> dict:
    """
    Returns a dict where the key is the type of the column and the value is a list of the columns of that type. This is
    more expansive than the dtypes as it will differentiate between date and datetime, and has different versions of
    numeric. This is used for formatting in Streamlit or pandas Styler.

    First the types inferred from the DataFrame, then based on the explicit parameters supplied.
    The type of each column will be inferred from the values in the DataFrame, to override the inferred format use
    the * columns parameters.

    :param df: pd. DataFrame
    :param small_number_cutoff: used to determine if a numeric column is small number or large number. If the max of
        any element in the column is less than the small_number_cutoff, then the column is small_number, otherwise
        it is large number
    :param date_columns: list of columns to force to date type
    :param datetime_columns: list of columns to force to datetime type
    :param boolean_columns: list of columns to force to boolean type
    :param large_number_columns: list of columns to force to large numbers type
    :param small_number_columns: list of columns to force to small_numbers type
    :param integer_columns: list of columns to force to integer type
    :param percent_columns: list of columns to force to percent type. And column where the numbers are all between
        -1 and 1 will be considered a percent column
    :return: dict
    """
    df = df.copy().infer_objects()
    res_df = pd.DataFrame(columns=["type"])
    # get the inferred data types
    for column in df.columns:
        if is_datetime64_any_dtype(df[column]):
            if np.all(df[column].dt.time == datetime.time(0, 0)):
                res_df.loc[column, "type"] = "date"
            else:
                res_df.loc[column, "type"] = "datetime"
        elif is_bool_dtype(df[column]):
            res_df.loc[column, "type"] = "boolean"
        elif is_integer_dtype(df[column]):
            res_df.loc[column, "type"] = "integer"
        elif is_numeric_dtype(df[column]):
            if (df[column].max() <= 1.0) and (df[column].min() >= -1.0):
                res_df.loc[column, "type"] = "percent"
            elif (df[column].max() < small_number_cutoff) and (df[column].min() > -small_number_cutoff):
                res_df.loc[column, "type"] = "small_number"
            else:
                res_df.loc[column, "type"] = "large_number"
        else:
            res_df.loc[column, "type"] = "other"

    # overwrite with the parameter defined columns
    if date_columns:
        for column in date_columns:
            res_df.loc[column, "type"] = "date"
    if datetime_columns:
        for column in datetime_columns:
            res_df.loc[column, "type"] = "datetime"
    if boolean_columns:
        for column in boolean_columns:
            res_df.loc[column, "type"] = "boolean"
    if large_number_columns:
        for column in large_number_columns:
            res_df.loc[column, "type"] = "large_number"
    if small_number_columns:
        for column in small_number_columns:
            res_df.loc[column, "type"] = "small_number"
    if integer_columns:
        for column in integer_columns:
            res_df.loc[column, "type"] = "integer"
    if percent_columns:
        for column in percent_columns:
            res_df.loc[column, "type"] = "percent"
    # turn into a dict
    res = {
        k: res_df.index[res_df["type"] == k].to_list()
        for k in ["date", "datetime", "boolean", "large_number", "small_number", "integer", "percent", "other"]
    }
    return res


def week_of_month(date: pd.Timestamp | datetime.datetime) -> int:
    """
    Return the week of the month for the specified date, where the week starts on Monday.
    The first week of the month is defined as the week that contains the first weekday.

    :param date: pd.Timestamp or datetime
    :return: int week of the month
    """
    first_day = date.replace(day=1)
    first_weekday = first_day.weekday()

    # [5, 6] are [Sat, Sun]
    offset = 1 if first_weekday in [5, 6] else 2

    # The -6 makes the week start on a Saturday
    return (date.day + first_day.weekday() - 6) // 7 + offset


def average_of_cols(df, row_name="average"):
    """
    Appends an average row for all numeric columns to the DataFrame

    :param df: DataFrame
    :param row_name: name for the new total row
    :return: DataFrame
    """
    return pd.concat([df, pd.DataFrame(df.mean(numeric_only=True).rename(row_name)).transpose()])


def make_series(x: pd.Series | pd.DataFrame) -> pd.Series:
    """
    Converts the input to a series. If x is a series then will return x unchanged. If x is a pd.DataFrame with one
    column then will return that column as a series. If it is anything else will raise an error.

    :param x: pd.Series or pd.DataFrame
    :return: pd.Series
    """
    if isinstance(x, pd.DataFrame):
        if len(x.columns) > 1:
            raise AttributeError(f"Dataframe has more than one column: {len(x.columns)} columns")
        return x.iloc[:, 0]
    elif isinstance(x, pd.Series):
        return x
    else:
        raise AttributeError(f"Parameter df is not a pd.DataFrame or pd.Series: {x}")


def _prep_df(df: pd.DataFrame | Styler) -> Styler:
    """
    Prepares the df for application of a style.

    :param df: pd.DataFrame or pd.Styler
    :return: pd.Styler
    """
    if isinstance(df, pd.DataFrame):
        df = df.style
    return df


def _make_slice(columns: str | list = None, rows: str | list = None) -> pd.IndexSlice:
    """
    Creates a pd.IndexSlice from the columns and rows

    :param columns: column string or list
    :param rows: index string or list
    :return: pd.IndexSlice
    """
    columns = make_iterable(columns, none_returns_none=True)
    rows = make_iterable(rows, none_returns_none=True)
    if columns and rows:
        return pd.IndexSlice[columns, rows]
    elif columns:
        return pd.IndexSlice[:, columns]
    elif rows:
        return pd.IndexSlice[rows, :]
    else:
        return pd.IndexSlice[[]]


def date_format(df: pd.DataFrame | Styler, columns: str | list = None, rows: str | list = None) -> Styler:
    """
    Make the columns and/or rows format YYYY-MM-DD

    :param df: pd.DataFrame or pd.Styler
    :param columns: column names to make in the date format
    :param rows: index names to make in the date format
    :return: pd.Styler
    """
    df = _prep_df(df)
    index_slice = _make_slice(columns, rows)
    return df.format("{:%Y-%m-%d}", subset=index_slice)


def datetime_format(df: pd.DataFrame | Styler, columns: str | list = None, rows: str | list = None) -> Styler:
    """
    Make the columns and/or rows format YYYY-MM-DD HH:MM

    :param df: pd.DataFrame or Styler
    :param columns: column names to make in the datetime format
    :param rows: index names to make in the date format
    :return: pd.Styler
    """
    df = _prep_df(df)
    index_slice = _make_slice(columns, rows)
    return df.format("{:%Y-%m-%d %H:%M}", subset=index_slice)


def number_format(
        df: pd.DataFrame | Styler, columns: str | list = None, rows: str | list = None, precision: int = 2
) -> Styler:
    """
    Make the columns and/or rows number format with a comma in the thousands and a given number of digits of precision

    :param df: pd. DataFrame or Styler
    :param columns: column names to make in the number format
    :param rows: index names to make in the date format
    :param precision: number of decimal places of precision
    :return: pd.Styler
    """
    df = _prep_df(df)
    index_slice = _make_slice(columns, rows)
    return df.format(thousands=",", precision=precision, subset=index_slice)


def percent_format(
        df: pd.DataFrame | Styler, columns: str | list = None, rows: str | list = None, precision: int = 2
) -> Styler:
    """
    Make the columns and/or rows percent format with a given number of digits of precision

    :param df: pd. DataFrame or Styler
    :param columns: column names to make in the number format
    :param rows: index names to make in the date format
    :param precision: number of decimal places of precision
    :return: pd.Styler
    """
    df = _prep_df(df)
    index_slice = _make_slice(columns, rows)
    return df.format(f"{{:.{precision}%}}", subset=index_slice)


def read_csv_time_series(
        filepath_or_buffer: str | Path | StringIO,
        datetime_col: int | str = 0,
        parser=None,
        all_numeric: bool = True,
        **kwargs,
) -> pd.DataFrame:
    """
    Reads time series csv file into pd.DataFrame from a given file path or buffer and validates the schema. The format
    of the file is the date or datetime in the first column and the time series data in the remaining columns.
    Implements the pandas read_csv with a datetime parser now that that functionality is deprecated from the pandas
    pacakge in v2.0.0. This will read a time series csv file and use the parser function to convert the dateime_col
    to datetime and make that the index

    :param filepath_or_buffer: File path, file-like object, or buffer
    :param datetime_col: column name for the datetimes for column number as integer
    :param parser: (optional) parser function to apply to the datetime string values to convert to datetimes
    :param all_numeric: if True then will test to confirm all values are numeric
    :param kwargs: Additional keyword arguments passed to `pd.read_csv`.
    :return: A pandas DataFrame containing the processed and validated data
    """
    if parser:
        df = pd.read_csv(filepath_or_buffer, **kwargs)
        datetime_col = df.columns[datetime_col] if isinstance(datetime_col, int) else datetime_col
        df[datetime_col] = parser(df[datetime_col].values)
        df = df.set_index(datetime_col)
    else:
        df = pd.read_csv(filepath_or_buffer, index_col=[datetime_col], parse_dates=[datetime_col], **kwargs)
    if all_numeric:
        df = df.apply(pd.to_numeric, errors="coerce")
        is_number = pa.Check(is_numeric_dtype, name="is_number")
        is_datetime = pa.Check(
            lambda x: is_datetime64_dtype(x) | is_datetime64_ns_dtype(x) | is_datetime64tz_dtype(x), name="is_datetime"
        )
        schema = pa.DataFrameSchema(
            {col: pa.Column(checks=is_number, nullable=True) for col in df.columns},
            index=pa.Index(checks=is_datetime),
            unique_column_names=True,
        )
        schema(df)
    return df


def asof_next(x: pd.DataFrame | pd.Series, index: Any) -> Any | pd.Series:
    """
    Similar to the pandas "asof()" method but instead of returning the prior value if there is not an exact match
    of the index, it returns the next value.

    :param x: pd.DataFrame or pd.Series
    :param index: index value to match
    :return: pd.Series if pd.DataFrame provided as x, otherwise returns scalar
    """
    # test that it is a sorted
    if not x.index.is_monotonic_increasing:
        raise ValueError("asof_next requires a sorted index")

    # backfill the data so a value of NaN for existing index value pulls from the next
    x = x.copy().bfill()

    # Find the exact match if exists
    if index in x.index:
        return x.loc[index]

    # if it is beyond the end of the index
    if index > x.index[-1]:
        if isinstance(x, pd.Series):
            return np.nan
        else:
            return pd.Series(np.nan, index=x.columns)

    # If no exact match, find the next higher index
    next_index = x.index[np.searchsorted(x.index, index, side="right")]

    # Return the value at the next higher index
    res = x.loc[next_index]
    if isinstance(res, pd.Series):
        res.name = index
    return res


def asof_prior(x: pd.DataFrame | pd.Series, index: Any) -> Any | pd.Series:
    """
    Wrapper to the pandas "asof()" method but instead of returning the prior value only if all values in the row are
    NoN in the pandas asof(), this will treat each column separately and return the first non-NaN value, which means
    that the values of a return Series for a DataFrame may be from different rows.

    :param x: pd.DataFrame or pd.Series
    :param index: index value to match
    :return: pd.Series if pd.DataFrame provided as x, otherwise returns scalar
    """
    return x.ffill().asof(index)


def index_from_df(df: pd.DataFrame) -> pd.Index | pd.MultiIndex:
    """
    Returns an Index or MultiIndex object from the dataframe provided. Each column is the index or multi-index
    values and the row index labels are the index or multi-index names.
    :param df: dataframe
    :return: Index or MultiIndex
    """
    if len(df) == 0:
        return pd.Index([])
    if len(df) == 1:
        return pd.Index(df.iloc[0, :])
    return pd.MultiIndex.from_frame(df.T)


def df_from_index(index: pd.Index | pd.MultiIndex) -> pd.DataFrame:
    """
    Returns a DataFrame representation of the contents of an Index or Multi-Index - Each column is the index or
    multi-index values, and the row index labels are the index or multi-index names.

    :param index: Index or MultiIndex
    :return: pd.DataFrame
    """
    if len(index) == 0:
        return pd.DataFrame()
    return index.to_frame().reset_index(drop=True).T


def add_index_levels(index: pd.Index | pd.MultiIndex, new_index_df: pd.DataFrame) -> pd.MultiIndex:
    """
    Add levels to a MultiIndex. The new_index_df is in the same format as the index_from_df() function parameter

    :param index: original index to have the new values appended
    :param new_index_df: DataFrame of new column names (index) and level values (cell) for each column
    :return: pd.Index | pd.MultiIndex
    """
    orig = df_from_index(index)
    # line up the column names so that concat works
    new_index_df.index = orig.columns
    return index_from_df(pd.concat([orig, new_index_df.T]))


def prepend_index_level(df: pd.DataFrame, name: str, value: str) -> pd.DataFrame:
    """
    Adds an index level to MultiIndex columns only and prepends at the top level. The value will be repeated for
    all the columns at that level

    :param df: pd. DataFrame
    :param name: name of the new index level
    :param value: value of the index level
    :return: pd. DataFrame
    """
    df.columns = add_index_levels(df.columns, pd.DataFrame({name: [value] * len(df.columns)}))
    df.columns = df.columns.reorder_levels([name] + df.columns.names[:-1])
    return df
