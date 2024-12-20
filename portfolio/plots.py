"""
Plots for use in Streamlit and others
"""

from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from utils import make_iterable


def update_title(fig: go.Figure, title: str) -> go.Figure:
    """
    Update the title of the chart.

    :param fig: go.Figure
    :param title: str of the title
    """
    return fig.update_layout(title={"text": title, "y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"})


def add_quick_date_buttons(fig: go.Figure) -> go.Figure:
    """
    Add quick buttons for 1M, 6M, YTD and 1Y

    :param fig: go.Figure
    :return: go.Figure
    """
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
    )
    return fig


def candlestick_plot(df: pd.DataFrame) -> go.Figure:
    """
    Returns a plotly object of a standard OHLC candlestick chart.

    :param df: symbol DataFrame in standard form of index = datetime, columns = [open, high, low, close, volume]
    :return: Plotly figure object
    """
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # include candlestick with rangeselector
    fig.add_trace(
        go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"]), secondary_y=True
    )
    # include a go.Bar trace for volumes
    fig.add_trace(go.Bar(x=df.index, y=df["volume"]), secondary_y=False)
    fig.layout.yaxis2.showgrid = False

    # Add quick buttons for 1M, 6M, YTD and 1Y
    fig = add_quick_date_buttons(fig)
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # hide weekends
        ],
    )
    return fig


def line_and_bars(df: pd.DataFrame, line_column: str, bar_column: str) -> go.Figure:
    """
    Line and bar graph from a pd.DataFrame. The index should be pd.DatetimeIndex.

    :param df: pd.DataFrame. Index of the datetime, column of the line and bar data
    :param line_column: str. column of the line data
    :param bar_column: str. column of the bar data
    :return: alt.Chart
    """
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df.index, y=df[line_column], mode="lines", showlegend=False), secondary_y=True)
    fig.add_bar(x=df.index, y=df[bar_column], showlegend=False, secondary_y=False)

    fig.update_layout(
        yaxis=dict(title_text=bar_column), yaxis2=dict(title_text=line_column, overlaying="y", side="right")
    )
    fig = add_quick_date_buttons(fig)
    return fig


def multi_line(df: pd.DataFrame) -> go.Figure:
    """
    Multi-line graph from a pd.DataFrame. The index should be pd.DatetimeIndex.

    :param df: pd.DataFrame. Index of the datetime, column of the line and bar data
    :return: alt.Chart
    """
    # Create figure with secondary y-axis
    fig = px.line(df)
    fig.update_layout(xaxis_title=None, yaxis_title=None, legend_title=None)
    fig = add_quick_date_buttons(fig)
    return fig


def two_axis_line(df: pd.DataFrame, left_columns: str | list, right_columns: str | list) -> go.Figure:
    """
    Multi-line graph from a pd.DataFrame where lines are plotted on the left or right axis

    :param df: pd.DataFrame
    :param left_columns: columns of the df to plot on the left axis
    :param right_columns: columns of the df to plot on the right axis
    :return: alt.Chart
    """
    left_columns = make_iterable(left_columns)
    right_columns = make_iterable(right_columns)
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add left axis lines
    for col in left_columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df[col], name=col, mode="lines"),
            secondary_y=False,
        )

    # Add right axis
    for col in right_columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df[col], name=col, mode="lines"),
            secondary_y=True,
        )

    return fig


def histogram(df: pd.DataFrame, number_of_bins: int = 20) -> go.Figure:
    """
    Histogram chart

    :param df: pd.DataFrame
    :param number_of_bins: number of bins
    :return: go.Figure
    """
    # old depricated method
    # fig = ff.create_distplot([df[c].dropna() for c in df.columns], df.columns)
    # fig.update_traces(opacity=0.7)
    fig = px.histogram(df, marginal="rug", histnorm="probability", nbins=number_of_bins, opacity=0.7, barmode="overlay")

    # Add a vertical line at x=0
    fig.add_shape(
        # Line shape
        type="line",
        # Vertical line at x=0
        x0=0,
        x1=0,
        # From the bottom to the top of the plot
        y0=0,
        y1=1,
        yref="paper",  # Use 'paper' reference for y-coordinates to span the entire plot
        line=dict(
            color="Black",
            width=2,
            dash="dot",
        ),
    )
    fig.update_layout(
        yaxis_title="probability",
        bargap=0.2,  # Gap between bars for clarity
        bargroupgap=0.1,  # Gap when multiple histograms are grouped
    )
    return fig


def equity_and_pnl(ser: pd.Series) -> go.Figure:
    """
    Returns a combined equity & pnl chart

    :param equity: pd.DataFrame with one column of equity data
    :return: go.Figure
    """
    df = pd.DataFrame(ser)
    df.columns = ["pnl"]
    df["equity"] = df["pnl"].cumsum()
    return line_and_bars(df, "equity", "pnl")


def equity_and_drawdown(
        equity: pd.DataFrame, underwater_equity: pd.DataFrame, drawdown_details: pd.DataFrame,
        ndrawdown_highlights: int = 5
) -> go.Figure:
    """
    Returns a combined equity & drawdown chart

    :param equity: pd. DataFrame with one column of equity data
    :param underwater_equity: pd.DataFrame with one column of underwater equity data
    :param drawdown_details: pd.DataFrame of standard drawdown details
    :param ndrawdown_highlights: number of worst drawdown to highlight
    :return: go.Figure
    """
    # combined equity and drawdown
    equity_chart = px.line(equity)
    drawdown_plot = px.line(underwater_equity)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.75, 0.25])
    # add each trace (or traces) to its specific subplot
    for i in equity_chart.data:
        fig.add_trace(i, row=1, col=1)
    for i in drawdown_plot.data:
        fig.add_trace(i, row=2, col=1)
    drawdown_details = drawdown_details.sort_values("drawdown")
    for x in range(ndrawdown_highlights):
        fig.add_vrect(
            x0=drawdown_details.iloc[x]["start"],
            x1=drawdown_details.iloc[x]["end"],
            fillcolor="red",
            opacity=0.2,
            layer="below",
            line_width=0,
        )
    fig.update_layout(xaxis_title=None, yaxis_title=None, legend_title=None, showlegend=False)
    return fig


def correlation(correlation_df: pd.DataFrame, one_sided: bool = True) -> go.Figure:
    """
    Returns a correlation plot

    :param correlation_df: pd.DataFrame as a correlation matrix, the output from df.corr()
    :param one_sided: if True then only the upper triangle is plotted, if False then the whole matrix is plotted
    :return: go.Figure
    """

    # Calculate the correlation matrix
    correlation_df = correlation_df.copy()

    if one_sided:
        # Mask the upper triangle
        mask = np.triu(np.ones(correlation_df.shape), k=1)
        correlation_df = correlation_df.where(mask == 0)

    fig = px.imshow(
        correlation_df,
        zmin=-1.0,
        zmax=1.0,
        color_continuous_scale=[(0, "red"), (0.5, "white"), (1, "green")],
        origin="upper",
        text_auto=".2f",
    )

    # Format the colorbar with two decimal places
    fig.update_coloraxes(colorbar_tickformat=".2f")

    # Customize hover text format to 2 decimal places
    fig.update_traces(hovertemplate="%{x}<br>%{y}<br>correlation: %{z:.2f}<extra></extra>")

    # if two-sided, but the x-axis labels on top
    fig.update_xaxes(side="top")

    return fig


def add_period_column(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Add a column to the DataFrame that is the period. For year the four digit YYYY year will be added. For month
    the three letter abbreviation will be added.

    :param df: pd.DataFrame
    :param period: period string
    :return: pd.DataFramge
    """
    df = df.copy()

    match period:
        case None:
            pass
        case _ if period.upper() in ["MONTH", "MONTHLY", "M", "ME"]:
            df["month"] = df.index.strftime("%b")
        case _ if period.upper() in ["YEAR", "YEARLY", "Y", "YE"]:
            df["year"] = df.index.strftime("%Y")
        case _:
            raise AttributeError(f"Period {period} not recognized")
    return df


def scatter(
        df: pd.DataFrame, x_column: str, y_column: str, period: Literal["month", "year"] | None = None
) -> go.Figure:
    """
    Returns a scatter plot. If period is supplied then the colors of the scatter plot will differ based on that period

    :param df: pd.DataFrame
    :param x_column: column to use as the x-axis
    :param y_column: column to use as the y-axis
    :param period: ['month', 'year', None]
    :return: go.Figure
    """
    df = add_period_column(df, period)
    fig = px.scatter(df, x=x_column, y=y_column, color=period)
    return fig


def scatter_matrix(df: pd.DataFrame, period: str = None) -> go.Figure:
    """
    Returns a scatter matrix. If period is supplied then the colors of the scatter plot will differ based on that period

    :param df: pd.DataFrame
    :param period: ['month', 'year', None]
    :return: go.Figure
    """
    columns = df.columns.tolist()
    df = add_period_column(df, period)
    fig = px.scatter_matrix(df, dimensions=columns, color=period)
    fig.update_traces(diagonal_visible=False)
    return fig
