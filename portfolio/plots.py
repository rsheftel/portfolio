"""
Plots for use in Streamlit and others
"""

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from plotly import graph_objects as go
from plotly.subplots import make_subplots


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
    fig.update_layout(
        xaxis_title=None,
        yaxis_title=None,
        legend_title=None
    )
    fig = add_quick_date_buttons(fig)
    return fig


def bar_chart(df: pd.DataFrame) -> go.Figure:
    """

    :param df:
    :return:
    """
    fig = ff.create_distplot([df[c].dropna() for c in df.columns], df.columns)
    fig.update_traces(opacity=0.7)

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
        yaxis_title="Probability",
        bargap=0.2,  # Gap between bars for clarity
        bargroupgap=0.1,  # Gap when multiple histograms are grouped
    )
    return fig
