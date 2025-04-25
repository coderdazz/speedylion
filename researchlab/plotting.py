import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from typing import Optional, Union

from core.data_funcs import filter_date_range, pd_type

# Constants
ALPHA_LINE = 0.8
ALPHA_FILL = 0.3
AXES_TYPE = Optional[plt.Axes]


def matplotlib_setting():
    """
    Set global rcParams for matplotlib to produce nice and large publication-quality figures.
    """
    plt.rcParams['figure.figsize'] = (16, 8)
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'cmr10'
    plt.rcParams['axes.formatter.use_mathtext'] = True
    # plt.rcParams['text.usetex'] = True
    # plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    return


def fig_to_pdf(pdf, close=False):
    """
    Save the current figure to a specified path. Automatically creates the folder if it doesn't exist.

    Parameters
    ----------
    filepath : str
        The path where the figure should be saved.
    close : bool, optional (default=False)
        Whether to close the figure after saving.
    """
    # check_dir_exist(filepath)
    pdf.savefig()
    if close:
        plt.close()
    return


def check_axes(ax: AXES_TYPE = None, nrows=1, ncols=1, figsize_single=(24, 12), **kwargs) -> Union[plt.Axes, np.ndarray]:
    """
    Create a new axes if `ax` is None; otherwise return the existing axes.

    Parameters
    ----------
    ax : plt.Axes, optional
        An existing matplotlib Axes object. If None, a new one is created.
    nrows : int, optional (default=1)
        Number of rows for the subplots.
    ncols : int, optional (default=1)
        Number of columns for the subplots.
    figsize_single : tuple, optional (default=(24, 12))
        The size of a single subplot.

    Returns
    -------
    plt.Axes or np.ndarray
        The axes object(s) for plotting.
    """
    if ax is None:
        w, h = figsize_single
        _, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * w, nrows * h), **kwargs)
    return ax


def convert_yaxis_to_percent(ax: plt.Axes) -> None:
    """
    Convert the ticks on the y-axis to percent without decimals (e.g., 4.0 becomes 400%).
    """
    def to_percent(x, position):
        pos_flag = x >= 0
        string = f"{abs(x) * 100:.0f}\\%"
        return string if pos_flag else "$-$" + string

    ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
    return


def plot_cumret(
    ret_df: pd_type,
    start_date: str = None,
    end_date: str = None,
    ax: AXES_TYPE = None,
    ylabel_ret="Cumulative Returns",
) -> plt.Axes:
    """
    Plot the cumulative returns from a return DataFrame or dictionary.

    Parameters
    ----------
    ret_df : DataFrame or dict
        The input return data for computing cumulative returns.
    start_date : str or datetime.date, optional
        The start date for the plot. Defaults to None.
    end_date : str or datetime.date, optional
        The end date for the plot. Defaults to None.
    ax : plt.Axes, optional
        The axes on which to plot. If None, a new one is created.
    ylabel_ret : str, optional (default="Cumulative Returns")
        The label for the y-axis.

    Returns
    -------
    plt.Axes
        The axes object with the plotted cumulative returns.
    """
    ax = check_axes(ax)
    ret_df = filter_date_range(pd.DataFrame(ret_df), start_date, end_date)
    ret_df.index.name = None
    ret_df.cumsum(axis=0).plot(ax=ax)
    ax.set(ylabel=ylabel_ret)
    convert_yaxis_to_percent(ax)
    return ax


def fill_between(
    ser: pd.Series,
    start_date: str = None,
    end_date: str = None,
    ax: AXES_TYPE = None,
    color: Optional[str] = None,
    fill_between_label: Optional[str] = None,
) -> plt.Axes:
    """
    Fill the area between a curve and the x-axis with a specified color and label.

    Parameters
    ----------
    ser : pd.Series
        The data series to plot.
    start_date : str or datetime.date, optional
        The start date for the plot. Defaults to None.
    end_date : str or datetime.date, optional
        The end date for the plot. Defaults to None.
    ax : plt.Axes, optional
        The axes on which to plot. If None, a new one is created.
    color : str, optional
        The fill color. Defaults to None.
    fill_between_label : str, optional
        The label for the filled area. Defaults to None.

    Returns
    -------
    plt.Axes
        The axes object with the filled area plot.
    """
    ax = check_axes(ax)
    ser = filter_date_range(ser, start_date, end_date)
    ax.fill_between(ser.index, ser, step="pre", alpha=ALPHA_FILL, color=color, label=fill_between_label)
    ax.legend()
    return ax


def plot_regimes(
    regimes: pd_type,
    n_c: int = 2,
    start_date: str = None,
    end_date: str = None,
    ax: AXES_TYPE = None,
    colors_regimes: Optional[list] = ['g', 'r'],
    labels_regimes: Optional[list] = ['Bull', 'Bear'],
) -> plt.Axes:
    """
    Plot regime identification based on a 1D label series or 2D probability matrix.

    Parameters
    ----------
    regimes : DataFrame or Series
        The regime data to plot.
    n_c : int, optional (default=2)
        The number of components (regimes) to plot.
    start_date : str or datetime.date, optional
        The start date for the plot. Defaults to None.
    end_date : str or datetime.date, optional
        The end date for the plot. Defaults to None.
    ax : plt.Axes, optional
        The axes on which to plot. If None, a new one is created.
    colors_regimes : list, optional
        The colors for the regimes.
    labels_regimes : list, optional
        The labels for the regimes.

    Returns
    -------
    plt.Axes
        The axes object with the regime plot.
    """
    regimes = filter_date_range(regimes, start_date, end_date)
    ax = check_axes(ax)

    if colors_regimes is None:
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors_regimes = [color_cycle[i % len(color_cycle)] for i in range(n_c)]
    else:
        assert len(colors_regimes) == n_c, (
            "Mismatch between length of color list and number of components. "
            "You can input `colors_regimes = None` for colors to be generated automatically."
        )

    if labels_regimes is None:
        labels_regimes = [f"Regime {i}" for i in range(1, n_c + 1)]
    else:
        assert len(labels_regimes) == n_c, (
            "Mismatch between length of label list and number of components. "
            "You can input `labels_regimes = None` for labels to be generated automatically."
        )

    for i in range(n_c):
        fill_between(regimes.iloc[:, i], ax=ax, color=colors_regimes[i], fill_between_label=labels_regimes[i])

    return ax


def plot_regimes_and_cumret(
    regimes: pd_type,
    ret_df: pd_type,
    n_c: int = 2,
    start_date: str = None,
    end_date: str = None,
    ax: AXES_TYPE = None,
    colors_regimes: Optional[list] = ['g', 'r'],
    labels_regimes: Optional[list] = ['Bull', 'Bear'],
    ylabel_ret="Cumulative Returns",
    legend_loc="upper left"
) -> tuple[plt.Axes, plt.Axes]:
    """
    Plot cumulative returns alongside regime identification in a single figure.

    Parameters
    ----------
    regimes : DataFrame or Series
        The regime data to plot.
    ret_df : DataFrame or dict
        The return data to plot.
    n_c : int, optional (default=2)
        The number of regimes/components.
    start_date : str or datetime.date, optional
        The start date for the plot. Defaults to None.
    end_date : str or datetime.date, optional
        The end date for the plot. Defaults to None.
    ax : plt.Axes, optional
        The axes on which to plot. If None, a new one is created.
    colors_regimes : list, optional
        The colors for the regimes.
    labels_regimes : list, optional
        The labels for the regimes.
    ylabel_ret : str, optional
        The label for the cumulative return y-axis.
    legend_loc : str, optional
        The location of the legend.

    Returns
    -------
    tuple
        The axes objects for cumulative returns and regimes.
    """
    ax = plot_cumret(ret_df, start_date=start_date, end_date=end_date, ax=ax, ylabel_ret=ylabel_ret)
    ax2 = ax.twinx()
    ax2.set(ylabel="Regime")

    plot_regimes(
        regimes, n_c, start_date=start_date, end_date=end_date,
        ax=ax2, colors_regimes=colors_regimes, labels_regimes=labels_regimes
    )

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=legend_loc)

    return ax, ax2
