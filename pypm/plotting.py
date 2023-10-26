"""
Module containing handy plotting functions
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-deep")


def _plot_io(**kwargs):
    """Helper method to optionally save the figure to file."""
    filename = kwargs.get("filename", None)
    showfig = kwargs.get("showfig", True)

    plt.tight_layout()
    if filename:
        plt.savefig(fname=filename, dpi=dpi)
    if showfig:
        plt.show()

def plot_time_series(ts, **kwargs):
    """Plot the portfolio time series
    
    Parameters
    ----------
    ts : pandas series
        Series of portfolio values
    
    Returns
    -------
    ax : matplotlib ax
        matplotlib axis
    """
    fig, ax = plt.subplots()
    ax.plot(ts)
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio value')
    ax.grid(True)
    
    _plot_io(**kwargs)
    
    return ax

def plot_pf_equity_curves(pf_equity_data, pf_ts, names=None, **kwargs):
    """Plot the portfolio equity curves
    
    Parameters
    ----------
    pf_equity_data : pandas DataFrame
        DataFrame of equity curves
    pf_ts : pandas series
        Series of portfolio values
    names : list, optionnal
        List of label names. By default use DataFrame column names
    
    Returns
    -------
    ax : matplotlib ax
        matplotlib axis
    """
    _lbl_names = names if names else pf_equity_data.columns
    
    fig, ax = plt.subplots(1)
    for ticker, lbl in zip(pf_equity_data.columns, _lbl_names):
        ax.plot(
            pf_equity_data.index, pf_equity_data[ticker],
            label=lbl, alpha=0.5,
            )
    ax.plot(pf_ts, color='black', label='Portfolio')
    ax.grid(True)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    
    _plot_io(**kwargs)
    
    return ax
    

def plot_covariance(cov_matrix, show_tickers=True, **kwargs):
    """Generate a basic plot of the covariance (or correlation) matrix, 
    given a covariance matrix.

    Parameters
    ----------
    cov_matrix : numpy 2D array
        Covariance matrix
    show_tickers : bool, optional
        If True display name of tickers. Optional. Default is True.


    Returns
    -------
    ax : matplotlib ax
        matplotlib axis
    """

    fig, ax = plt.subplots()

    cax = ax.imshow(cov_matrix)
    fig.colorbar(cax)

    if show_tickers:
        ax.set_xticks(np.arange(0, cov_matrix.shape[0], 1))
        ax.set_xticklabels(cov_matrix.index)
        ax.set_yticks(np.arange(0, cov_matrix.shape[0], 1))
        ax.set_yticklabels(cov_matrix.index)
        plt.xticks(rotation=90)

    _plot_io(**kwargs)

    return ax

def plot_return_distribution(rets, period='M', nbins=10):
    """Display a distribution of returns.
    
    Parameters
    ----------
    rets : pandas time series
        Series of returns.
    period : str, optional
        Period for resampling data. Defaut is 'M' standing for monthly.
        Can be either 'M', 'D' or 'Y'.
    nbins : int, optional
        Number of bins of the distribution. By default 10.
    """
    
    if period not in ('M', 'D', 'Y'):
        print('Provided resampling period not supported %s. Switch to default' % period)
        period = 'M'
    
    period_mapping = {'M': 'Monthly', 'D': 'Daily', 'Y': 'Annual'}
    
    # Resample data 
    rets = rets.resample(period).apply(lambda x: np.prod(x + 1) - 1) * 100.0
    
    # Determine distribution
    hist, bin_edges = np.histogram(rets, bins=nbins, density=True)
    
    # Get statistics about returns
    stats = rets.describe()
    
    # Plot. Two axes. The left one contains a distribution plot and the 
    # right one contains distribution stats
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax0, ax1 = plt.subplot(gs[0]), plt.subplot(gs[1])
    ax1.axis('off')
        
    # Distribution plot
    sns.distplot(rets['Returns'].values, kde=True, color="b", ax=ax0)
    ax0.grid(True)
    ax0.set_xlabel('Returns (%)')
    ax0.set_ylabel('Density (-)')
        
    # stats
    cell_text = [[d, '%0.2f' % val] for d, val in stats.iterrows()]
    perf_table = ax1.table(cellText=cell_text, loc='center')
        
    plt.suptitle('%s Returns Analysis' % rules_mapping.get(rule, ''))
    plt.show()


def plot_efficient_frontier():
    NotImplementedError()
    
def plot_weights(weights, names, **kwargs):
    """Plot the portfolio weights as a horizontal bar chart.
    
    Parameters
    ----------
    weights : list
        The portfolio weights.
    names : list of strings
        Labels associated to weights. Should be in the same order.

    Returns
    -------
    ax : matplotlib axis
    """
    
    _weights = {name: w for (name, w) in zip(names, weights)}
    
    fig, ax = plt.subplots()
    
    desc = sorted(_weights.items(), key=lambda x: x[1], reverse=True)
    labels = [i[0] for i in desc]
    vals = [i[1] for i in desc]

    y_pos = np.arange(len(labels))

    ax.barh(y_pos, vals)
    ax.set_xlabel("Weight")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    _plot_io(**kwargs)
    
    return ax

