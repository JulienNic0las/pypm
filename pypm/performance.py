#! python2
# -*- coding:Utf-8 -*-

"""
"""

from __future__ import division
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

__version__ = '0.0'
__author__ = 'Julien NICOLAS'

def _period_helper(period):
    """Helper to transform a period given as a string into a number of
    days.
    
    Parameters
    ----------
    period : str
        Period given as a string. Accepted formats are the following
            - 'ny' for n years
            - 'nm' for n months
            - 'nd' for n days
    
    Returns
    -------
    n : int
        Number of days"""
    assert isinstance(period, str)
    if 'd' in period:
        try:
            n = int(period.split('d')[0])
        except:
            raise ValueError('Wrong format for period %s' % (period,))
    if 'm' in period:
        try:
            n = int(period.split('m')[0]) * 30
        except:
            raise ValueError('Wrong format for period %s' % (period,))
    if 'y' in period:
        try:
            n = int(period.split('y')[0]) * 365
        except:
            raise ValueError('Wrong format for period %s' % (period,))
    return n

def chunk_data(df, start_date=None, end_date=None, period=None):
    """Function to return an extract of a pandas DataFrame from a given
    date to a given date.
    
    Parameters
    ----------
    df : Pandas DataFrame 
        Dataframe of prices with dates as index.
    start_date : date object. Optional
        Default is None.
    end_date : date object. Optional
        Default is None.
    period : str, opt
        Must a string with the following format: %iy or %id or %im with
        i standing for a number (integer) and y, d or m respectively for
        year, day or month.
    
    Returns
    -------
    df : Pandas DataFrame
        A subset of input dataframe.
    
    """
    if period:
        n = _period_helper(period)
        end = df.index[-1]
        start = end - timedelta(days=n)
                
    else:
        start, end = start_date, end_date
        if not start_date:
            start = df.index[0]
        if not end_date:
            end = df.index[-1]
            
    return df.ix[start:end]

def get_returns(prices):
    """Function to return a pandas DataFrame of daily returns.
    
    Parameters
    ----------
    prices : pandas DataFrame
        Dataframe of daily prices
        
    Returns
    -------
    rets : pandas DataFrame
        DataFrame of daimy returns"""
    rets = 100 * (prices - prices.shift(1)) / prices.shift(1)
    for col in rets.columns:
        rets[col][0] = 0.0   
    return rets
    
def get_annualized_return(prices, period=None):
    """Function to compute annualized return of an asset based on daily
    prices and a given number of days (period) of observation. This is
    equivalent to the CAGR (Compound Annual Growth Rate).
        
        annualized_ret = 100 * ((return + 1)**(252/period) - 1)
    
    Parameters
    ----------
    prices : pandas DataFrame
        DataFrame of daily prices
    period : int, opt
        Number of trading days to consider for the computation of the 
        annulized return. Default is None meaning that all available 
        trading days are used. 
    
    Returns
    -------
    annualized_ret : pandas DataFrame
        DataFrame of annualized returns.
    
    """
    
    if period:
        assert isinstance(period, int)
        n = period
        prices = chunk_data(prices, period=str(period) + 'd')
    else:
        n = len(prices)
    
    prices_mat = prices.as_matrix()
    total_ret = (prices_mat[-1][:] / prices_mat[0][:]) - 1
    
    annualized_ret = dict()
    for index, ret in enumerate(total_ret):
        symbol = prices.columns[index]
        annualized_ret[symbol] = 100 * ((ret + 1)**(252/n) - 1)
        
    return pd.DataFrame(annualized_ret, index=['Expct. Ret (CAGR)'])
    
def get_sma(data, period=200):
    """Return the Simple Moving Average over a given period
    
    Parameters
    ----------
    data : Pandas DataFrame
        Dataframe of daily prices.
    period : int, opt
        Number of trading days. Default is 200.

    Returns
    -------
    ma : pandas DataFrame
        Dataframe of moving average.
    
    """
    ma = pd.rolling_mean(data, window=period)
    return ma
    
def get_ewma(data, period=200):
    """Return the Exponentional Weighted Moving Average over a given 
    period
    
    Parameters
    ----------


    Returns
    -------
    
    
    """
    ma = pd.ewma(data, span=period)
    return ma
    
def get_volatility(data, period=252):
    """ 
    
    
    Parameters
    ----------


    Returns
    -------
    
    
    """
    return data.cov()**0.5 / np.sqrt(period)
    
def get_sharpe_ratio(prices, rf=0.0, period=None):
    """ 
    
    
    Parameters
    ----------


    Returns
    -------
    
    
    """
    if period:
        name = 'Sharpe ' + period
        asked_period = timedelta(days=_period_helper(period))
        if asked_period > (prices.index[-1] - prices.index[0]):
            sharpe = {symbol: float('nan') for symbol in prices.columns}
        else:    
            prices = chunk_data(prices, period=period)
            mean = prices.mean()
            vol = prices.std()
            sharpe = np.sqrt(252) * (mean - rf/252) / vol
    else:
        name = 'Sharpe'
        mean = prices.mean()
        vol = prices.std()
        sharpe = np.sqrt(252) * (mean - rf/252) / vol

    return pd.DataFrame(dict(sharpe), index=[name], columns=prices.columns)
    
def get_sortino_ratio(prices, rf=0.0, period=None):
    """ 
    
    
    Parameters
    ----------


    Returns
    -------
    
    
    """
    if period:
        name = 'Sortino ' + period
        asked_period = timedelta(days=_period_helper(period))
        if asked_period > (prices.index[-1] - prices.index[0]):
            sortino = {symbol: float('nan') for symbol in prices.columns}
        else:    
            prices = chunk_data(prices, period=period)
            mean = prices.mean()
            vol = prices[prices <= 0.0].std()
            sortino = np.sqrt(252) * (mean - rf/252) / vol
    else:
        name = 'Sortino'
        mean = prices.mean()
        vol = prices[prices <= 0.0].std()
        sortino = np.sqrt(252) * (mean - rf/252) / vol

    return pd.DataFrame(dict(sortino), index=[name], columns=prices.columns)
    
    
def get_sterling_ratio(prices, rf=0.0):
    """ 
    
    
    Parameters
    ----------


    Returns
    -------
    
    
    """
    # Compute the annual compounded rate of return
    annualized_rate_of_ret = get_annualized_return(prices)
    drawdowns = get_drawdowns(prices)
    # Resample drawdown to compute max drawdown for each year before 
    # deriving the average.
    drawdowns = drawdowns.resample('A', how='min')
    mean_drawdowns = drawdowns.mean()
    sterling = {symbol: (annualized_rate_of_ret[symbol].values[0] - rf) \
                        / np.abs((mean_drawdowns[symbol] - 10.0))  \
                        for symbol in prices.columns}
    return pd.DataFrame(sterling, index=['Sterling'])

def get_calmar_ratio(prices, rf=0.0):
    """ Compute the CalMAR ratio (California Managed Account Reports).
    The CalMAR ratio uses the average annual rate of return for the
    last 36 months (then data set must larger 36 months) divided by
    the maximum drawdown for the same period."""
    # Make sure that prices cover at least 36 months (756 US trading 
    # days.
    if len(prices.index) < 756:
        calmar = {symbol: float('nan') for symbol in prices.columns}
        return pd.DataFrame(calmar, index=['Calmar'])
    
    # Get only last 756 trading days
    start, end = prices.index[-757], prices.index[-1]
    prices = chunk_data(prices, start_date=start, end_date=end)

    annualized_rate_of_ret = get_annualized_return(prices)
    drawdowns = get_drawdowns(prices)
    max_drawdowns = np.abs(drawdowns.min())
    calmar = {symbol: (annualized_rate_of_ret[symbol].values[0] - rf) \
                      / max_drawdowns[symbol] for symbol in prices.columns}
    return pd.DataFrame(calmar, index=['Calmar'])

def get_drawdowns(prices, start_date=None, end_date=None):
    """ 
    
    
    Parameters
    ----------


    Returns
    -------
    
    
    """
    drawdowns = chunk_data(prices, start_date=None, end_date=None)
    roll_max = np.maximum.accumulate(drawdowns)
    drawdowns = 100 * (drawdowns / roll_max - 1.)
    return drawdowns
        
def get_maxdrawdown(prices, start_date=None, end_date=None):
    """ """
    prices = chunk_data(prices, start_date=None, end_date=None)
    dates = prices.index
    values = {symbol: prices[symbol].values for symbol in prices.columns}
    
    max_drawdowns = {}
    for symbol, vals in values.iteritems():
        end = np.argmax(np.maximum.accumulate(vals) - vals)
        start = np.argmax(vals[:end]) 
        maxDD = 100 * (vals[end] - vals[start]) / vals[start]
        maxDD_duration = (dates[end] - dates[start]).days
        location = ((dates[start], dates[end]), (vals[start], vals[end])) 
        max_drawdowns[symbol] = maxDD, maxDD_duration, location
    
    return max_drawdowns

# TODO: Rewrite this function
# TODO: Check drawdown duration
def get_performance(df_prices, rf=0.0):
    """ 
    
    
    Parameters
    ----------


    Returns
    -------
    
    
    """
    df_returns = get_returns(df_prices)
    
    # Use the describe() method of pandas to get basic stats about the 
    # price evolutions
    percs = [0.01, 0.25, 0.50, 0.75, 0.99]
    perfs = df_prices.describe(percentiles=percs)
    
    # Compute expected return
    expected_return = get_annualized_return(df_prices)
    perfs = pd.concat([perfs, expected_return], axis=0)
    
    # Compute stats about drawdowns
    dd = get_drawdowns(df_prices)
    average_dd = pd.DataFrame(dict(dd.mean()), index=['Avg. DD'])
    maxdds = get_maxdrawdown(df_prices)
    maxdd = {symbol: val[0] for symbol, val in maxdds.iteritems()}
    maxdd_duration = {symbol: val[1] for symbol, val in maxdds.iteritems()}
    maxdd = pd.DataFrame(maxdd, index=['MaxDD'])
    perfs = pd.concat([perfs, maxdd], axis=0)
    maxdd_duration = pd.DataFrame(maxdd_duration, index=['MaxDD days'])
    perfs = pd.concat([perfs, maxdd_duration], axis=0)
    
    # Compute performance ratios for last 1, 3 and 5 years
    sharpe = get_sharpe_ratio(df_returns, rf=rf, period='1y')
    perfs = pd.concat([perfs, sharpe], axis=0)
    sharpe = get_sharpe_ratio(df_returns, rf=rf, period='3y')
    perfs = pd.concat([perfs, sharpe], axis=0)
    sharpe = get_sharpe_ratio(df_returns, rf=rf, period='5y')
    perfs = pd.concat([perfs, sharpe], axis=0)
    sortino = get_sortino_ratio(df_returns, rf=rf, period='1y')
    perfs = pd.concat([perfs, sortino], axis=0)
    sortino = get_sortino_ratio(df_returns, rf=rf, period='3y')
    perfs = pd.concat([perfs, sortino], axis=0)
    sortino = get_sortino_ratio(df_returns, rf=rf, period='5y')
    perfs = pd.concat([perfs, sortino], axis=0)
    sterling = get_sterling_ratio(df_prices, rf=rf)
    perfs = pd.concat([perfs, sterling], axis=0)
    calmar = get_calmar_ratio(df_prices, rf=rf)
    perfs = pd.concat([perfs, calmar], axis=0)
    
    return np.round(perfs, decimals=3)
    
if __name__ == '__main__':
    
    pass
