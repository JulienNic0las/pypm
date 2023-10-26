"""
Module containing metrics for financial analyses
"""
from datetime import datetime
from functools import wraps

import numpy as np
import pandas as pd

from . import utils

def df_to_ts(func):
    """Decorator in charge of treating dataframes as series"""
    @wraps(func)
    def wrapper(arg, **kwargs):
        if isinstance(arg, pd.DataFrame):
            res = {c : func(arg[c], **kwargs) for c in arg.columns}
            return res
        elif kwargs['as_float']:
            return func(arg, **kwargs)
        res = {arg.name: func(arg, **kwargs)}
        return res
    return wrapper


def returns(df):
    """Return the returns of a dataframe"""
    rets = (df - df.shift(1)) / df.shift(1)
    rets.iloc[0] = 0.0
    return rets

def total_return(df):
    """Return the total return of a dataframe"""
    return (df.values[-1] - df.values[0]) / df.values[0]

@df_to_ts  
def mean_return(ts, as_float=False):
    return returns(ts).values.mean()

@df_to_ts  
def std_return(ts, as_float=False):
    return returns(ts).values.std()

@df_to_ts 
def annualized_return(ts, as_float=False):
    """Return the Compound Annual Growth Return (CAGR)"""
    # Compute total return
    total_ret = total_return(ts)
    # Compute floating number of years from time series begining to end
    nb_years = (ts.index[-1] - ts.index[0]).days / 365
    cagr = 100 * ((total_ret + 1)**(1 / nb_years) - 1)
    return cagr

@df_to_ts  
def annual_avg_return(ts, as_float=False):
    """Return the Annual Average Return (AAR)"""
    aar = returns(ts).values.mean() * 252
    return aar
    
@df_to_ts
def annual_std_return(ts, as_float=False):
    """Return the Annual Volatility Return"""
    avr = returns(ts).values.std() * np.sqrt(252)
    return avr

def cagr(ts, as_float=False):
    return annualized_return(ts, as_float=as_float)

@df_to_ts  
def drawdowns(ts, as_float=False):
    roll_max = np.maximum.accumulate(ts.values)
    drawdowns = 100 * (ts.values / roll_max - 1.0)
    return drawdowns 

@df_to_ts
def max_drawdown(ts, as_float=False):
    """Return the maximum drawdown"""
    dates, values = ts.index,  ts.values
    try:
        end = np.argmax(np.maximum.accumulate(values) - values)
        start = np.argmax(values[:end])
        max_dd = 100 * (values[end] - values[start]) / values[start]
    except Exception as e:
        print(e)
        return 0.0
    return max_dd

@df_to_ts
def longest_drawdown(ts, as_float=False):
    """Return the longest drawdown"""
    # Get all drawdowns. Make it a dataframe to ease date index manip
    df = ts.to_frame()
    df['dd'] = drawdowns(ts)
    # Get dates when drawdown is null
    zero_dd = df[df['dd'] == 0].index
    # get drawdown durations
    dd_durations = zero_dd[1:] - zero_dd[:-1]
    # Get max as days
    longest_dd = dd_durations.max().days
    return longest_dd
       
@df_to_ts  
def sharpe(ts, rf=0.0, horizon=None, as_float=False):
    """Return the sharpe ratio (annualized)"""
    rets = returns(ts)
    mean, vol = rets.mean(), rets.std()
    sharpe = np.sqrt(252) * (mean - rf) / vol
    return sharpe

@df_to_ts  
def sortino(ts, rf=0.0, horizon=None, as_float=False):
    """Return the sortino ratio (annualized)"""
    rets = returns(ts)
    mean, vol = rets.mean(), rets[rets <= 0.0].std()
    sortino = np.sqrt(252) * (mean - rf) / vol
    return sortino

@df_to_ts  
def ulcer_index(ts, as_float=False):
    pass


def _test(ts):
    """Testing function"""
    pass


if __name__ == '__main__':

    import histdata

    import matplotlib.pyplot as plt

    # Get sample data
    tickers = ['FR0013412285', 'LU1598690169']
    data = histdata.get_historical_data(
        tickers,
        from_date=datetime(2010, 1, 1, 0, 0),
        clean=True, bdays=True,
        )
    df = pd.concat({ticker: ohlc['Close'] for ticker, ohlc in data.items()}, axis=1)
    print(df)
        
    # Compute returns
    rets = returns(df)
    print(rets)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(rets.index, rets[rets.columns[0]], label=rets.columns[0])
    ax.plot(rets.index, rets[rets.columns[1]], label=rets.columns[1])
    ax.grid(True)
    ax.legend()
    
    plt.show()
    
    
