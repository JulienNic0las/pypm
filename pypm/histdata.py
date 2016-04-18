#! python2
# -*- coding:Utf-8 -*-

"""
Module to download historical prices (Open High Low Close) from 
differnt sources: Yahoo Finance, Google Finance, Euronext and from
user own data base (files)
"""

from __future__ import division
from datetime import datetime, date, timedelta
import copy
import numpy as np
import pandas as pd
from pandas_datareader import wb

__version__ = '0.0'
__author__ = 'Julien NICOLAS'

def unix_time_ms(dt):
    """Return number of milliseconds since 1970 (unix time)"""
    ref = datetime.utcfromtimestamp(0)
    return int((dt - ref).total_seconds() * 1000)
    
def _clean_hist_data(df_dict, bdays=False):
    """Function to clean historical data for several asset. Starting
    date of each asset historical data becomes the newest of the lastest
    avalaible date. In addition missing values are filled forward using 
    last available value.
    
    Parameters
    ----------
    df_dict : Dict
        Dictionnay of pandas DataFrame
    bdays : Bool, opt
        If True available prices are reindexed using pandas buissness 
        days. Missing values will be fillforward using last available 
        date. Default is False. Note: holidays are not removed. 
    
    Returns
    -------
    ohlcs : Dict
        Dictionnary of cleaned pandas dataframe
    """
    
    ohlcs = df_dict
    
    # Make sure that each data start at the same date
    start_date = max([df.index[0] for df in ohlcs.values()])
    end_date = max([df.index[-1] for df in ohlcs.values()])
    for symbol, df in ohlcs.iteritems():
        ohlcs[symbol] = chunk_data(df, start_date=start_date)
        
    # Reindex historical data and fill with last valid value
    if not bdays:
        indexes = []
        for ohlc in ohlcs.values():
            indexes += list(ohlc.index.values)
        indexes = np.sort(list(set(indexes)))
    else:
        indexes = pd.bdate_range(start_date, end_date)

    for symbol, df in ohlcs.iteritems():
        new_df = pd.DataFrame(index=indexes, columns=[''])
        new_df = pd.concat((new_df, df), axis=1)
        ohlcs[symbol] = new_df.fillna(method='ffill')
        
    return ohlcs
        

# Maybe move this part to a "utils.py"
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
                
        end = df.index[-1]
        start = end - timedelta(days=n)
                
    else:
        start, end = start_date, end_date
        if not start_date:
            start = df.index[0]
        if not end_date:
            end = df.index[-1]
            
    return df.ix[start:end]

def _get_from_yahoo(ticker):
    
    df = wb.DataReader(ticker, data_source='yahoo')
    del(df['Close'])
    del(df['Volume'])
    df = df.rename(columns={'Adj Close': 'Close'})
    return df
    
def _get_from_google(ticker):
    
    return wb.DataReader(ticker, data_source='google')

def _get_from_quandl(ticker):
    
    try:
        import Quandl
    except:
        raise ImportError('Module Quandl is missing')
    df = Quandl.get(ticker)
    return df
    
def _get_from_euronext(isin):
    
    import io
    import requests
    
    # Build url of the product to download
    today = datetime.now()
    from_date, to_date = 0, unix_time_ms(today)
    template = 'https://www.euronext.com/nyx_eu_listings/price_chart/' \
               'download_historical?typefile=csv&layout=vertical&' \
               'typedate=dmy&separator=point&mic=XPAR&isin={0}&' \
               'namefile=Price_Data_Historical&from={1}&to={2}' \
               '&adjusted=1&base=0'
    url = template.format(isin, from_date, to_date)
                          
    # Download csv file from Euronext
    response = requests.get(url).content
    
    # Build Pandas DataFrame of OHLC
    df = pd.read_csv(io.StringIO(response.decode('utf-8')), skiprows=3,
                     usecols=['Date', 'Open', 'High', 'Low', 'Close'])
    df.index = pd.to_datetime(df['Date'].values, format='%d/%m/%Y')
    del(df['Date'])
    df.index.name = 'Date'
    
    return  df
    
def _get_from_file(fname):  
    
    df = pd.read_csv(fname, sep='\t')
    df.index = pd.to_datetime(df['Date'].values, format='%d/%m/%Y')
    del(df['Date'])
    df.index.name = 'Date'
    return df

def get_economic_indicator(from_date=None, to_date=None):
    """Function to download historical economical indicators from Quandl
    
    Parameters
    ----------
    from_date : datetime object
        Start date of the data to gather
    to_date : datetime object
        End date of the data to gather
        
    Returns
    -------
    df : pandas DataFrame
        Pandas DataFrame of US growth indicators
    """
    
    func = _get_from_quandl
    tickers = ['FRED/RRSFS',    # Real Retail and Food Services Sales
               'FRED/INDPRO',   # Industrial Production Index
               'FRED/UNRATE']   # Civilian Unemployment Rate
    
    df = None
    for ticker in tickers:
        data = chunk_data(func(ticker), from_date, to_date)
        data = data.rename(columns={'VALUE': ticker})
        if df is None:
            df = data
        else:
            df = pd.concat((df, data), axis=1)

    return df

def get_historical_data(tickers, from_date=None, to_date=None, src='yahoo',
                        clean=False, freq='D', fname=None, bdays=False):
    """Function to download historical OHLC for one or several tickers.
    
    Parameters
    ----------
    tickers : str or list of str
        Name of or list of name of the product to download
    from_date : datetime object
        Start date of the data to gather
    to_date : datetime object
        End date of the data to gather
    src : str, opt
        Source of the data. Optional. Can be either 'yahoo', 'google' or
        'euronext'. Default is 'yahoo'. If data are not available in the
        specified source, every other source will be tried.
    fname : str, opt
        Name of the file to read in case of src=='file'. Not yet implemented.
    homogeneous : bool, opt
        If True return ohlc with the same range of date. Default is False.
    clean : bool, opt
        If True, available date of every tickers are compared and non
        available values are forward filled.
    freq : str, opt
        Can be either 'D' (days, default) or 'W' (week), 'M' (months),
        'A' (annual).
    bdays : bool, opt
        If True original date index are replaced by buisness days. 
    
    Returns
    -------
    ohlc : dict
        Dictionary of ohlc pandas dataframe. {ticker: df}.
    """
    
    src_list = ['euronext','yahoo', 'google']
    assert src in src_list
    assert freq in ('D', 'W', 'M', 'A')
    
    def get_ohlc(func, tickers, start_date, end_date):
        if isinstance(tickers, list):
            ohlc = {ticker: chunk_data(func(ticker), start_date, end_date) 
                    for ticker in tickers}
        else:
            ohlc = {tickers: chunk_data(func(tickers), start_date, end_date)}
        return ohlc
    
    # Mapper of the extracting functions
    func_map = {'yahoo': _get_from_yahoo,
                'google': _get_from_google,
                'euronext': _get_from_euronext}
    
    while src_list:
        try:
            func = func_map[src]
            ohlc = get_ohlc(func, tickers, from_date, to_date)
            break
        except:
            # Remaining source list            
            src_list.remove(src)
            if src_list:
                src = src_list[0]
    
    # If every source has been tried but no values have been extracted  
    if not src_list:
        raise IOError('Extracting data from network failed for every '
                      'available sources. Please check ticker(s) or '
                      'connection.')
      
    if bdays:
        if not from_date:
            from_date = min([ohlc[tick].index[0] for tick in ohlc.keys()])
        if not to_date:
            to_date = datetime.today()
            
        new_index = pd.bdate_range(from_date, to_date)
        
        for tick, data in ohlc.iteritems():
            ohlc[tick] = ohlc[tick].reindex(new_index)
            ohlc[tick] = ohlc[tick].fillna(method='ffill')
            
        
    # Keep same range of dates
    if clean:
        ohlc = _clean_hist_data(ohlc, bdays=bdays)
        
    # Collapse data
    for tick, data in ohlc.iteritems():
         ohlc[tick] = data.groupby(pd.TimeGrouper(freq)).nth(0)
                
    return ohlc
            

if __name__ == '__main__':

    from_date = datetime(2001, 12, 16, 0, 0)
    to_date = datetime(2016, 01, 22, 0, 0)
    
    tickers = ['VTI', 'BND']

    data = get_historical_data(
        tickers,
        #from_date,
        #to_date,
        src='euronext',
        clean=False,
        bdays=True)
        
    print data[tickers[0]]
                
                               
    
    
    
