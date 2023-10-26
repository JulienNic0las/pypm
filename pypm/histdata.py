"""

"""
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
import pandas_datareader.data as web

from . import utils

def _clean_ohlc(df):
    del(df['Close'])
    del(df['Volume'])
    df = df.rename(columns={'Adj Close': 'Close'})
    return df

def _get_from_yahoo(ticker, start_date=None, end_date=None):
    
    # Data reader with Yahoo no longer supported. Use yfinance instead
    import yfinance as yfin
    yfin.pdr_override()

    df = web.get_data_yahoo(ticker, start=start_date, end=end_date)
    
    if df.empty:
        raise IOError('Yahoo finance failed to fetch %s' % ticker)
    
    return _clean_ohlc(df)

def _get_from_google(ticker, start_date=None, end_date=None):
    raise NotImplementedError()

def _get_from_quandl(ticker, start_date=None, end_date=None):
    raise NotImplementedError()

def _get_from_euronext(isin, start_date=None, end_date=None):
    raise NotImplementedError()

def _get_from_eurostat(ticker, start_date=None, end_date=None):
    raise NotImplementedError()
    
def _get_from_file(fname, star_date=None, end_date=None):
    raise NotImplementedError()


def get_historical_data(tickers, from_date=None, to_date=None, src='web',
        clean=False, fname=None, bdays=False, verbose=False,
        ):
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
        Source of the data. Optional. Can be either 'web' or 'file'
        Default is 'web'. 
    fname : str, opt
        Name of the file to read in case of src=='file'. Not yet implemented.
    clean : bool, opt
        If True, available date of every tickers are compared and non
        available values are forward filled.
    bdays : bool, opt
        If True original date index are replaced by buisness days.
    verbose : bool, opt
        If True, print the entire error message. Default is False.

    Returns
    -------
    ohlc : dict
        Dictionary of ohlc pandas dataframe. {ticker: df}.
    """
    
    assert src in ('web', 'file')
    
    if not isinstance(tickers, list):
        tickers = [tickers,]
    
    # Mapper of the extracting functions
    if src == 'file':
        func_map = {'file': _get_from_file}
    else:
        func_map = {
            'yahoo': _get_from_yahoo,
            'euronext': _get_from_euronext,
            }

    # Get data
    ohlcs = dict()
    for ticker in tickers:
        for src, func in func_map.items():
            try:
                data = func(ticker, start_date=from_date, end_date=to_date)
            except Exception as e:
                msg = 'Failed to get %s from %s' % (ticker, src)
                if verbose:
                    msg += '\n' + str(e)
                print(msg)
            else:
                ohlcs[ticker] = data
                break
            
    if not ohlcs:
        err_msg = 'Extracting data failed for all tickers'
        raise IOError(err_msg)
    
    if clean:
        ohlcs = utils.clean_hist_data(ohlcs, bdays=bdays)

    return ohlcs

