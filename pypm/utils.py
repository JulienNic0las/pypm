"""
Module containing utilitary functions.
"""
import numpy as np
import pandas as pd

def unix_time_ms(dt):
    """Return number of milliseconds since 1970 (unix time)"""
    ref = datetime.utcfromtimestamp(0)
    return int((dt - ref).total_seconds() * 1000)

def resample(df, ruler=None, method='D'):
    raise NotImplemented()

def chunk_data(df, start_date=None, end_date=None):
    """"""
    if not start_date:
        start_date = df.index[0]
    if not end_date:
        end_date = df.index[-1]
    return df[(df.index >= start_date) & (df.index <= end_date)]

def clean_hist_data(df_dict, bdays=False):
    """Function to clean historical data. Matching starting and ending 
    date for each ticker. Missing values are filled forward using
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

    # Make sure that each data start at the same date
    start_date = max([df.index[0] for df in df_dict.values()])
    end_date = min([df.index[-1] for df in df_dict.values()])
    for ticker, df in df_dict.items():
        df_dict[ticker] = df[(df.index >= start_date) & (df.index <= end_date)]

    # Reindex data
    if not bdays:
        indexes = []
        for df in df_dict.values():
            indexes += list(df.index.values)
        indexes = np.sort(list(set(indexes)))
    else:
        indexes = pd.bdate_range(start_date, end_date)
        
    for ticker in df_dict.keys():
        df_dict[ticker] = df_dict[ticker].reindex(indexes)
        df_dict[ticker] = df_dict[ticker].fillna(method='ffill')

    return df_dict


if __name__ == '__main__':
    pass
