#! python2
# -*- coding:Utf-8 -*-

from __future__ import division
import numpy as np
import pandas as pd

import performance
import histdata

class Portfolio(object):
    """ """
    def __init__(self, symbols, start_date=None, end_date=None,
                 weights=None, src='yahoo'):
    
        self.symbols = symbols
        self.weights = self.set_weights(weights)
        
        # Download OHLC (Open High Low Close) data from source (dict)
        self.ohlcs = self._get_data(
            symbols,
            start_date,
            end_date,
            src=src)
        
        # Extract close price only and build a unique dataframe
        prices = None
        for symbol in symbols:
            data = pd.DataFrame(self.ohlcs[symbol]['Close'])
            data = data.rename(columns={'Close': symbol})
            if prices is None:
                prices = data
            else:
                prices = pd.concat((prices, data), axis=1)
        
        # Drop N/A as available daily prices unfortunately have not the
        # same starting date
        self.daily_prices = prices.dropna()
        
        # Compute daily returns
        self.daily_returns = self.daily_prices - self.daily_prices.shift(1)
        
    
    def set_weights(self, weights=None):
        """Set asset weights in portfolio"""
        
        return self._set_weights(weights)
        
        
    def _set_weights(self, weights):
        
        if not weights:
            return np.array([1/len(self.symbols) for s in self.symbols])
        
        if isinstance(weights, list):
            if len(weights) != len(self.symbols):
                raise ValueError('Weights must be of the same size as '
                                 'the list of symbols of the portfolio')
        elif isinstance(weights, (np.ndarray, np.generic)):
            if weights.shape[0] != len(self.symbols):
                raise ValueError('Weights must be of the same size as '
                                 'the list of symbols of the portfolio')
        else:
            raise ValueError('Wrong type for weights. Must be either a'
                             ' list or a numpy array')
        if round(np.sum(weights), 2) != 1.0:
            raise ValueError('The sum of weights must be equal to 1. '
                             'weights = ' + str(weights))
                                      
        return np.array(weights)
    
    
    def _get_data(self, symbols, start_date, end_date, src):
        """Fetch historical data as a dict of ohlc (Open High Low Close)
        pandas DataFrames."""
        
        ohlcs = histdata.get_historical_data(
            symbols,
            start_date, end_date,
            src,
            clean=True,
            )
            
        return ohlcs
        
        
    def get_performance(self):
        """Return performance metrics of the portfolio and each asset
        which compose it based on historical data only"""
        
        # Compute each asset performance metrics
        asset_perf = performance.get_performance(self.daily_prices)
        
        # Compute weighted portfolio perfomance metrics
        holdings = (self.daily_prices * self.weights).sum(axis=1)
        holdings = pd.DataFrame(holdings, columns=['PORTFOLIO'])
        portfolio_perf = performance.get_performance(holdings)
        
        return pd.concat((asset_perf, portfolio_perf), axis=1)


    def get_correlation(self):
        """Return correlation matix of the portfolio assets"""
        
        return self.daily_returns.corr()
        

if __name__ == '__main__':
    
    tickers = ['LU0496786574', 'FR0010411439']
    weights = [0.8, 0.2]
    
    portfolio = Portfolio(tickers, weights=weights, src='euronext')
    
    # Asset correlation
    correletion = portfolio.get_correlation()
    
    # Portfolio performance
    print portfolio.get_performance()
