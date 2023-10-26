"""
Module defining the portfolio object.
"""
import copy
import numpy as np
import pandas as pd

from . import histdata, utils, metrics

import matplotlib.pyplot as plt
import seaborn as sns


def _get_historical_data(tickers, start=None, end=None):
    """Helper function to build a DataFrame of ticker values.
    
    Parameters
    ----------
    tickers : list of str
        List tickers to get historical data
    start : datetime object, opt
        Start date of the data to gather
    end : datetime object, opt
        End date of the data to gather
    
    Returns
    -------
    df : DataFrame
        DataFrame of historical data
    """
    
    data = histdata.get_historical_data(
        tickers,
        from_date=start, to_date=end,
        clean=True, bdays=True,
        )
    
    if len(data.keys()) != len(tickers):
        raise IOError('Error during downloading of some historical data')    
        
    df = pd.concat({ticker: ohlc['Close'] for ticker, ohlc in data.items()}, axis=1)
    
    return df


class Portfolio(object):
    """ Portfolio object.
    
    Accessible attributes
    ---------------------
    tickers : list
        List of ISIN codes. Strings.
    ts : Pandas series
        Portfolio time series starting from value 1. Based on provided
        tickers and weights.
    weights : tuple
        Portfolio weights associated with tickers. Weights are comprises
        betweeen 0 and 1.
    start : Datetime 
        Earliest date in the portfolio time series
    end : Datetime
        Latest date in the portfolio time series
    metrics : dict
        Default metrics: Sharpe, Sortino, max drawdown, expected return.
    corr_mat : numpy 2D array
        Correlation matrix.
    
    Accessible methods
    ------------------
    change_tickers 
        Update tickers.
    
    """
    
    def __init__(self, tickers, weights=None, start=None, end=None):
        
        if not isinstance(tickers, list):
            raise TypeError('Provided argument for tickers is not a list')
    
        self._raw_hist_data = None         # Historical values of each ticker of the portfolio
        self.ts = None                     # TODO: change that name
        self._equity_hist_data = None      # Normalized historical evolution of each ticker
        self._weights = None
        
        self.tickers = tickers
        self.start = start
        self.end = end
        self.weights = weights

        # Build portfolio
        self._build_pf_ts()
    
    @property
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, values):
        
        if not values:
            values = (1 / len(self.tickers) for ticker in self.tickers)
        elif len(values) != len(self.tickers):
            raise ValueError('Size of weights does not match number of tickers')
        elif abs(np.sum(values) - 1.0) > 1e-4:
            raise ValueError('Sum of provided weights not egal to 1.0 : %0.4f' % np.sum(values))
            
        self._weights = values
        
        # Rebuild portfolio time series
        if self._raw_hist_data is not None:
            rets = metrics.returns(self._raw_hist_data)
            ts = ((rets + 1).cumprod() * self._weights).sum(axis=1)
            ts.name = 'portfolio_ts'
            self.ts = ts
    
    @property
    def equity_hist_data(self):
        rets = metrics.returns(self._raw_hist_data)
        self._equity_hist_data = (rets + 1).cumprod()
        return self._equity_hist_data
    
    def _build_pf_ts(self):
        """ """
        # Get historical data. Buisness days.
        self._raw_hist_data = _get_historical_data(
            self.tickers, start=self.start, end=self.end,
            )
        # Build portfolio historical data based on provided weights starting
        # from one euro invested
        rets = metrics.returns(self._raw_hist_data)
        ts = ((rets + 1).cumprod() * self.weights).sum(axis=1)
        ts.name = 'portfolio_ts'
        self.ts = ts
        
        # Update start and end to match the time serie
        self.start, self.end = ts.index[0], ts.index[-1]
    
    def change_tickers(self, tickers, weights=None):
        """Modify the list of tickers in the portfolio.
        
        Parameters
        ----------
        tickers : list
            List of tickers (ISIN codes) as strings.
        weights : list, optional
            List of weights associated with the provided tickers. Optional.
            If no weights are provided, an homogeneous repartition will
            be applied.
        """
        self.tickers = tickers
        self.weights = weights
        self._build_pf_ts()
        
    @property    
    def sharpe(self):
        return metrics.sharpe(self.ts, as_float=True)
        
    @property
    def metrics(self):
        """Compute main metrics. Return a dict"""
        m = {
            'cagr': metrics.cagr(self.ts, as_float=True),
            'max_dd': metrics.max_drawdown(self.ts, as_float=True),
            'sharpe': metrics.sharpe(self.ts, as_float=True),
            'sortino': metrics.sortino(self.ts, as_float=True),
            }
        m = {k: round(v, 3) for k, v in m.items()}
        return m
    
    @property
    def corr_mat(self):
        """Return the correlation matrix of the portfolio"""
        return self._raw_hist_data.corr()


def _monte_carlo_optimization(pf, metric_callable):
    """ """
    
    results = dict()
    for i in range(10):
        weights = np.random.random(len(pf.tickers))
        weights = (weights / weights.sum()).tolist()
        pf.weights = weights
        results[i] = [weights, metric_callable(pf.ts)]
    return results
        
        
def optimize_portfolio(pf, obj_func, bnds=None):
    """Find the weights combinaison that minimize the objective function.
    
    Parameters
    ----------
    pf : object
        Portfolio object
    obj_func : callable
        Objective function to minimize.
    bnds : iterable, optional
        List (or tuple) of bounds (ex: [(0, 1), (0.2, 0.5)]. Optional.
        By default bounds are set between 0 and 1 for each weight.
        
    Returns
    -------
    res : dict
        Results of the optimization. Weights are accessible with res.x.
    """
    
    from scipy.optimize import minimize
    from scipy.optimize import approx_fprime
    
    # Make a copy of the provided portfolio object to prevent side effects
    _pf = copy.deepcopy(pf)
    
    # Initial values for weights
    weights = [w for w in _pf.weights]
    
    # Handle default bounds
    if not bnds or len(bnds) != len(_pf.weights):
        bnds = [(0, 1) for w in _pf.weights]
    
    # Make sure that the sum of weights are equal to 1
    cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
    
    def objective_func(x, *args, **kwargs):
        # Set portfolio weights
        _pf.weights = (x / x.sum()).tolist()
        # Compute metric
        val = obj_func(_pf.ts)
        return val
        
    def jacobian_func(x):
        jac = approx_fprime(x, objective_func, epsilon=0.01)
        return jac
        
    res = minimize(
        objective_func, weights, method='SLSQP', bounds=bnds,
        constraints=cons, jac=jacobian_func,
        )
    
    return res
    
# Work in progress
def efficient_frontier(pf, n_iter=10):
    """ """
    
    results = dict()
    
    for i in range(n_iter):
        weights = np.random.random(len(pf.tickers))
        weights = (weights / weights.sum()).tolist()
        pf.weights = weights
        results[i] = [
            weights,
            metrics.annual_avg_return(pf.ts, as_float=True),
            metrics.annual_std_return(pf.ts, as_float=True),
            metrics.sharpe(pf.ts, as_float=True),
            ]

    # Get portfolio with the highest Sharpe Ratio
    max_sharpe_id = max(results, key=lambda x:results[x][3])
    
    msg = 'Portfolio with the highest found Sharpe ratio:\n'
    msg += '  \n'.join(['%s : %0.3f' % (ticker, w) for (ticker, w) in zip(pf.tickers, results[max_sharpe_id][0])]) + '\n'
    msg += 'With a Sharpe ratio of %0.3f' % results[max_sharpe_id][3]
    print(msg)
        
    fig, ax = plt.subplots()
    ax.scatter(
        [v[2] for k, v in results.items()],
        [v[1] for k, v in results.items()],
        c=[v[3] for k, v in results.items()],
        )
    ax.grid(True)
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Annual return')
    
    plt.show()
        
    return results[max_sharpe_id][0]
    
