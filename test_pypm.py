"""" 
File to test the pypm module.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pypm import histdata
from pypm import metrics
from pypm import plotting
from pypm import portfolio


# Build a portfolio for testing purposes
pf = portfolio.Portfolio(
    [
    'FR0013412285', # SP500
    'FR0013041530', # SP500 hedged
    'LU1681038672', # Russell 2000
    'FR0013412038', # Europe large
    'LU1598689153', # Europe small
    'LU1598690169', # Europe value
    'FR0011869304', # Real estate
    'FR0013412020', # Emerging
    ],
    weights=(0.2, 0.2, 0.15, 0.25, 0.05, 0.05, 0.05, 0.05),
    start=None, end=None
    )

# Display portfolio main metrics
print(pf.metrics)

# Display portfolio equity curve
plotting.plot_pf_equity_curves(
pf.equity_hist_data, pf.ts,
    names=[
        'SP500', 'SP500_Hedged', 'Russell_2000', 'EU_Large',
        'EU_Small', 'EU_Value', 'Real_Estate', 'Emerging',
        ],        
    )

# Display portfolio covariance matrix
plotting.plot_covariance(pf.corr_mat)

# Weight optimization
obj_func = lambda x: -metrics.sharpe(x, as_float=True)
res = portfolio.optimize_portfolio(pf, obj_func, bnds=None)
print(res)

plotting.plot_weights(
    res.x,
    names=[
        'SP500', 'SP500_Hedged', 'Russell_2000', 'EU_Large',
        'EU_Small', 'EU_Value', 'Real_Estate', 'Emerging',
         ],
    )
    
    
