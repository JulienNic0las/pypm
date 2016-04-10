TODO
---

 * [ ] Build `portfolio` class to manage several assets at once. This object must be able to work with the `optimizer`, `backtest` and `performance` modules
 * [ ] Build `optimizer`
 * [ ] build plot module to handle:
     - equity curve
     - underwater historical data
     - indicators
 * [ ] Build a `backtest` module relying on `backtrader` module able to to:
     - take a `portfolio` object in input
     - manage weights and reallocation
     - derive some useful outputs (value, weights, trades, etc...)
     - to use economical indicators
     - perform benchmarks (buy-and-hold based on the exposure time, or classic 60/40, and return alpha/beta plot)
 * [ ] Build an interface between `backtrader` indicators and my own indicators (pandas) (proxy?)
 * [ ] Look to `pyfolio` module developped by Quantopian guys to validate performance metrics and extract useful statistic and plotting functions
 
