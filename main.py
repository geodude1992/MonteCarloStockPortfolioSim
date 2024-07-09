"""
Implement the Monte Carlo Method to simulate a stock Portfolio
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

#import data
def get_data(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start, end)
    stockData = stockData['Close']
    returns = stockData.pct_change()    # percent change to see daily change
    meanReturns = returns.mean()
    covarMatrix = returns.cov()
    return meanReturns, covarMatrix

stockList = ['GME', 'NVDA', 'TSLA', 'AAPL']
stocks = [stock for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

meanReturns, covarMatrix = get_data(stocks, startDate, endDate)

weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

#print(weights)

# Monte Carlo Simulation
# number of simulations
mc_sims = 10
T = 10 # timeframe in days


meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T

portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

initialPortfolio = 10000

for m in range(0, mc_sims):
    # Monte Carlo loops
    Z = np.random.normal(size=(T, len(weights)))
    L = np.linalg.cholesky(covarMatrix)
    dailyReturns = meanM + np.inner(L, Z)
    portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T) + 1) * initialPortfolio

plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('Monte Carlo Simulation of a stock portfolio')
plt.show()