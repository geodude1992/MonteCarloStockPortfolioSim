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
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covarMatrix = returns.cov()
    return meanReturns, covarMatrix

stockList = ['GME', 'NVDA', 'TSLA', 'AAPL']
stocks = [stock for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

meanReturns, covarMatrix = get_data(stocks, startDate, endDate)

print(meanReturns)