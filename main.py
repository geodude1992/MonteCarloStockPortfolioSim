"""
Implement the Monte Carlo Method to simulate a stock Portfolio
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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

# Define stocks and dates
stockList = ['GME', 'NVDA', 'TSLA', 'AAPL']
stocks = [stock for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

# Get mean returns and covariance matrix
meanReturns, covarMatrix = get_data(stocks, startDate, endDate)

weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

#print(weights)





# Monte Carlo Simulation
mc_sims = 100   # number of simulations
T = 100     # timeframe in days


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

"""
plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('Monte Carlo Simulation of a stock portfolio')
plt.show()
"""

# Calculate Value at risk (VaR)
def mcVaR(returns, alpha=5):
    """ Input: pandas series of returns
        Output: percentile on return distribution to a given confidence level alpha"""
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    else:
        raise TypeError("Expected a pandas data series.")


# Calculate Conditional Value at Risk (CVaR)
def mcCVaR(returns, alpha=5):
    """ Input: pandas series of returns
        Output: CVaR or expected shortfall to a given confidence level alpha"""
    if isinstance(returns, pd.Series):
        belowVar = returns <= mcVaR(returns, alpha=alpha)
        return returns[belowVar].mean()
    else:
        raise TypeError("Expected a pandas data series.")

portfolio_results = pd.Series(portfolio_sims[-1,:])

VaR = initialPortfolio - mcVaR(portfolio_results, alpha=5)
CVaR = initialPortfolio - mcCVaR(portfolio_results, alpha=5)


print('VaR ${}'.format(round(VaR, 2)))
print('CVaR ${}'.format(round(CVaR, 2)))


# Create Plotly figure for portfolio simulation
fig = go.Figure()

for sim in range(mc_sims):
    fig.add_trace(go.Scatter(x=np.arange(T), y=portfolio_sims[:, sim],
                             mode='lines',
                             name=f'Simulation {sim+1}'))

fig.update_layout(title='Monte Carlo Simulation of a stock portfolio',
                  xaxis_title='Days',
                  yaxis_title='Portfolio Value ($)',
                  showlegend=True,
                  plot_bgcolor = 'lightblue',  # Set background color
                  paper_bgcolor = 'white',  # Set background color outside the plot
                  margin = dict(l=50, r=50, t=80, b=50),  # Adjust margins
                  xaxis = dict(showline=True, linewidth=2, linecolor='white'),  # Customize x-axis
                  yaxis = dict(showline=True, linewidth=2, linecolor='white'))  # Customize y-axis

#fig.show()
