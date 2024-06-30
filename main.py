import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt

# Import data
def get_data(stocks, start, end):
    stockData = yf.download(stocks, start=start, end=end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturn = returns.mean()
    covMatrix = returns.cov()
    return meanReturn, covMatrix

stocklist = ['TCS', 'MRF', 'HAL','BEL']
stocks = [stock + '.NS' for stock in stocklist]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

meanReturn, covMatrix = get_data(stocks, startDate, endDate)

print(meanReturn)
print(covMatrix)

weights = np.random.random(len(meanReturn))
weights /= np.sum(weights)

print(weights)

# Monte Carlo Method
# number of simulations
mc_sims = 1000
T = 1000 # time frame in days

meanM = np.full(shape=(T, len(weights)), fill_value=meanReturn)
meanM = meanM.T

initialPortfolio = 30000

portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)
for m in range(0,mc_sims):
    Z = np.random.normal(size=(T, len(weights)))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanM + np.inner(L,Z)
    portfolio_sims[:, m] = np.cumprod(np.inner(weights,dailyReturns.T) + 1)*initialPortfolio

plt.plot(portfolio_sims)
plt.xlabel('Days')
plt.ylabel('Portfolio Value ($)')
plt.title('Monte Carlo Simulations')
plt.show()

