

import pandas as pd
import numpy as np
import quandl
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# Plot of E.ON(a big scale energy company in Europe)
# stock prices since beginning of 2019 (up to July)

# prices = quandl.get("FSE/EON_X",
#                     authtoken="5XL4GERkAFeCDBQr2y7J",
#                     start_date='2019-01-01', end_date='2019-07-31'
#                     ).reset_index(drop=False)[['Date', 'Close']]
#
# plt.figure(figsize=(15, 5))
# plt.plot(prices['Date'], prices['Close'])
# plt.xlabel('Days')
# plt.ylabel('Stock Prices, €')
# plt.show()
start_date = '2019-07-01'
end_date = '2019-07-31'
pred_end_date = '2019-08-31'

# We get daily closing stock prices of E.ON for July 2019
S_eon = quandl.get("FSE/EON_X",
                   authtoken="5XL4GERkAFeCDBQr2y7J",
                   start_date=start_date, end_date=end_date
                   ).reset_index(drop=False)[['Date', 'Close']]

# print(S_eon.head())
# print(S_eon.tail())
So = S_eon.loc[S_eon.shape[0] - 1, "Close"]
dt = 1
n_of_wkdays = pd.date_range(start = pd.to_datetime(end_date,
              format = "%Y-%m-%d") + pd.Timedelta('1 days'),
              end = pd.to_datetime(pred_end_date,
              format = "%Y-%m-%d")).to_series(
              ).map(lambda x:
              1 if x.isoweekday() in range(1,6) else 0).sum()
T = n_of_wkdays
N = T / dt
t = np.arange(1, int(N) + 1)
returns = (S_eon.loc[1:, 'Close'] - \
          S_eon.shift(1).loc[1:, 'Close']) / \
          S_eon.shift(1).loc[1:, 'Close']
mu = np.mean(returns)
print(mu)
sigma = np.std(returns)
print(sigma)
scen_size = 2
b = {str(scen): np.random.normal(0, 1, int(N)) for scen in range(1, scen_size + 1)}
W = {str(scen): b[str(scen)].cumsum() for scen in range(1, scen_size + 1)}
# print(W)
drift = (mu - 0.5 * sigma**2) * t
print("drift:\n", drift)
diffusion = {str(scen): sigma * W[str(scen)] for scen in range(1, scen_size + 1)}
print("diffusion:\n", diffusion)
S = np.array([So * np.exp(drift + diffusion[str(scen)]) for scen in range(1, scen_size + 1)])
S = np.hstack((np.array([[So] for scen in range(scen_size)]), S))
print(S)
# Plotting the simulations
plt.figure(figsize=(20, 10))

for i in range(scen_size):
    plt.title("Daily Volatility: " + str(sigma))
    plt.plot(pd.date_range(start=S_eon["Date"].max(),
                           end=pred_end_date, freq='D').map(lambda x:
                                                            x if x.isoweekday() in range(1, 6) else np.nan).dropna(),
             S[i, :])
    plt.ylabel('Stock Prices, €')
    plt.xlabel('Prediction Days')

plt.show()