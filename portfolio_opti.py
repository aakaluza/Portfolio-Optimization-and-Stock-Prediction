import pandas as pd
import numpy as np
from pypfopt import plotting
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
import matplotlib.pyplot as plt
from yahoo_fin.stock_info import get_data

#read in the csv file
stocks = pd.read_csv("/Users/logandrawdy/Documents/GA Tech- Year Three/CS 4641/Project/stocks_for_portfolio.csv")
#get only stocks with a sharpe ratio greater than 1
#stocks = stocks.loc[stocks['Sharpe Ratio'] > 1] #adjust these to desire
#get stocks with a average return greater than 0
stocks = stocks.loc[stocks['Average Return'] > .3] #adjust these to desire
ticks = stocks['tick'].tolist()
prices_for_portfolio = pd.DataFrame()
start_date = '11/03/2018'
end_date = '11/03/2023'
for tick in ticks:
    print(tick)
    try:
        market_data = get_data(tick, start_date, end_date, True, "1d")
        market_adj_close = market_data['adjclose']
        prices_for_portfolio[tick] = market_adj_close
    except:
        print("No data for stock")
        continue
mu = mean_historical_return(prices_for_portfolio)
S = CovarianceShrinkage(prices_for_portfolio).ledoit_wolf()
ef = EfficientFrontier(mu, S)
#plot the efficient frontier
weights = ef.min_volatility()
weights = ef.clean_weights()
#delete any weights that are 0
for key in list(weights):
    if weights[key] == 0:
        del weights[key]
performance = ef.portfolio_performance(verbose=True)
#print expected return
print(performance)
#save the weights to a csv file
weights_df = pd.DataFrame.from_dict(weights, orient='index')
print(weights_df)
#weights_df.to_csv("/Users/logandrawdy/Documents/GA Tech- Year Three/CS 4641/Project/weights.csv")
