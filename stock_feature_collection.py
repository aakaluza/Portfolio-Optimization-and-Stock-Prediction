from yahoo_fin.stock_info import get_data, get_quote_table
import yfinance as yf
import pandas as pd

path_to_stock_tickers = "/Users/logandrawdy/Downloads/nasdaq-listed.csv"
#read in the csv file
stock_tickers = pd.read_csv(path_to_stock_tickers)
#convert the csv file to a list
list_of_ticks = stock_tickers["Symbol"].tolist()

start_date = '11/03/2018'
end_date = '11/03/2023'
#S&P 500 index
market_index_symbol = "^GSPC"
market_data = get_data(market_index_symbol, start_date, end_date, True, "1d")
market_adj_close = market_data['adjclose']
#market percent change
market_adj_close_pct_change = market_adj_close.pct_change() 
#market average return
market_avg_return = market_adj_close_pct_change.mean() * 252
#market variance
market_variance = market_adj_close_pct_change.var()
#long running average of 10 year treasury yield
risk_free_rate = 0.0425
#new pandas dataframe for all the features
features_df = pd.DataFrame(columns = ["tick", "Average Return", "Volatility", "Beta", "CAPM", "Sharpe Ratio", "Average Volume", "PE Ratio", "EPS"])

for tick in list_of_ticks:
    try:
        quote_table = get_quote_table(tick)
    except:
        continue
    #get all data from yahoo finance that has enough information to be used
    try:
        data = get_data(tick, start_date, end_date, True, "1d")
    except:
        continue
    if data.isna().values.any():
        continue
    #get only the adjusted close column
    adj_close = data['adjclose']
    #get the percent change of the adjusted close
    adj_close_pct_change = adj_close.pct_change()
    #get the average return of the adjusted close per year
    avg_return = adj_close_pct_change.mean() * 252
    #get the covariance of the adjusted close and the market
    cov = adj_close_pct_change.cov(market_adj_close_pct_change)
    #get volatility
    volatility = adj_close_pct_change.std()
    #get beta
    beta = cov/market_variance
    #get CAPM
    capm = risk_free_rate + beta*(market_avg_return - risk_free_rate)
    #get sharpe ratio
    sharpe_ratio = (capm - risk_free_rate)/volatility
    #get trading avg volume over 3 months
    try:
        average_volume = quote_table["Volume"]
    except:
        continue
    #get PE ratio
    pe_ratio = quote_table["PE Ratio (TTM)"]
    if pe_ratio == None:
        continue
    #get EPS
    try:
        eps = quote_table["EPS (TTM)"]
    except:
        continue
    if eps == None:
        continue
    #features to be used in the model
    features = [tick, avg_return, volatility, beta, capm, sharpe_ratio, average_volume, pe_ratio, eps]
    #append to the dataframe
    features_df.loc[len(features_df)] = features

#print the dataframe
print(features_df)
#save the dataframe as a csv file
#change tick to be the index
features_df = features_df.set_index("tick")
features_df.to_csv("stock_features.csv")
    
    


