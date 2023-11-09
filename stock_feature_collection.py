from yahoo_fin.stock_info import get_data, get_quote_table
import yfinance as yf
import pandas as pd

path_to_stock_tickers_nasdaq = "/Users/logandrawdy/Documents/GA Tech- Year Three/CS 4641/Project/NASDAQ.csv"
path_to_stock_tickers_nyse = "/Users/logandrawdy/Documents/GA Tech- Year Three/CS 4641/Project/NYSE.csv"
path_to_stock_tickers_sgx = "/Users/logandrawdy/Documents/GA Tech- Year Three/CS 4641/Project/SGX.csv"
path_to_stock_tickers_lse = "/Users/logandrawdy/Documents/GA Tech- Year Three/CS 4641/Project/LSE.csv"
list_of_paths = [(path_to_stock_tickers_nasdaq, "NASDAQ"), (path_to_stock_tickers_nyse, "NYSE"), (path_to_stock_tickers_sgx, "SGX"), (path_to_stock_tickers_lse, "LSE")]
#new pandas dataframe for all the features
features_df = pd.DataFrame(columns = ["exchange","tick", "Average Return", "Volatility", "Beta", "CAPM", "Sharpe Ratio", "Average Volume", "EPS", "RSI"])
for path_to_stock_tickers, exchange in list_of_paths:
    #read in the csv file
    stock_tickers = pd.read_csv(path_to_stock_tickers)
    #convert the csv file to a list
    list_of_ticks = stock_tickers["Symbol"].tolist()
    #start and end date for the data
    start_date = '11/03/2018'
    end_date = '11/03/2023'
    if exchange == "NASDAQ":
        market_index_symbol = "^GSPC"
        market_adj_close_pct_change = 1
    elif exchange == "NYSE":
        market_index_symbol = "^GSPC"
        market_adj_close_pct_change = 1
    elif exchange == "SGX":
        market_index_symbol = "^STI"
        market_adj_close_pct_change = 0.74
    elif exchange == "LSE":
        market_index_symbol = "^FTSE"
        market_adj_close_pct_change = 1.23
    else:
        print("Exchange not found")
        continue
    market_data = get_data(market_index_symbol, start_date, end_date, True, "1d")
    market_adj_close = market_data['adjclose']
    #market percent change
    market_adj_close_pct_change = market_adj_close_pct_change * market_adj_close.pct_change() 
    #market average return
    market_avg_return = market_adj_close_pct_change.mean() * 252
    #market variance
    market_variance = market_adj_close_pct_change.var()
    #long running average of 10 year treasury yield
    risk_free_rate = 0.0425

    for tick in list_of_ticks:
        print(tick)
        #get the quote table from yahoo finance
        try:
            quote_table = get_quote_table(tick)
        except:
            continue
        #get all data from yahoo finance that has enough information to be used
        try:
            data = get_data(tick, start_date, end_date, True, "1d")
        except:
            print("No data for stock")
            continue
        if data.isna().values.any():
            print("NaN values in data")
            continue
        #get only the adjusted close column
        adj_close = data['adjclose']
        #get the percent change of the adjusted close
        adj_close_pct_change = adj_close.pct_change()
        #get the average return of the adjusted close per year
        avg_return = adj_close_pct_change.mean() * 252
        if avg_return == float('inf') or avg_return == "inf":
            print("weird edge case")
            continue
        if exchange == "LSE":
            avg_return = avg_return * 1.23
        elif exchange == "SGX":
            avg_return = avg_return * 0.74
        print("avg return:" + str(avg_return))
        #get the covariance of the adjusted close and the market
        cov = adj_close_pct_change.cov(market_adj_close_pct_change)
        print("cov:" + str(cov))
        #get volatility
        volatility = adj_close_pct_change.std()
        print("volatility:" + str(volatility))
        #get beta
        beta = cov/market_variance
        print("beta:" + str(beta))
        #get CAPM
        capm = risk_free_rate + beta*(market_avg_return - risk_free_rate)
        print("capm:" + str(capm))
        #get sharpe ratio
        sharpe_ratio = (capm - risk_free_rate)/volatility
        print("sharpe ratio:" + str(sharpe_ratio))
        #get trading avg volume over 3 months
        try:
            average_volume = quote_table["Avg. Volume"]
            print("average volume:" + str(average_volume))
        except:
            print("no average volume")
            continue
        #get EPS
        try:
            eps = quote_table["EPS (TTM)"]
            print("eps:" + str(eps))
        except:
            print("no eps")
            continue
        if eps == None or eps =="" or eps == "nan":
            print("no eps")
            continue
        #get last 14 days of percent change
        fourteen_day_change = adj_close_pct_change[-14:]
        #seperate gains and losses
        gains = []
        losses = []
        for change in fourteen_day_change:
            if change == None:
                continue
            elif change > 0:
                gains.append(change)
            else:
                losses.append(change)
        #get average gain
        print("gains")
        print(gains)
        print("losses")
        print(losses)
        if sum(gains) == 0:
            avg_gain = 0
        else:
            avg_gain = sum(gains)/len(gains)
        #get average loss
        if sum(losses) == 0:
            avg_loss = 1
        else:
            avg_loss = sum(losses)/len(losses)
        #get RS
        rs = abs(avg_gain/avg_loss)
        #get RSI
        rsi = 100 - (100/(1+rs))
        print("rsi:" + str(rsi))
        #features to be used in the model
        features = [exchange, tick, avg_return, volatility, beta, capm, sharpe_ratio, average_volume, eps, rsi]
        #append to the dataframe
        print(features)
        features_df.loc[len(features_df)] = features
        print("success")

#save the dataframe as a csv file
features_df.to_csv("stock_features.csv")
        
        


