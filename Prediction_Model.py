import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyESN import ESN
from yahoo_fin.stock_info import get_data, get_quote_table

#get data from csv
weights_data = pd.read_csv("/Users/logandrawdy/Documents/GA Tech- Year Three/CS 4641/Project/weights.csv")
tickers = weights_data["tickers"].tolist()
weights = weights_data["weights"].tolist()
#for each ticker, get the data from yahoo finance
total_portfolio_daily_return = None
for i in range(1, len(tickers)):
    print(tickers[i])
    data = get_data(tickers[i], "10/03/2018", "11/03/2023", True, "1d")
    market_adj_close = data['adjclose'].tolist()
    market_adj_close = np.array(market_adj_close)
    weighted_return = weights[i] * market_adj_close
    if total_portfolio_daily_return is None:
        total_portfolio_daily_return = weighted_return
    else:
        #check which array is longer
        if len(total_portfolio_daily_return) > len(weighted_return):
            #pad the shorter array with 0s
            weighted_return = np.pad(weighted_return, (0, len(total_portfolio_daily_return) - len(weighted_return)), 'constant')
        elif len(total_portfolio_daily_return) < len(weighted_return):
            #pad the shorter array with 0s
            total_portfolio_daily_return = np.pad(total_portfolio_daily_return, (0, len(weighted_return) - len(total_portfolio_daily_return)), 'constant')
        total_portfolio_daily_return = total_portfolio_daily_return + weighted_return
data = total_portfolio_daily_return

def MSE(prediction, actual):
    return np.mean(np.power(np.subtract(np.array(prediction),actual),2))
    
def run_echo(sr, n, window, trainlen):

    prediction_length = len(data) - trainlen

    esn = ESN(n_inputs = 1,
          n_outputs = 1,
          n_reservoir = 500,
          sparsity=0.2,
          random_state=23,
          spectral_radius=sr,
          noise = n)

    current_set = []
    prediction = None
    for i in range(0, prediction_length):
        pred_training = esn.fit(np.ones(trainlen),data[i:trainlen+i])
        prediction = esn.predict(np.ones(window))
        current_set.append(prediction[0])
    current_set = np.reshape(np.array(current_set),(-1,prediction_length))
    mse = MSE(current_set, data[trainlen:trainlen+prediction_length])
    
    return (mse, current_set, prediction_length, prediction)

error, validation_set, prediction_length, prediction = run_echo(1.2, .005, 5, 1152)
print("MSE: ", error)
print("Prediction: ", prediction)

plt.figure(figsize=(18,8))
plt.plot(range(0, len(data)),data[0:len(data)],'k',label="target system")
plt.plot(range(len(data) - prediction_length, len(data)),validation_set.reshape(-1,1),'r', label="prediction")
lo,hi = plt.ylim()
plt.plot([len(data) - prediction_length,len(data) - prediction_length],[lo+np.spacing(1),hi-np.spacing(1)],'k:')
plt.legend(loc=(0.61,0.12),fontsize='x-large')
plt.xlabel("Time (days)")
plt.ylabel("Price")
plt.show()