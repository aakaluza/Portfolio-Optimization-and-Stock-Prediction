import pandas as pd
#load in csv file
path_to_csv = "/Users/logandrawdy/Documents/GA Tech- Year Three/CS 4641/Project/stock_features.csv"
path_to_cleaned_csv = "/Users/logandrawdy/Documents/GA Tech- Year Three/CS 4641/Project/stock_features_cleaned.csv"
features_df = pd.read_csv(path_to_csv)
#drop the rows with NaN values
features_df = features_df.dropna()
#delete the index column
del features_df['Unnamed: 0']
#set the index to the tickers
features_df = features_df.set_index('tick')
#save the dataframe to a csv file
features_df.to_csv(path_to_cleaned_csv)

