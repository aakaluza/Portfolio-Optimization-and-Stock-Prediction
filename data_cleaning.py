import pandas as pd
import numpy as np
from scipy import stats
#load in csv file
path_to_csv = "/Users/logandrawdy/Documents/GA Tech- Year Three/CS 4641/Project/stock_features.csv"
path_to_cleaned_csv = "/Users/logandrawdy/Documents/GA Tech- Year Three/CS 4641/Project/stock_features_cleaned.csv"
features_df = pd.read_csv(path_to_csv)
#drop the rows with NaN values
features_df = features_df.dropna()
#delete the index column
del features_df['Unnamed: 0']
#make it a numpy array
features_numpy = features_df.to_numpy()
#calculate zscores
z_scores = stats.zscore(features_numpy[:, 2:].astype(float))
abs_z_scores = np.abs(z_scores)
#get all rows in features that have zscores less than 3
features_numpy = features_numpy[(abs_z_scores < 3).all(axis=1)]
#convert the numpy array back to a dataframe
features_df = pd.DataFrame(features_numpy, columns=features_df.columns)
#set the tick as the index
features_df.set_index('tick', inplace=True)
#save the dataframe to a csv file
features_df.to_csv(path_to_cleaned_csv)

"""
Here we are getting rid of all the rows that have NaN values in them, as well as any outliers that are more than 3 standard deviations
away from the mean for each feature.
"""
