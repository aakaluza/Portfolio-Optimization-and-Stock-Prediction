import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.cluster import KMeans

#load in csv file
path_to_csv = "/Users/logandrawdy/Documents/GA Tech- Year Three/CS 4641/Project/stock_features_cleaned.csv"
#pandas dataframe
features_df = pd.read_csv(path_to_csv)
#change the exchange column to a color for plotting
#line under was used for original PCA without clustering
#features_df["exchange"].replace({"NASDAQ": 'red', "NYSE": 'blue', "SGX": 'green', "LSE": 'yellow'}, inplace=True)
#convert to numpy array
features = features_df.to_numpy()
#calculate z scores for each feature, excluding the exchange column
z_scores = stats.zscore(features[:,2:].astype(float), axis=0)
#take the absolute value of the z scores
abs_z_scores = np.abs(z_scores)
#append z scores to the features array
features = np.append(features, abs_z_scores, axis=1)
#remove rows with z scores greater than 3
features = features[(features[:,10:] < 3).all(axis=1)]
#remove the z score columns
features = np.delete(features, np.s_[10:], axis=1)
#save as new dataframe
features_no_outliers = pd.DataFrame(data=features, columns=["exchange","tick", "Average Return", "Volatility", "Beta", "CAPM", "Sharpe Ratio", "Average Volume", "EPS", "RSI"])
#get the ticks
ticks = features[:,0]
#get the exchange
exchange = features[:,1]
#delete the ticks and exchange columns
features = np.delete(features, np.s_[:2], axis=1)
#run pca on the features
pca = PCA(n_components=3)
pca.fit(features)
#get the principal components
principal_components = pca.transform(features)
#get the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print(explained_variance_ratio)
#add ticks back to front of principal components
principal_components = np.append(ticks.reshape(-1,1), principal_components, axis=1)
#run kmeans elbow method to find optimal number of clusters
"""
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(principal_components[:,1:])
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
Yielded 5 clusters
"""
#fit kmeans model
kmeans = KMeans(n_clusters=5, random_state=0).fit(principal_components[:,1:])
#create new dataframe with the labels
principal_components_df = pd.DataFrame(data=principal_components, columns=["tick", "PC1", "PC2", "PC3"])
principal_components_df["label"] = kmeans.labels_
#add labels to the dataframe with no outliers
features_no_outliers["label"] = kmeans.labels_
#get the Average Return,Volatility,Beta,CAPM,Sharpe Ratio,Average Volume,EPS,RSI, for each cluster
average_return = np.zeros(5)
volatility = np.zeros(5)
beta = np.zeros(5)
capm = np.zeros(5)
sharpe_ratio = np.zeros(5)
average_volume = np.zeros(5)
eps = np.zeros(5)
rsi = np.zeros(5)
for i in range(5):
    cluster = features_no_outliers.loc[features_no_outliers['label'] == i]
    average_return[i] = cluster['Average Return'].mean()
    volatility[i] = cluster['Volatility'].mean()
    beta[i] = cluster['Beta'].mean()
    capm[i] = cluster['CAPM'].mean()
    sharpe_ratio[i] = cluster['Sharpe Ratio'].mean()
    average_volume[i] = cluster['Average Volume'].mean()
    eps[i] = cluster['EPS'].mean()
    rsi[i] = cluster['RSI'].mean()
#find the best and worst clusters for each feature
"""
print(np.argmax(average_return))
print(np.argmin(volatility))
print(np.argmin(beta))
print(np.argmax(capm))
print(np.argmax(sharpe_ratio))
print(np.argmax(average_volume))
print(np.argmax(eps))
print(np.argmin(rsi))
"""
#save the dataframe to a csv file
principal_components_df.to_csv("/Users/logandrawdy/Documents/GA Tech- Year Three/CS 4641/Project/principal_components.csv")
features_no_outliers.to_csv("/Users/logandrawdy/Documents/GA Tech- Year Three/CS 4641/Project/features_no_outliers.csv")
#plot the 3d pca with the clusters
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(principal_components[:,1], principal_components[:,2], principal_components[:,3], c=kmeans.labels_)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.show()

"""
Based on the results of the PCA, the following clusters have the best performance
regarding maximizing and minimizing the average returns and relative market risk:
Cluster 2: Maximizes Average Return and Average Volume
Cluster 3: Minimizes volatility and RSI, and maximizes CAPM, Sharpe Ratio, and EPS, 
Cluster 0: Minimizes Beta (volatiltity relative to the market)

Therefore, we can focus on clusters 2 and 3 to find stocks that create a well balanced portfolio

Also note that in our PCA PC1 explains 99.9% of the variance, so we can use PC1 as a proxy for the original features.
"""