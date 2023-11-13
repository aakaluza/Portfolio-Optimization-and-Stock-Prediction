import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.cluster import KMeans

#load in csv file
path_to_csv = "stock_features_cleaned.csv"
#pandas dataframe
features_df = pd.read_csv(path_to_csv)
#convert to numpy array
features_numpy = features_df.to_numpy()
#normalize the data
features_numpy_normalized = stats.zscore(features_numpy[:, 2:].astype(float))
#PCA
pca = PCA(n_components=3)
pca.fit(features_numpy_normalized)
features_pca = pca.transform(features_numpy_normalized)
#get the explained variance ratio (showed that 3 components is enough)
#print(pca.explained_variance_ratio_)
#check correlation between components (showed that they are not correlated)
#print(np.corrcoef(features_pca[:,0], features_pca[:,1]))
#print(np.corrcoef(features_pca[:,0], features_pca[:,2]))
#print(np.corrcoef(features_pca[:,1], features_pca[:,2]))
#check orthogonality between components (showed that they are pseudo-orthogonal)
#print(np.dot(features_pca[:,0], features_pca[:,1]))
#print(np.dot(features_pca[:,0], features_pca[:,2]))
#print(np.dot(features_pca[:,1], features_pca[:,2]))
#run kmeans on the data (showed that 6 clusters is enough)
kmeans = KMeans(n_clusters=6, random_state=0).fit(features_pca)
#put labels into dataframe
features_df['labels'] = kmeans.labels_
average_return = np.zeros(6)
volatility = np.zeros(6)
beta = np.zeros(6)
capm = np.zeros(6)
sharpe_ratio = np.zeros(6)
rsi = np.zeros(6)
for i in range(6):
    cluster = features_df.loc[features_df['labels'] == i]
    average_return[i] = cluster['Average Return'].mean()
    volatility[i] = cluster['Volatility'].mean()
    beta[i] = cluster['Beta'].mean()
    capm[i] = cluster['CAPM'].mean()
    sharpe_ratio[i] = cluster['Sharpe Ratio'].mean()
    rsi[i] = cluster['RSI'].mean()
#find the best and worst clusters for each feature
print((average_return))
print((volatility))
print((beta))
print((capm))
print((sharpe_ratio))
print((rsi))
"""
From the clustering, we can see that 
cluster 3 has the highest average return, but at the expense of being significantly more volatile, a very low sharpe ratio, and capm
therefore this cluster should be avoided.
cluster 4 has the second highest average return, and is significantly less volatile than cluster 3, however its sharpe ratio and capm
are still not acceptable for investment.
cluster 5 has the third highest average return, and is significantly less volatile than 3 and 4, has an acceptable sharpe ratio and capm, is slightly
however is slightly more volatile than the market, but also has the closest relative stength to 30, which is the ideal value to buy at.
cluster 0 has the next highest return, but higher volatility than cluster 5, so therefore this cluster is dominated by cluster 5.
cluster 1 has a negative average return, and is more volatile than cluster 5, so it cannot be used for risk reduction.
cluster 2 has a negative average return, but it is minimal, and has the lowest volatility, and is less risk than the market significantly, so this cluster can be
used for risk reduction.

Therefore, we will use cluster 2 and cluster 5 to attempt to build a balanced portfolio.
"""
#save to csv
features_df.to_csv("stock_features_cleaned_with_clusters.csv")
#save clusters wanted for portfolio
cluster_2 = features_df.loc[features_df['labels'] == 2]
cluster_5 = features_df.loc[features_df['labels'] == 5]
#add the two clusters together
portfolio = cluster_2._append(cluster_5)
#save to csv
portfolio.to_csv("stocks_for_portfolio.csv")

#plot the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(features_pca[:,0], features_pca[:,1], features_pca[:,2], c=kmeans.labels_.astype(float))
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, )
ax2.scatter(features_pca[:,0], features_pca[:,1], c=kmeans.labels_.astype(float))
ax2.set_xlabel('PCA 1')
ax2.set_ylabel('PCA 2')

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.scatter(features_pca[:,0], features_pca[:,2], c=kmeans.labels_.astype(float))
ax3.set_xlabel('PCA 1')
ax3.set_ylabel('PCA 3')

fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
ax4.scatter(features_pca[:,1], features_pca[:,2], c=kmeans.labels_.astype(float))
ax4.set_xlabel('PCA 2')
ax4.set_ylabel('PCA 3')

plt.show()
