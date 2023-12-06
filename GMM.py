import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data
path_to_csv = "stock_features_cleaned.csv"
features_df = pd.read_csv(path_to_csv)
selected_features = features_df.iloc[:, 2:]  # Adjust columns if needed

# Create individual histograms for each feature before PCA fitting
plt.figure(figsize=(10, 8))
for i, column in enumerate(selected_features.columns):
    plt.subplot(3, 3, i+1)  # Adjust subplot layout as per the number of features
    plt.hist(selected_features[column], bins=20)
    plt.title(column)
    plt.tight_layout()
plt.title('Feature Distributions before PCA')
plt.show()

# convert to numpy array
features_numpy = features_df.to_numpy()

# Normalize the data
features_numpy_normalized = stats.zscore(features_numpy[:, 2:].astype(float))

# Perform PCA for GMM
pca = PCA(n_components=3)
pca.fit(features_numpy_normalized)
features_gmm_pca = pca.transform(features_numpy_normalized)

# Step 1: Visualize PCA-transformed features
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(features_gmm_pca[:, 0], features_gmm_pca[:, 1], features_gmm_pca[:, 2])
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
plt.title('PCA-transformed Features')

# Fit GMM to the PCA-transformed features
gmm = GaussianMixture(n_components=6, random_state=0).fit(features_gmm_pca)
gmm_labels = gmm.predict(features_gmm_pca)

# Plot the PCA data for GMM clusters
fig2 = plt.figure()

# 3D Scatter Plot
ax = fig2.add_subplot(221, projection='3d')
ax.scatter(features_gmm_pca[:, 0], features_gmm_pca[:, 1], features_gmm_pca[:, 2], c=gmm_labels)
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')

# 2D Scatter Plot: PCA1 vs PCA2
ax2 = fig2.add_subplot(222)
ax2.scatter(features_gmm_pca[:, 0], features_gmm_pca[:, 1], c=gmm_labels)
ax2.set_xlabel('PCA 1')
ax2.set_ylabel('PCA 2')

# 2D Scatter Plot: PCA1 vs PCA3
ax3 = fig2.add_subplot(223)
ax3.scatter(features_gmm_pca[:, 0], features_gmm_pca[:, 2], c=gmm_labels)
ax3.set_xlabel('PCA 1')
ax3.set_ylabel('PCA 3')

# 2D Scatter Plot: PCA2 vs PCA3
ax4 = fig2.add_subplot(224)
ax4.scatter(features_gmm_pca[:, 1], features_gmm_pca[:, 2], c=gmm_labels)
ax4.set_xlabel('PCA 2')
ax4.set_ylabel('PCA 3')

plt.tight_layout()
plt.show()


#append gmm cluster labels to the original numpy array
features_with_labels = np.column_stack((features_numpy, gmm_labels))
updated_features_df = pd.DataFrame(features_with_labels, columns=list(features_df.columns) + ['gmm_labels'])

# Cluster Analysis
average_return = np.zeros(6)
volatility = np.zeros(6)
beta = np.zeros(6)
capm = np.zeros(6)
sharpe_ratio = np.zeros(6)
rsi = np.zeros(6)

for i in range(6):
    cluster = updated_features_df.loc[updated_features_df['gmm_labels'] == i]
    average_return[i] = cluster['Average Return'].mean()
    volatility[i] = cluster['Volatility'].mean()
    beta[i] = cluster['Beta'].mean()
    capm[i] = cluster['CAPM'].mean()
    sharpe_ratio[i] = cluster['Sharpe Ratio'].mean()
    rsi[i] = cluster['RSI'].mean()

# Find the best and worst clusters for each feature
print((average_return))
print((volatility))
print((beta))
print((capm))
print((sharpe_ratio))
print((rsi))

best_average_return_cluster = np.argmax(average_return)
worst_average_return_cluster = np.argmin(average_return)

best_volatility_cluster = np.argmin(volatility)
worst_volatility_cluster = np.argmax(volatility)

best_beta_cluster = np.argmax(beta)
worst_beta_cluster = np.argmin(beta)

best_capm_cluster = np.argmax(capm)
worst_capm_cluster = np.argmin(capm)

best_sharpe_ratio_cluster = np.argmax(sharpe_ratio)
worst_sharpe_ratio_cluster = np.argmin(sharpe_ratio)

best_rsi_cluster = np.argmin(rsi)
worst_rsi_cluster = np.argmax(rsi)

# Print the results
print("Average Return:")
print(average_return)
print(f"Best cluster: {best_average_return_cluster}, Worst cluster: {worst_average_return_cluster}")

print("\nVolatility:")
print(volatility)
print(f"Best cluster: {best_volatility_cluster}, Worst cluster: {worst_volatility_cluster}")

print("\nBeta:")
print(beta)
print(f"Best cluster: {best_beta_cluster}, Worst cluster: {worst_beta_cluster}")

print("\nCAPM:")
print(capm)
print(f"Best cluster: {best_capm_cluster}, Worst cluster: {worst_capm_cluster}")

print("\nSharpe Ratio:")
print(sharpe_ratio)
print(f"Best cluster: {best_sharpe_ratio_cluster}, Worst cluster: {worst_sharpe_ratio_cluster}")

print("\nRSI:")
print(rsi)
print(f"Best cluster: {best_rsi_cluster}, Worst cluster: {worst_rsi_cluster}")

#save results to new csv 
updated_features_df.to_csv("stocks_features_cleaned_gmm_clusters.csv", index=False)

##############################################
# Modify which clusters are selected for Portfolio HERE:

cluster_2 = updated_features_df.loc[updated_features_df['gmm_labels'] == 2]
cluster_5 = updated_features_df.loc[updated_features_df['gmm_labels'] == 5]
#add the two (or 3 or 4) clusters together
portfolio = cluster_2._append(cluster_5)
#save to csv
portfolio.to_csv("stocks_for_portfolio_gmm.csv")
