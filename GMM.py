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
# convert to numpy array
features_numpy = features_df.to_numpy()

# Normalize the data
features_numpy_normalized = stats.zscore(features_numpy[:, 2:].astype(float))

# Perform PCA for GMM
pca = PCA(n_components=3)
pca.fit(features_numpy_normalized)
features_gmm_pca = pca.transform(features_numpy_normalized)

print(features_gmm_pca)

# Fit GMM to the PCA-transformed features
gmm = GaussianMixture(n_components=6, random_state=0).fit(features_gmm_pca)
gmm_labels = gmm.predict(features_gmm_pca)

#append gmm cluster labels to the original numpy array
features_with_labels = np.column_stack((features_numpy, gmm_labels))

updated_features_df = pd.DataFrame(features_with_labels, columns=list(features_df.columns) + ['gmm_labels'])

#save results to new csv 
updated_features_df.to_csv("stocks_features_cleaned_gmm_clusters.csv", index=False)

# Plot the data for GMM clusters
fig = plt.figure()

# 3D Scatter Plot
ax = fig.add_subplot(221, projection='3d')
ax.scatter(features_gmm_pca[:, 0], features_gmm_pca[:, 1], features_gmm_pca[:, 2], c=gmm_labels)
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')

# 2D Scatter Plot: PCA1 vs PCA2
ax2 = fig.add_subplot(222)
ax2.scatter(features_gmm_pca[:, 0], features_gmm_pca[:, 1], c=gmm_labels)
ax2.set_xlabel('PCA 1')
ax2.set_ylabel('PCA 2')

# 2D Scatter Plot: PCA1 vs PCA3
ax3 = fig.add_subplot(223)
ax3.scatter(features_gmm_pca[:, 0], features_gmm_pca[:, 2], c=gmm_labels)
ax3.set_xlabel('PCA 1')
ax3.set_ylabel('PCA 3')

# 2D Scatter Plot: PCA2 vs PCA3
ax4 = fig.add_subplot(224)
ax4.scatter(features_gmm_pca[:, 1], features_gmm_pca[:, 2], c=gmm_labels)
ax4.set_xlabel('PCA 2')
ax4.set_ylabel('PCA 3')

plt.tight_layout()
plt.show()