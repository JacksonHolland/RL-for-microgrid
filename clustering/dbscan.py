import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def convert_array_columns(df, columns):
    for column in columns:
        df[column] = df[column].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
    return df

# Load the DataFrame with cluster labels
features_df = pd.read_csv('trajectory_features.csv')

# Convert array columns back to numeric arrays
array_columns = ['obs_mean', 'obs_var', 'act_mean', 'act_var']
features_df = convert_array_columns(features_df, array_columns)

# Flatten the array columns into separate numeric columns
for column in array_columns:
    array_df = pd.DataFrame(features_df[column].tolist(), index=features_df.index)
    array_df.columns = [f'{column}_{i}' for i in array_df.columns]
    features_df = features_df.drop(column, axis=1).join(array_df)

# Normalize the features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features_df.drop(columns=['Cluster'], errors='ignore'))

# Apply DBSCAN clustering with adjusted parameters
dbscan = DBSCAN(eps=4.52, min_samples=5)
clusters = dbscan.fit_predict(normalized_features)

# Add cluster labels to the features DataFrame
features_df['Cluster'] = clusters

# Save the clustered DataFrame
features_df.to_csv('trajectory_features_with_dbscan_clusters.csv', index=False)

# Filter out noise points for metrics (optional)
filtered_features = normalized_features[clusters != -1]
filtered_clusters = clusters[clusters != -1]

# Check if there are sufficient clusters to compute metrics
if len(set(filtered_clusters)) > 1:
    silhouette_avg = silhouette_score(filtered_features, filtered_clusters)
    davies_bouldin_avg = davies_bouldin_score(filtered_features, filtered_clusters)
    calinski_harabasz_avg = calinski_harabasz_score(filtered_features, filtered_clusters)
else:
    silhouette_avg = davies_bouldin_avg = calinski_harabasz_avg = float('nan')

print(f'Silhouette Score: {silhouette_avg}')
print(f'Davies-Bouldin Score: {davies_bouldin_avg}')
print(f'Calinski-Harabasz Score: {calinski_harabasz_avg}')

# Visualize the clustering results
# Pair plot
selected_features = ['cumulative_reward', 'cumulative_cost', 'reward_mean', 'reward_var', 'Cluster']
pair_plot = sns.pairplot(features_df[selected_features], hue='Cluster', palette='viridis')
pair_plot.fig.suptitle('Pair Plot of Selected Features Colored by DBSCAN Cluster', y=1.02)
pair_plot.savefig('pair_plot_dbscan_clusters.png')
plt.show()

# PCA scatter plot
pca = PCA(n_components=2)
pca_components = pca.fit_transform(normalized_features)

pca_df = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = clusters

plt.figure(figsize=(10, 7))
pca_plot = sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette='viridis', data=pca_df)
plt.title('2D PCA Scatter Plot of DBSCAN Clusters')
plt.savefig('pca_scatter_plot_dbscan_clusters.png')
plt.show()