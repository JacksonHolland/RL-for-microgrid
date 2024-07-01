import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def convert_array_columns(df, columns):
    for column in columns:
        df[column] = df[column].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
    return df

# Assume features_df is the DataFrame with extracted features
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
normalized_features = scaler.fit_transform(features_df)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=0)
clusters = kmeans.fit_predict(normalized_features)

# Add cluster labels to the features DataFrame
features_df['Cluster'] = clusters

# Save the clustered DataFrame
features_df.to_csv('trajectory_features_with_clusters.csv', index=False)

# Evaluate the clustering performance
silhouette_avg = silhouette_score(normalized_features, clusters)
davies_bouldin_avg = davies_bouldin_score(normalized_features, clusters)
calinski_harabasz_avg = calinski_harabasz_score(normalized_features, clusters)

print(f'Silhouette Score: {silhouette_avg}')
print(f'Davies-Bouldin Score: {davies_bouldin_avg}')
print(f'Calinski-Harabasz Score: {calinski_harabasz_avg}')

# Optional: pair-plot
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# # Load the DataFrame with cluster labels
# features_df = pd.read_csv('trajectory_features_with_clusters.csv')

# # Select a subset of features to visualize
# selected_features = ['cumulative_reward', 'cumulative_cost', 'reward_mean', 'reward_var', 'Cluster']

# # Create a pair plot
# pair_plot = sns.pairplot(features_df[selected_features], hue='Cluster', palette='viridis')
# pair_plot.fig.suptitle('Pair Plot of Selected Features Colored by Cluster', y=1.02)

# # Save the plot as an image file
# pair_plot.savefig('pair_plot_clusters.png')
# plt.show()

# Optional: 2D Scatter Plot with PCA:
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the DataFrame with cluster labels
features_df = pd.read_csv('trajectory_features_with_clusters.csv')

# Normalize the features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features_df.drop(columns=['Cluster']))

# Apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)
pca_components = pca.fit_transform(normalized_features)

# Create a DataFrame with PCA components and cluster labels
pca_df = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = features_df['Cluster']

# Plot the PCA components
plt.figure(figsize=(10, 7))
pca_plot = sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette='viridis', data=pca_df)
plt.title('2D PCA Scatter Plot of Clusters')

# Save the plot as an image file
plt.savefig('pca_scatter_plot_clusters.png')
plt.show()
