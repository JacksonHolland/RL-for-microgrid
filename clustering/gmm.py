import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
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

def evaluate_gmm_best(n_components, covariance_type, init_params, n_init, max_iter, tol, normalized_features):
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        init_params=init_params,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=42
    )
    clusters = gmm.fit_predict(normalized_features)
    
    silhouette_avg = silhouette_score(normalized_features, clusters)
    davies_bouldin_avg = davies_bouldin_score(normalized_features, clusters)
    calinski_harabasz_avg = calinski_harabasz_score(normalized_features, clusters)
    
    return silhouette_avg, davies_bouldin_avg, calinski_harabasz_avg, clusters

# Function to use the best hyperparameters
def use_best_hyperparameters():
    n_components = 2
    covariance_type = 'spherical'
    init_params = 'kmeans'
    n_init = 10
    max_iter = 100
    tol = 1e-4

    silhouette_avg, davies_bouldin_avg, calinski_harabasz_avg, clusters = evaluate_gmm_best(
        n_components, covariance_type, init_params, n_init, max_iter, tol, normalized_features
    )
    
    print(f'Silhouette Score: {silhouette_avg}')
    print(f'Davies-Bouldin Score: {davies_bouldin_avg}')
    print(f'Calinski-Harabasz Score: {calinski_harabasz_avg}')

    # Add the best cluster labels to the features DataFrame
    features_df['Cluster'] = clusters

    # Save the clustered DataFrame
    features_df.to_csv('trajectory_features_with_gmm_clusters.csv', index=False)

    # Plot pair plot
    selected_features = ['cumulative_reward', 'cumulative_cost', 'reward_mean', 'reward_var', 'Cluster']
    pair_plot = sns.pairplot(features_df[selected_features], hue='Cluster', palette='viridis')
    pair_plot.fig.suptitle('Pair Plot of Selected Features Colored by GMM Cluster', y=1.02)
    pair_plot.savefig('pair_plot_gmm_clusters.png')
    plt.show()

    # PCA scatter plot
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(normalized_features)

    pca_df = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = clusters

    plt.figure(figsize=(10, 7))
    pca_plot = sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette='viridis', data=pca_df)
    plt.title('2D PCA Scatter Plot of GMM Clusters')
    plt.savefig('pca_scatter_plot_gmm_clusters.png')
    plt.show()

# Call the function to use the best hyperparameters
use_best_hyperparameters()
