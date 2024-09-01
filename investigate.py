import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def load_data(file_path, dataset_name, num_trajectories=1000):
    with h5py.File(file_path, 'r') as f:
        data = f[dataset_name][:]
    if data.shape[0] > num_trajectories:
        indices = np.random.choice(data.shape[0], num_trajectories, replace=False)
        data = data[indices]
    return data

def compute_stats(data):
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data)
    }

def print_stats(name, real_stats, synthetic_stats):
    print(f"\nStatistics for {name}:")
    for stat in ['mean', 'std', 'min', 'max']:
        print(f"{stat.capitalize()}:")
        print(f"  Real: {real_stats[stat]}")
        print(f"  Synthetic: {synthetic_stats[stat]}")
        print(f"  Difference: {np.abs(real_stats[stat] - synthetic_stats[stat])}")

def compare_correlations(real_data, synthetic_data, name):
    real_corr = np.corrcoef(real_data.reshape(-1, real_data.shape[-1]).T)
    synthetic_corr = np.corrcoef(synthetic_data.reshape(-1, synthetic_data.shape[-1]).T)
    corr_diff = np.abs(real_corr - synthetic_corr)
    print(f"\nCorrelation difference for {name}:")
    print(f"Max difference: {np.max(corr_diff):.4f}")
    print(f"Mean difference: {np.mean(corr_diff):.4f}")

def jensen_shannon_distance(p, q):
    # Ensure p and q have the same length
    min_len = min(len(p), len(q))
    p = p[:min_len]
    q = q[:min_len]
    
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    p = p / np.sum(p)
    q = q / np.sum(q)
    m = 0.5 * (p + q)
    return 0.5 * (stats.entropy(p, m) + stats.entropy(q, m))

def compute_js_distance(real_data, synthetic_data):
    real_flat = real_data.reshape(-1, real_data.shape[-1])
    synthetic_flat = synthetic_data.reshape(-1, synthetic_data.shape[-1])
    
    # Ensure real_flat and synthetic_flat have the same number of rows
    min_rows = min(real_flat.shape[0], synthetic_flat.shape[0])
    real_flat = real_flat[:min_rows]
    synthetic_flat = synthetic_flat[:min_rows]
    
    js_distances = []
    for i in range(real_flat.shape[1]):
        js_distances.append(jensen_shannon_distance(real_flat[:, i], synthetic_flat[:, i]))
    return np.mean(js_distances)

# Print statistics
for name, real, synthetic in [("Actions", real_actions, synthetic_actions),
                              ("Observations", real_obs, synthetic_obs),
                              ("Rewards", real_rewards, synthetic_rewards),
                              ("Costs", real_costs, synthetic_costs)]:
    real_stats = compute_stats(real)
    synthetic_stats = compute_stats(synthetic)
    print_stats(name, real_stats, synthetic_stats)
    
    # Kolmogorov-Smirnov test
    ks_statistic, p_value = stats.ks_2samp(real.flatten(), synthetic.flatten())
    print(f"Kolmogorov-Smirnov test: statistic={ks_statistic:.4f}, p-value={p_value:.4f}")

# Correlation Structure
compare_correlations(real_actions, synthetic_actions, "Actions")
compare_correlations(real_obs, synthetic_obs, "Observations")

# Jensen-Shannon Distance
print("\nJensen-Shannon Distances:")
print(f"Actions: {compute_js_distance(real_actions, synthetic_actions):.4f}")
print(f"Observations: {compute_js_distance(real_obs, synthetic_obs):.4f}")
print(f"Rewards: {compute_js_distance(real_rewards, synthetic_rewards):.4f}")
print(f"Costs: {compute_js_distance(real_costs, synthetic_costs):.4f}")

# Dimensionality Check
print("\nDimensionality check:")
print(f"Real actions shape: {real_actions.shape}")
print(f"Synthetic actions shape: {synthetic_actions.shape}")
print(f"Real observations shape: {real_obs.shape}")
print(f"Synthetic observations shape: {synthetic_obs.shape}")
print(f"Real rewards shape: {real_rewards.shape}")
print(f"Synthetic rewards shape: {synthetic_rewards.shape}")
print(f"Real costs shape: {real_costs.shape}")
print(f"Synthetic costs shape: {synthetic_costs.shape}")