import numpy as np
import pandas as pd
from feature_extraction.load_safety_gym import load_dataset, get_parser

def extract_features(dataset, time_steps_per_trajectory=1000):
    num_trajectories = len(dataset['observations']) // time_steps_per_trajectory
    
    # Reshape data to separate trajectories
    observations = dataset['observations'].reshape((num_trajectories, time_steps_per_trajectory, -1))
    actions = dataset['actions'].reshape((num_trajectories, time_steps_per_trajectory, -1))
    rewards = dataset['rewards'].reshape((num_trajectories, time_steps_per_trajectory))
    costs = dataset['costs'].reshape((num_trajectories, time_steps_per_trajectory))
    terminals = dataset['terminals'].reshape((num_trajectories, time_steps_per_trajectory))
    timeouts = dataset['timeouts'].reshape((num_trajectories, time_steps_per_trajectory))
    
    features = []

    for i in range(num_trajectories):
        trajectory_features = {
            'obs_mean': np.mean(observations[i], axis=0),
            'obs_var': np.var(observations[i], axis=0),
            'act_mean': np.mean(actions[i], axis=0),
            'act_var': np.var(actions[i], axis=0),
            'cumulative_reward': np.sum(rewards[i]),
            'cumulative_cost': np.sum(costs[i]),
            'reward_mean': np.mean(rewards[i]),
            'reward_var': np.var(rewards[i]),
            'cost_mean': np.mean(costs[i]),
            'cost_var': np.var(costs[i]),
            'terminal': terminals[i].any(),
            'timeout': timeouts[i].any(),
            'trajectory_length': len(rewards[i])
        }
        features.append(trajectory_features)
    
    # Convert list of feature dictionaries to a DataFrame
    features_df = pd.DataFrame(features)
    
    return features_df

if __name__ == "__main__":
    args = get_parser().parse_args()
    dataset = load_dataset('Point', 'Goal1')

    # Call the function to extract features
    features_df = extract_features(dataset)
    print(features_df.head())

    # Optional: Save the features to a CSV file for easier inspection
    features_df.to_csv('trajectory_features.csv', index=False)
