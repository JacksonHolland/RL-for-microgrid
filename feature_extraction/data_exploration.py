import argparse
import pandas as pd
import gymnasium as gym
import dsrl.offline_safety_gymnasium


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent", type=str, default="Point", help="agents = [Point, Car]"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Goal1",
        help="tasks = [Circle1, Circle2, Goal1, Goal2, Button1, Button2, Push1, Push2]"
    )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    # make environments
    env_name = args.agent + args.task
    print("Environment {} loading...".format(env_name))

    id = f'Offline{env_name}Gymnasium-v0'
    env = gym.make(id)

    # load dataset
    dataset = env.get_dataset()
    print(
        "loaded data status: ",
        env.observation_space.contains(dataset["observations"][0])
    )
    obs, info = env.reset()
    # Select 20 consecutive time steps
    start_index = 0  # Change this index to start from a different time step
    end_index = start_index + 20

    # Extract the relevant slices from the dataset
    observations_subset = dataset['observations'][start_index:end_index]
    actions_subset = dataset['actions'][start_index:end_index]
    rewards_subset = dataset['rewards'][start_index:end_index]
    costs_subset = dataset['costs'][start_index:end_index]

    # Create a DataFrame to display the data in a table
    data = {
      'Time Step': list(range(start_index, end_index)),
     'State': list(observations_subset),
     'Action': list(actions_subset),
     'Reward': list(rewards_subset),
      'Cost': list(costs_subset),
    }

    df = pd.DataFrame(data)

    # Display the DataFrame
    print(df.to_string(index=False))

    # Optional: Save the DataFrame to a CSV file for easier inspection
    df.to_csv('trajectory_sample.csv', index=False)
    

        # Reset and interact with environment
    obs, info = env.reset()
    for _ in range(100):
         obs, reward, terminal, truncate, info = env.step(env.action_space.sample())

    print("done")