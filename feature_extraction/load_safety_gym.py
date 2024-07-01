import argparse
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

def load_dataset(agent="Point", task="Goal1"):
    env_name = agent + task
    print("Environment {} loading...".format(env_name))

    id = f'Offline{env_name}Gymnasium-v0'
    env = gym.make(id)

    # Load dataset
    dataset = env.get_dataset()
    print(
        "Loaded data status: ",
        env.observation_space.contains(dataset["observations"][0])
    )
    obs, info = env.reset()
    return dataset


   
