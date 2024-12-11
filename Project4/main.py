import argparse
import yaml
import time
from test import test
from environment import Environment
from gym.vector import make as make_vec_env
from gym.vector import SyncVectorEnv
import gym
from agent_dqn import Agent_DQN
from dqn_model import ActorCritic
import os

from gym.vector import SyncVectorEnv

def create_env(env_name, config, atari_wrapper=False, test=False, render_mode=None):
    """
    Factory function to create an instance of the custom Environment.
    """
    def _init():
        return gym.make(env_name, render_mode=render_mode)  # Ensure it creates a valid gym.Env
    return _init

from multiprocessing import Process
import torch.optim as optim

def worker(worker_id, global_model, config, env_name):
    """
    Worker function for A3C to train on independent environments asynchronously.
    """
    
    # Initialize local model and optimizer
    local_model = ActorCritic(
        in_channels=4,
        num_actions=gym.make(env_name).action_space.n
    ).to("cpu")
    optimizer = optim.Adam(global_model.parameters(), lr=config["learning_rate"])

    # Create individual environment for the worker
    env = gym.make(env_name)
    agent = Agent_DQN(env, config, local_model, global_model, optimizer)

    # Set up worker-specific logging
    agent.log_dir = os.path.join(config["log_dir"], f"worker_{worker_id}")
    os.makedirs(agent.log_dir, exist_ok=True)

    # Worker log file
    log_file_path = os.path.join(agent.log_dir, f"worker_{worker_id}_log.txt")
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"Worker {worker_id} started.\n")

    print(f"Worker {worker_id} logging to {log_file_path}")
    
    # Train the local model
    agent.train()

    # Signal completion
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"Worker {worker_id} finished.\n")




def parse():
    parser = argparse.ArgumentParser(description="DS551/CS525 RL Project3")
    parser.add_argument('--config_name', required=True, help='configuration name')
    args = parser.parse_args()
    print('args:', args)
    return args


def load_config(config_name):
    with open('configs.yml', 'r') as file:
        configs = yaml.safe_load(file)
        if config_name in configs:
            print(f"Loaded config with name '{config_name}'")
            return configs[config_name]
        else:
            raise ValueError(f"Config with name '{config_name}' not found in configs.yml")


def run(config):
    start_time = time.time()
    print('config:', config)
    env_name = config.get('env_name', 'ALE/MsPacman-v5')

    if config.get('train_dqn') or config.get('train_dqn_again'):
        # Global shared model
        global_model = ActorCritic(
            in_channels=4,
            num_actions=gym.make(env_name).action_space.n
        )

        global_model.share_memory()  # Enable shared memory for multiprocessing

        # Create workers
        # Start workers
        processes = []
        for worker_id in range(config["num_workers"]):
            print(f"Starting worker {worker_id}")
            p = Process(target=worker, args=(worker_id, global_model, config, env_name))
            p.start()
            processes.append(p)

        # Wait for workers
        for p in processes:
            p.join()
            if p.exitcode != 0:
                print(f"Worker process {p.pid} exited with code {p.exitcode}")


    print('running time:', time.time() - start_time)

    if config.get('test_dqn'):
        record_video = config.get('record_video', False)
        render_mode_value = "rgb_array" if record_video else None

        # Create a single environment for testing
        env_factories = [create_env(env_name, config, atari_wrapper=True, test=True, render_mode=render_mode_value)]
        env = SyncVectorEnv(env_factories)  # Single environment for testing


        agent = Agent_DQN(env, config)

        # Extract model path
        model_path = config.get('model_path', None)
        if not model_path:
            raise ValueError("Model path must be specified in the configuration for testing.")

        # Test the model
        print("Starting testing...")
        agent.test(env, total_episodes=100, model_path=model_path, render=False, record_video=record_video)


    print('running time:', time.time() - start_time)




if __name__ == '__main__':
    args = parse()
    config = load_config(args.config_name)
    run(config)
