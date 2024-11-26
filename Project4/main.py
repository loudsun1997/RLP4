import argparse
import yaml
import time
from test import test
from environment import Environment


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
        env = Environment(env_name, config, atari_wrapper=True, test=False)
        from agent_dqn import Agent_DQN
        agent = Agent_DQN(env, config)
        print('Training DQN')
        agent.train()

    if config.get('test_dqn'):
        record_video = config.get('record_video', False)
        render_mode_value = "rgb_array" if record_video else None
        env = Environment(env_name, config, atari_wrapper=True, test=True, render_mode=render_mode_value)
        from agent_dqn import Agent_DQN
        agent = Agent_DQN(env, config)
        test(agent, env, total_episodes=100, record_video=record_video)
    print('running time:', time.time() - start_time)


if __name__ == '__main__':
    args = parse()
    config = load_config(args.config_name)
    run(config)
