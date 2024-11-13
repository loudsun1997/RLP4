import argparse
import time
import xml.etree.ElementTree as ET
from test import test
from environment import Environment
from agent_dqn import Agent_DQN

def parse():
    parser = argparse.ArgumentParser(description="DS551/CS525 RL Project3")
    parser.add_argument('--config_name', required=True, help='name of the configuration to use from the XML file')
    parser.add_argument('--note', required=False, help='a short note to be saved with the training session')
    parser.add_argument('--train', action='store_true', help='whether to train the agent')
    parser.add_argument('--test', action='store_true', help='whether to test the agent')
    parser.add_argument('--record_video', action='store_true', help='whether to record video during testing')
    args = parser.parse_args()
    return args

def load_config(config_name):
    try:
        tree = ET.parse('configs/configs.xml')  # Corrected the path
        root = tree.getroot()
        for config_elem in root.findall('config'):
            if config_elem.get('name') == config_name:
                # Convert the XML element to a dictionary
                config = {}
                for param in config_elem:
                    config[param.tag] = param.text
                return config
        raise ValueError(f"Configuration '{config_name}' not found in the XML file.")
    except FileNotFoundError:
        raise FileNotFoundError("Configuration file 'configs/configs.xml' not found.")
    except ET.ParseError:
        raise ValueError("Error parsing the configuration XML file.")

def run(args):
    start_time = time.time()
    config = load_config(args.config_name)

    if args.note:
        config['note'] = args.note

    # Initialize environment and agent based on config
    env_name = config.get('env_name', 'BreakoutNoFrameskip-v4')
    if args.test:
        render_mode_value = "rgb_array" if args.record_video else None
        env = Environment(env_name, args, atari_wrapper=True, test=True, render_mode=render_mode_value)
        agent = Agent_DQN(env, args, config)
        agent.init_game_setting()
        test(agent, env, total_episodes=100, record_video=args.record_video)
    elif args.train:
        env = Environment(env_name, args, atari_wrapper=True, test=False)
        agent = Agent_DQN(env, args, config)
        print("Starting training...")
        agent.train()
        print(f"Training completed in {time.time() - start_time} seconds.")
    else:
        print("Please specify --train or --test.")

def main():
    args = parse()
    run(args)

if __name__ == "__main__":
    main()