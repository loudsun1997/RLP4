import argparse
import os
from environment import Environment
from agent_dqn import Agent_DQN

"""
evaluation.py

This script is used to evaluate a trained reinforcement learning agent using a specified model.

Functions:
	parse() -> argparse.Namespace:
		Parses command-line arguments to get the model name.

	load_model(model_name: str):
		Loads the specified model from the models directory.

	evaluate(args: argparse.Namespace):
		Evaluates the model using the specified configuration.

Example usage:
	python evaluation.py --model_name <model_name>
"""

def parse():
	parser = argparse.ArgumentParser(description="DS551/CS525 RL Project3 Evaluation")
	parser.add_argument('--model_name', required=True, help='name of the model to evaluate')
	args = parser.parse_args()
	return args

def load_model(model_name):
	model_path = os.path.join('/models', model_name)
	if not os.path.exists(model_path):
		raise ValueError(f"Model '{model_name}' not found in the models directory.")
	return model_path

def evaluate(args):
	model_path = load_model(args.model_name)
	env_name = 'ALE/MsPacman-v5'  # Assuming the environment name is fixed for evaluation
	env = Environment(env_name, args, atari_wrapper=True, test=True)
	agent = Agent_DQN(env, args)
	agent.load(model_path)
	agent.evaluate()

if __name__ == '__main__':
	args = parse()
	evaluate(args)