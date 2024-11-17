import os
import json
from collections import deque
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .logger import Logger

class Plotter:
	def __init__(self, log_dir, max_entries=1000):
		self.logger = Logger(log_dir, max_entries)

	def plot_average_rewards(self, window_size=30):
		self.logger.load_log()
		if len(self.logger.logs) < window_size:
			print("Not enough data to plot.")
			return

		rewards = [log['reward'] for log in self.logger.logs]
		avg_rewards = [sum(rewards[i:i+window_size]) / window_size for i in range(len(rewards) - window_size + 1)]

		plt.plot(range(window_size, len(rewards) + 1), avg_rewards)
		plt.xlabel('Episode')
		plt.ylabel('Average Reward')
		plt.title('Average Reward over Last {} Episodes'.format(window_size))
		plt.show()

# Example usage:
# plotter = Plotter(log_dir='/path/to/logs')
# plotter.logger.log(episode=1, reward=10, performance={'accuracy': 0.8})
# plotter.plot_average_rewards(window_size=30)
