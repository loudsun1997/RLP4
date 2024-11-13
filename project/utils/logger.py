import os
import json
from collections import deque

class Logger:
	def __init__(self, log_dir, max_entries=1000):
		self.log_dir = log_dir
		os.makedirs(log_dir, exist_ok=True)
		self.log_file = os.path.join(log_dir, 'training_log.json')
		self.max_entries = max_entries
		self.logs = deque(maxlen=max_entries)

	def log(self, episode, reward, performance):
		entry = {
			'episode': episode,
			'reward': reward,
			'performance': performance
		}
		self.logs.append(entry)
		self._save_log()

	def _save_log(self):
		with open(self.log_file, 'w') as f:
			json.dump(list(self.logs), f)

	def load_log(self):
		if os.path.exists(self.log_file):
			with open(self.log_file, 'r') as f:
				self.logs = deque(json.load(f), maxlen=self.max_entries)

	def get_average_reward(self):
		if not self.logs:
			return 0
		return sum(log['reward'] for log in self.logs) / len(self.logs)

	def get_latest_performance(self):
		if not self.logs:
			return None
		return self.logs[-1]['performance']

# Example usage:
# logger = Logger(log_dir='/path/to/logs')
# logger.log(episode=1, reward=10, performance={'accuracy': 0.8})
# avg_reward = logger.get_average_reward()
# latest_performance = logger.get_latest_performance()