import random
import numpy as np
import torch
from collections import deque
from networks.standard_dqn import DQNModel
from buffers.replay_buffer import ReplayBuffer
from utils.logger import Logger
from utils.plotter import plot_rewards
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

class Agent_DQN:
	def __init__(self, env, args, config):
		self.env = env
		self.config = config
		self.config_name = config['name']
		self.config_note = config.get('note', '')

		# Device configuration
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# DQN and target network
		self.q_net = DQNModel(input_shape=(4, 84, 84), num_actions=env.action_space.n).to(self.device)
		self.target_net = DQNModel(input_shape=(4, 84, 84), num_actions=env.action_space.n).to(self.device)
		self.target_net.load_state_dict(self.q_net.state_dict())
		self.target_net.eval()

		# Optimizer and loss function
		self.optimizer = optim.Adam(self.q_net.parameters(), lr=config['learning_rate'])
		self.loss_fn = torch.nn.SmoothL1Loss()  # Huber loss

		# Replay buffer
		self.buffer = ReplayBuffer(maxlen=config['buffer_size'])
		self.batch_size = config['batch_size']

		# Epsilon-greedy parameters
		self.epsilon = config['initial_epsilon']
		self.final_epsilon = config['final_epsilon']
		self.epsilon_decay_rate = (self.epsilon - self.final_epsilon) / config['epsilon_decay_episodes']

		# Learning parameters
		self.gamma = config['gamma']
		self.target_update_freq = config['target_update_freq']
		self.learn_step_counter = 0  # Counts steps to track target network update frequency
		self.num_episodes = config['num_episodes']

		# Rewards tracking
		self.rewards_history = []
		self.moving_avg_window = config['moving_avg_window']

		if args.test_dqn:
			print('Loading trained model...')
			self.q_net.load_state_dict(torch.load('dqn_model.pth'))
			self.q_net.eval()

	def init_game_setting(self):
		"""Prepare anything needed at the start of a new game."""
		pass

	def make_action(self, observation, test=True):
		"""Choose an action using an epsilon-greedy policy."""
		if test or random.random() > self.epsilon:
			state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
			with torch.no_grad():
				q_values = self.q_net(state)
			action = q_values.max(1)[1].item()
		else:
			action = random.randrange(self.env.action_space.n)
		return action

	def push(self, state, action, reward, next_state, done):
		"""Store experiences in replay buffer."""
		self.buffer.append((state, action, reward, next_state, done))

	def replay_buffer(self):
		"""Sample a batch from the replay buffer."""
		batch = self.buffer.sample(self.batch_size)
		states, actions, rewards, next_states, dones = zip(*batch)
		return (
			np.array(states),
			np.array(actions),
			np.array(rewards),
			np.array(next_states),
			np.array(dones)
		)

	def update(self):
		"""Perform a single training step: sample from buffer, compute loss, and update network weights."""
		if len(self.buffer) < 50000:  # Start training after replay buffer has 50,000 samples
			return

		# Sample from replay buffer
		states, actions, rewards, next_states, dones = self.replay_buffer()

		# Convert to tensors and permute to [batch_size, channels, height, width]
		states = torch.tensor(states, dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)
		next_states = torch.tensor(next_states, dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)
		actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
		rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
		dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

		# Current Q values
		q_values = self.q_net(states).gather(1, actions)

		# Target Q values
		with torch.no_grad():
			max_next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
			target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

		# Compute Huber Loss
		loss = self.loss_fn(q_values, target_q_values)

		# Optimize the Q-network
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		# Increment the step counter
		self.learn_step_counter += 1
		if self.learn_step_counter % self.target_update_freq == 0:
			# Update the target network
			self.target_net.load_state_dict(self.q_net.state_dict())

	def train(self):
		"""Train the DQN on Ms. Pac-Man."""
		for episode in range(self.num_episodes):
			state = self.env.reset()
			done = False
			total_reward = 0

			while not done:
				action = self.make_action(state)
				next_state, reward, done, _, _ = self.env.step(action)
				total_reward += reward

				# Store transition in replay buffer
				self.push(state, action, reward, next_state, done)

				# Update network
				self.update()

				state = next_state

			# Track rewards for each episode
			self.rewards_history.append(total_reward)

			# Decay epsilon
			if episode < 20000:
				self.epsilon -= self.epsilon_decay_rate
			self.epsilon = max(self.final_epsilon, self.epsilon)

			# Print progress
			print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {self.epsilon}")

		# Save the model after training
		model_filename = f'dqn_model_{self.config_name}_{self.config_note}.pth'
		torch.save(self.q_net.state_dict(), model_filename)
		print(f"Model saved as {model_filename}.")

		# Plot learning curve
		plot_rewards(self.rewards_history, self.moving_avg_window)

	def evaluate(self, num_episodes=40):
		"""Evaluate the agent's performance over a number of episodes."""
		total_rewards = []
		for episode in range(num_episodes):
			state = self.env.reset()
			done = False
			total_reward = 0

			while not done:
				action = self.make_action(state, test=True)
				next_state, reward, done, _, _ = self.env.step(action)
				total_reward += reward
				state = next_state

			total_rewards.append(total_reward)
			print(f"Evaluation Episode {episode}, Total Reward: {total_reward}")

		average_reward = np.mean(total_rewards)
		print(f"Average Reward over {num_episodes} episodes: {average_reward}")
		return average_reward
