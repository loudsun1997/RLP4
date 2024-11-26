#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Note 10/22/24: orig run with lr = 0.0001 and num_episodes = 3e3, the result was a mean of 1.72
#               - Changing lr to 1.5e-4 and num_episodes = 3e4
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN
from dqn_model import DuelingDQN
from prioritized_replay_buffer import PrioritizedReplayBuffer
from test import test
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN, self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.env = env # Training environment
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Tensor type configuration for compatibility across devices
        self.FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
        self.ByteTensor = torch.cuda.ByteTensor if torch.cuda.is_available() else torch.ByteTensor
        self.Tensor = self.FloatTensor
        
        # Neural network parameters
        print('args:', args)
        self.batch_size = args['batch_size']
        self.gamma = 0.99 # Discount Factor
        self.lr = args['learning_rate'] # Learning Rate
        self.max_memory = int(10e4) # Memory size for replay buffer
        self.num_episodes = int(8e4) # Total episodes for training
        self.decay_rate = args['decay_rate']
        self.training_steps = int(5e3) 
        self.target_update_freq = args['target_update_freq'] # Target network update frequency 
        self.f_skip = 4 # Frame skip for updating every 4 frames
        self.log_rate = 100 # Log progress every 100 episodes
        self.learn_step_counter = 0 # For updating the target network
        self.log_dir = args['log_dir']
        # Replay buffer variables 
        if args['buffer_type'] == 'std_buff':
            self.buffer = deque(maxlen=self.max_memory) # Sample list for the replay buffer 
            self.buffer_type = 'std_buff'
        elif args['buffer_type'] == 'prioritized_buff':
            self.buffer = PrioritizedReplayBuffer(capacity=self.max_memory, alpha=0.6)
            self.buffer_type = 'prioritized_buff'
        # Epsilon-greedy parameters
        self.epsilon = args['initial_epsilon']
        self.epsilon_min = args['epsilon_min']
        self.epsilon_decay = args['epsilon_decay']
        
        # Q-network and target network initialization
        self.dqn_type = args['dqn_type'] # pick between double and dueling or normal 
        if args['dqn_type'] == 'dqn' or args['dqn_type'] == 'double dqn':
            self.policy_net = DQN(in_channels=4, num_actions=self.env.action_space.n).to(self.device)
            self.target_net = DQN(in_channels=4, num_actions=self.env.action_space.n).to(self.device)
        elif args['dqn_type'] == 'dueling dqn' or args['dqn_type'] == 'double dueling dqn':
            self.policy_net = DuelingDQN(input_dim=4, output_dim=self.env.action_space.n).to(self.device)
            self.target_net = DuelingDQN(input_dim=4, output_dim=self.env.action_space.n).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode to save memory
        
        # Optimizer and Huber Loss
        if args['optimizer'] == 'adam':
            self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=self.lr)
        elif args['optimizer'] == 'rmsprop':
            self.optimizer = optim.RMSprop(params=self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.SmoothL1Loss()  # Huber loss
        
        # Rewards tracking
        self.rewards_history = []  # Store total rewards for each episode
        self.moving_avg_window = 30  # Window size for moving average
        self.avg_reward = 100 # 100 episodes to average reward
        
        if args['test_dqn']:
            # you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            self.policy_net.load_state_dict(torch.load(args['model_path'], map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())

            # Set both networks to eval mode to disable gradients and save memory
            self.policy_net.eval()
            self.target_net.eval()


    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        ###########################
        pass
    
    
    def make_action(self, observation, test=False):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # Set epsilon for testing 
        if test: 
            self.epsilon = 0.0125 # Low epsilon for evaluation mode - Paper uses 0.05
            observation = observation/255.0 # Normalizing observation
        else:
            self.epsilon = self.epsilon# max(self.epsilon - self.epsilon_decay, self.epsilon_min)  # Decay epsilon in training
        
        # Reshape and transpose observation to match expected input format (channel-first)
        state = torch.tensor(observation.reshape((1, 84, 84, 4)), dtype=torch.float32, device=self.device)
        state = state.transpose(1, 3).transpose(2, 3)  # Apply transpose after tensor creation
        q_values = self.policy_net(state).data.cpu().numpy()

        # Select action based on the epsilon-greedy policy
        if random.random() > self.epsilon:
            action = np.argmax(q_values) # Choose action with highest Q-value
        else:
            action = random.randint(0, self.env.action_space.n - 1) # Random action for exploration
        ###########################
        return action
    
    def push(self, state, action, reward, next_state, dead, done):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.buffer.append((state, action, reward, next_state, dead, done))
        ###########################
        
        
    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        if self.buffer_type == 'std_buff':
            batch = random.sample(self.buffer, self.batch_size)
            state, action, reward, next_state, dead, done = zip(*batch) # Unpack into components
            
            return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(dead), np.array(done)
        elif self.buffer_type =='prioritized_buff':
            batch = self.buffer.sample(self.batch_size, beta=0.4)  # Adjust beta during training
            states, actions, rewards, next_states, deads, dones, weights, indices = batch
            
            return (np.array(states), np.array(actions), np.array(rewards), 
                    np.array(next_states), np.array(deads), np.array(dones), 
                    np.array(weights), indices)
    def update(self):
        """
        Perform a single training step: sample from buffer, compute loss, and update network weights.
        """
        if len(self.buffer) < 40000:  # Start training after replay buffer has 50,000 samples
            return
        
        if self.buffer_type == 'std_buff':
            # Sample from replay buffer
            states, actions, rewards, next_states, deads, dones = self.replay_buffer()
        elif self.buffer_type =='prioritized_buff':
             states, actions, rewards, next_states, deads, dones, weights, indices = self.replay_buffer()
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        deads = torch.tensor(deads, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        if self.buffer_type == 'prioritized_buff':
            weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # Current Q values for each (state, action) pair
        q_values = self.policy_net(states).gather(1, actions)

        if self.dqn_type == 'dqn':
            with torch.no_grad():
                next_actions = self.target_net(next_states).argmax(1).unsqueeze(1)
                next_q_values = self.target_net(next_states).gather(1, next_actions)
                target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        elif self.dqn_type == 'double dqn' or self.dqn_type == 'double dueling dqn':
            with torch.no_grad():
                next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
                next_q_values = self.target_net(next_states).gather(1, next_actions)
                target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Ensure target_q_values has the same shape as q_values
        target_q_values = target_q_values.view(-1, 1)

        if self.buffer_type == 'prioritized_buff':
            # TD errors
            td_errors = (target_q_values - q_values).detach().cpu().numpy()
            # Update priorities
            self.buffer.update_priorities(indices, td_errors)
            # Compute weighted loss
            loss = (weights * (q_values - target_q_values) ** 2).mean()
        elif self.buffer_type =='std_buff':
            # Compute Huber Loss
            loss = self.loss_fn(q_values, target_q_values)
        
        # Optimize the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Increment the step counter and update the target network periodically
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # Initialize tracking variables
        self.steps_until_done = 0
        self.steps_per_episode = []
        self.rewards = []
        self.mean_rewards = []
        self.best_reward = -float('inf')  # Track the best reward for saving the model
        self.last_saved_reward = 0

        for episode in range(self.num_episodes):
            state = self.env.reset()
            state = state / 255.0 # Normalize the state 
            done = False
            total_reward = 0
            episode_steps = 0
            last_life = 3 
            
            while (not done) and episode_steps < 10000:
                episode_steps += 1
                self.steps_until_done += 1

                # Select an action
                action = self.make_action(state)
                next_state, reward, done, _, life = self.env.step(action)
                
                # Apply reward clipping
                clipped_reward = max(min(reward,1),-1) # clip reward in range 1,-1
                
                # Track lives
                now_life = life['lives'] if 'lives' in life else 0
                dead = now_life < last_life
                last_life = now_life
                next_state = next_state / 255.0  # Normalize next state
                
                # Store transition in replay buffer
                if self.buffer_type == 'prioritized_buff':
                    self.buffer.push(state, action, reward, next_state, dead, done)
                elif self.buffer_type == 'std_buff':
                    self.push(state, action, clipped_reward, next_state, dead, done)

                # Update state and total reward
                state = next_state
                total_reward += reward

                # Track rewards and log progress
                if done: 
                    self.rewards.append(total_reward)
                    self.mean_reward = np.mean(self.rewards[-self.avg_reward:])
                    self.mean_rewards.append(self.mean_reward)
                    self.steps_per_episode.append(episode_steps)

                    #logging rewards and episode in text file
                    if (episode+1) % self.log_rate == 0: 
                        self.log_episode_rewards(episode)
                        self.update_plot() #track progress via plot
                    
                    #save the best model
                    if self.mean_reward > self.best_reward and self.steps_until_done > self.training_steps:
                        self.save_model()

                # Perform optimization step periodically
                if len(self.buffer) >= self.training_steps and self.steps_until_done % self.f_skip == 0:
                    self.update()

                # Update the target network periodically
                if self.steps_until_done % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            # Store total reward for this episode
            self.rewards_history.append(total_reward)

            # Decay epsilon after each episode
            if episode < self.decay_rate:
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
        print("Training completed.")
        # Plot the learning curve
        self.plot_rewards()
        ###########################
    
    # Saving and Logging Functions
    def plot_rewards(self):
        """
        Plot the reward curve, using a moving average to smooth the graph.
        """
        fig = plt.figure(figsize=(12, 5))
        plt.title("Reward vs. Episode")
        plt.xlabel("Episodes")
        plt.ylabel("Average reward in last {} episodes".format(self.moving_avg_window))
        
        # Calculate moving average
        rewards = np.array(self.rewards_history)
        moving_avg = np.convolve(rewards, np.ones(self.moving_avg_window) / self.moving_avg_window, mode='valid')
        
        # Plot moving average reward
        plt.plot(moving_avg)
        plt.show()
        # Save the figure to log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        fig_path = os.path.join(self.log_dir, 'rewards_v_episode_30.png')
        fig.savefig(fig_path)
        plt.close(fig)  # Close the figure to free up memory

    def save_model(self):
        '''
        Save the current policy weight and update rewards
        '''
        checkpoint = os.path.join(self.log_dir, 'model.pth')
        os.makedirs(self.log_dir, exist_ok=True)  # Create log directory if it doesn't exist
        torch.save(self.policy_net.state_dict(), checkpoint)

        #update rewards
        self.last_saved_reward = self.mean_reward
        self.best_reward = max(self.mean_reward, self.best_reward)
        return     

    def update_plot(self):
        """
        Plot reward progress over episodes
        """
        fig = plt.figure()
        plt.clf()
        plt.title('Reward vs. Episode during Training')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(self.rewards, label='Total Reward')
        if len(self.rewards) >= self.avg_reward:
            plt.plot(self.mean_rewards, label='Mean Reward - Last 100 Episodes')
        plt.legend(loc='upper right')
        os.makedirs(self.log_dir, exist_ok=True)
        fig_path = os.path.join(self.log_dir, 'rewards_v_episode.png')
        fig.savefig(fig_path)
        plt.close(fig)  # Close the figure to free up memory

    def log_episode_rewards(self, episode):
        os.makedirs(self.log_dir, exist_ok=True)
    
        # Log detailed episode information
        episode_log_path = os.path.join(self.log_dir, 'episode_logs.txt')
        with open(episode_log_path, 'a') as f:
            f.write("="*20)
            f.write('\n')
            f.write('Current steps = ' + str(self.steps_until_done))
            f.write('\n')
            f.write('Current epsilon = ' + str(self.epsilon))
            f.write('\n')
            f.write('Current episode = ' + str(episode+1))
            f.write('\n')
            f.write('Current mean reward = ' + str(self.mean_reward))
            f.write('\n')
            f.write('Best mean reward = ' + str(self.best_reward))
            f.write('\n')
            f.close()

        #log mean reward
        rewards_log_path = os.path.join(self.log_dir, 'rewards_log.txt')
        with open(rewards_log_path, 'a') as f: 
            f.write(str(self.mean_reward))
            f.write('\n')
            f.close()

        return 