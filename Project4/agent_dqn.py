#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

from agent import Agent
from dqn_model import DQNModel, ActorCriticModel

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

class Agent_DQN:
    def __init__(self, env, args):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor-Critic Model
        self.actor_critic = ActorCriticModel(input_shape=(4, 84, 84), num_actions=env.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=1.5e-4)

        # Replay Buffer
        self.buffer = deque(maxlen=100000)
        self.batch_size = 32

        # Training Parameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / 5000
        self.rewards_history = []
        self.num_episodes = 5000
        self.timestep_limit = 100000

    def choose_action(self, state):
        """Choose an action using the actor's policy."""
        # Rearrange dimensions to [batch_size, channels, height, width]
        state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            policy, _ = self.actor_critic(state_tensor)
        policy = policy.cpu().numpy().flatten()
        action = np.random.choice(len(policy), p=policy)
        return action


    def store_transition(self, state, action, reward, next_state, done):
        """Store transitions in the replay buffer."""
        self.buffer.append((state, action, reward, next_state, done))


    def sample_batch(self):
        """Sample a batch from the replay buffer."""
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32).permute(0, 3, 1, 2).to(self.device),  # [batch, channels, height, width]
            torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device),
            torch.tensor(np.array(next_states), dtype=torch.float32).permute(0, 3, 1, 2).to(self.device),  # [batch, channels, height, width]
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device),
        )



    def update(self):
        """Update the Actor-Critic model using a sampled batch."""
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_batch()

        # Compute policy and value outputs for current states
        policy, values = self.actor_critic(states)
        values = values.squeeze(1)  # Ensure `values` has shape [batch_size]

        # Compute target value
        with torch.no_grad():
            _, next_values = self.actor_critic(next_states)
            next_values = next_values.squeeze(1)  # Ensure `next_values` has shape [batch_size]
            target_values = rewards.squeeze(1) + self.gamma * next_values * (1 - dones.squeeze(1))

        # Advantage computation
        advantages = target_values - values

        # Compute actor loss (policy gradient with advantage)
        action_masks = torch.zeros_like(policy).scatter_(1, actions, 1.0)
        log_probs = torch.log(policy + 1e-8)
        actor_loss = -(log_probs * action_masks).sum(dim=1) * advantages
        actor_loss = actor_loss.mean()

        # Compute critic loss (mean squared error)
        critic_loss = F.mse_loss(values, target_values)  # Shapes now match [batch_size]

        # Total loss
        loss = actor_loss + critic_loss

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




    def train(self):
        """Train the Actor-Critic model."""
        for episode in range(self.num_episodes):
            print(f"Resetting environment for Episode {episode}...")
            state = self.env.reset()  # Shape: (84, 84, 4) from environment
            print(f"Environment reset. Initial state shape: {state.shape}")
            done, truncated = False, False
            total_reward = 0

            while not (done or truncated):
                action = self.choose_action(state)  # state is passed directly
                next_state, reward, done, truncated, _ = self.env.step(action)  # Returns next stacked state
                self.store_transition(state, action, reward, next_state, done)
                state = next_state

                self.update()
                total_reward += reward

            self.rewards_history.append(total_reward)
            
            print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {self.epsilon}")

            # Compute and print average reward over the last 20 episodes
            if len(self.rewards_history) >= 20:
                avg_last_20 = np.mean(self.rewards_history[-20:])
                print(f"Episode {episode}, Reward: {total_reward}, Avg Last 20: {avg_last_20:.2f}, Epsilon: {self.epsilon}")
            else:
                print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {self.epsilon}")

            # Epsilon decay
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

        # Save model
        torch.save(self.actor_critic.state_dict(), "actor_critic.pth")
        self.plot_rewards()


    def plot_rewards(self):
        """Plot the reward curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.rewards_history, label="Rewards")
        plt.xlabel("Episodes")
        plt.ylabel("Total Rewards")
        plt.title("Training Progress")
        plt.legend()
        plt.savefig("./rewards_plot.png")  # Save the plot to a file
        plt.show()  
