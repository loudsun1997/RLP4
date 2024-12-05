#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):
    """Initialize a deep Q-learning network

    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    This is just a hint. You can build your own structure.
    """

    def __init__(self, in_channels=4, num_actions=4):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
                i.e The number of most recent frames stacked together, here we use 4 frames, which means each state in Breakout is composed of 4 frames.
        num_actions: number of action-value to output, one-to-one correspondence to action in game.

        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 512)  # Adjust the size according to the output of conv layers
        self.fc2 = nn.Linear(512, num_actions)
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # Using the original Deepmind architecture
        # x = x.permute(0, 3, 1, 2).float().to(self.device) / 255.0  # Normalization
        # CNN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Linear layers
        x = F.relu(self.fc1(torch.flatten(x, start_dim=1)))
        x = self.fc2(x)

        ###########################
        return x

class DuelingDQN(nn.Module):
    def __init__(self, input_dim=4, output_dim=4, hidden_dim=512):
        super(DuelingDQN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # input tensor (BS,4,84,84) -> conv1 (BS,32,20,20) -> conv2 (BS,64,9,9) -> conv3 (BS,64,7,7)
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=32, kernel_size=8, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )
        
        # input (BS, 64*7*7) -> linear1 (BS, 512) -> linear2 (BS, 1)
        self.value_stream = nn.Sequential(
            nn.Linear(in_features=64*7*7, out_features=self.hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=1)
        )
        
        # input (BS, 64*7*7) -> linear1 (BS, 512) -> linear2 (BS, output_dim)
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_features=64*7*7, out_features=self.hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=self.output_dim)
        )
        
    def forward(self, state):
        features = self.conv_layer(state)
        # flatten (BS,64,7,7) -> (BS, *) = (BS,64*7*7)
        features = features.reshape(features.size(0), -1)
        
        # values = (BS,1), advantages = (BS, output_dim)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # qvals = (BS, output_dim)
        qvals = values + (advantages - advantages.mean())
        
        return qvals

class SplitQNetwork(nn.Module):
    def __init__(self, input_dim=4, output_dim=4, hidden_dim=512):
        super(SplitQNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Shared convolutional layers
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=32, kernel_size=8, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )

        # Global value stream
        self.global_stream = nn.Sequential(
            nn.Linear(64 * 7 * 7, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)  # Single scalar output for the global value
        )

        # Action-specific stream
        self.action_stream = nn.Sequential(
            nn.Linear(64 * 7 * 7, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)  # One value per action
        )

    def forward(self, state):
        # Extract features using the convolutional layers
        features = self.conv_layer(state)
        features = features.reshape(features.size(0), -1)  # Flatten

        # Compute the global value
        global_value = self.global_stream(features)

        # Compute the action-specific values
        action_values = self.action_stream(features)

        # Combine the global value and action-specific values
        q_values = global_value + action_values

        return q_values

class NoisyLinear(nn.Module):
    """
    Noisy linear layer with learnable noise for exploration.
    Paper: "Noisy Networks for Exploration" (Fortunato et al., 2018)
    """
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.full((out_features, in_features), 0.0))
        self.weight_sigma = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.bias_mu = nn.Parameter(torch.full((out_features,), 0.0))
        self.bias_sigma = nn.Parameter(torch.full((out_features,), sigma_init))

        # Non-learnable parameters
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        std = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-std, std)
        self.bias_mu.data.uniform_(-std, std)

    def forward(self, x):
        # Sample noise
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

        # Compute the noisy weights and biases
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon

        return F.linear(x, weight, bias)

class DuelingDQNNoisy(nn.Module):
    """Updated Dueling DQN with Noisy Networks"""

    def __init__(self, input_dim=4, output_dim=4, hidden_dim=512):
        super(DuelingDQNNoisy, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Shared convolutional layers
        self.conv_layer = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=8, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )

        # Value and Advantage streams with NoisyLinear
        self.value_stream = nn.Sequential(
            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, output_dim)
        )

    def forward(self, state):
        features = self.conv_layer(state).reshape(state.size(0), -1)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class SplitQNetworkNoisy(nn.Module):
    """Updated Split Q-Network with Noisy Networks"""

    def __init__(self, input_dim=4, output_dim=4, hidden_dim=512):
        super(SplitQNetworkNoisy, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Shared convolutional layers
        self.conv_layer = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=8, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )

        # Global value stream
        self.global_stream = nn.Sequential(
            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, 1)
        )

        # Action-specific stream
        self.action_stream = nn.Sequential(
            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, output_dim)
        )

    def forward(self, state):
        features = self.conv_layer(state).reshape(state.size(0), -1)
        global_value = self.global_stream(features)
        action_values = self.action_stream(features)
        q_values = global_value + action_values
        return q_values