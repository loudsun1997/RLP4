#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQNModel(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNModel, self).__init__()
        
        # Input shape should be (4, 84, 84) if using 4-frame stacking with 84x84 resizing
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute the size of the output from the convolutional layers
        # This will depend on the input dimensions and the specific convolution layers
        conv_output_size = self._get_conv_output(input_shape)

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _get_conv_output(self, shape):
        """Helper function to calculate the output size after the conv layers."""
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return int(np.prod(x.size()))

    def forward(self, x):
        # Pass input through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten for the fully connected layers
        x = x.reshape(x.size(0), -1)  # Use reshape instead of view
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # No activation on the output layer

