import numpy as np
import random

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        """
        Initialize the prioritized replay buffer.
        
        Parameters:
        - capacity: Maximum size of the buffer.
        - alpha: Controls the level of prioritization (0 = uniform, 1 = full prioritization).
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0

    def push(self, state, action, reward, next_state, dead, done):
        max_priority = float(max(self.priorities) if self.priorities else 1.0)
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, dead, done))
            self.priorities.append(max_priority)  # Ensure this is a scalar
        else:
            self.buffer[self.position] = (state, action, reward, next_state, dead, done)
            self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of transitions with priorities.
        
        Parameters:
        - batch_size: Number of transitions to sample.
        - beta: Controls the importance-sampling weights.
        """
        # Debugging: Check the contents of priorities
        for p in self.priorities:
            if not isinstance(p, (int, float)):
                print(f"Invalid priority detected: {p} (type: {type(p)})")

        # Calculate probabilities
        scaled_priorities = np.array(self.priorities, dtype=np.float32) ** self.alpha
        sampling_probs = scaled_priorities / sum(scaled_priorities)

        # Sample indices
        indices = random.choices(range(len(self.buffer)), k=batch_size, weights=sampling_probs)
        transitions = [self.buffer[idx] for idx in indices]

        # Calculate importance-sampling weights
        weights = (len(self.buffer) * sampling_probs[indices]) ** -beta
        weights /= max(weights)  # Normalize weights

        states, actions, rewards, next_states, deads, dones = zip(*transitions)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(deads), np.array(dones), 
                np.array(weights), indices)

    def update_priorities(self, indices, td_errors, epsilon=1e-6):
        """
        Update priorities based on TD errors.
        
        Parameters:
        - indices: The indices of the sampled transitions.
        - td_errors: The TD errors of the sampled transitions.
        - epsilon: Small constant to ensure non-zero priority.
        """
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = float(abs(td_error) + epsilon)

    def __len__(self):
        """
        Return the current size of the buffer.
        """
        return len(self.buffer)