import gymnasium as gym
import numpy as np
from atari_wrapper import make_wrap_atari

class Environment(object):
    def __init__(self, env_name, args, atari_wrapper=False, test=False, render_mode=None):
        print(f"Initializing environment: {env_name}")
        if atari_wrapper:
            clip_rewards = not test
            self.env = make_wrap_atari(env_name, clip_rewards, render_mode=render_mode)
        else:
            self.env = gym.make(env_name, render_mode=render_mode)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space  # Ensure it's a valid gym.Space
        self.metadata = self.env.metadata  # Compatibility with SyncVectorEnv

    def seed(self, seed):
        '''
        Control the randomness of the environment
        '''
        self.env.seed(seed)

    def reset(self):
        '''
        Reset the environment and return the initial observation.
        '''
        observation, _ = self.env.reset()
        return np.array(observation)

    def step(self, action):
        '''
        Execute the given action and return the next observation, reward, done, truncated, and info.
        '''
        if not self.env.action_space.contains(action):
            raise ValueError('Invalid action!')

        observation, reward, done, truncated, info = self.env.step(action)
        return np.array(observation), reward, done, truncated, info

    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        return self.observation_space

    def get_random_action(self):
        return self.action_space.sample()

    def close(self):
        '''
        Close the environment.
        '''
        self.env.close()
