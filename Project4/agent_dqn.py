import random
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Resize, Grayscale
from collections import deque
from agent import Agent
from dqn_model import ActorCritic
from gym.vector import SyncVectorEnv
import matplotlib.pyplot as plt
from gymnasium.wrappers.monitoring import video_recorder
from tqdm import tqdm


torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class Agent_DQN(Agent):
    def __init__(self, env, config, local_model, global_model, optimizer):
        """
        Initialize the DQN agent with consistent parameters.
        """
        #self.envs = env
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.envs = env
        self.local_model = local_model
        self.global_model = global_model
        self.optimizer = optimizer
        self.device = torch.device("cpu")  # Workers typically use CPU
        self.n_steps = config.get("n_steps", 10)
        self.gamma = config.get("gamma", 0.99)
        self.hx = torch.zeros((1, 512), device=self.device)


        # Hyperparameters
        self.n_steps = config.get("n_steps", 10)
        self.gamma = config.get("gamma", 0.99)
        self.ent_coef = config.get("ent_coef", 0.01)
        self.vf_coef = config.get("vf_coef", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.log_dir = config["log_dir"]
        self.log_rate = config.get("log_rate", 100)

        # Preprocessing
        self.resize = Resize((84, 84))
        self.grayscale = Grayscale()
        self.frame_stack = deque(maxlen=4)  # Maintain a stack of the last 4 frames
        
        

        # Initialize observation
        self.obs, _ = self.envs.reset()
        self.obs = self._preprocess_obs(self.obs)
        self.hx = torch.zeros((self.envs.num_envs, 512), device=self.device)  # Initialize GRU hidden state

        
        print(f"Shape of self.obs after preprocessing: {self.obs.shape}")

        # Tracking variables
        self.rewards = []
        self.mean_rewards = []
        self.best_reward = -float("inf")
        os.makedirs(self.log_dir, exist_ok=True)  # Ensure log directory exists

    def _preprocess_obs(self, obs):
        """
        Preprocess observations: resize, grayscale, and stack frames.
        """
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device) / 255.0  # Normalize and move to the correct device
        obs = obs.permute(0, 3, 1, 2)  # Convert to (batch, channels, height, width)
        obs = self.resize(obs)
        obs = self.grayscale(obs)

        # Stack frames
        if len(self.frame_stack) == 0:
            for _ in range(4):  # Initialize with the same frame
                self.frame_stack.append(obs)
        else:
            self.frame_stack.append(obs)

        # Stack along the channel dimension
        stacked_obs = torch.cat(list(self.frame_stack), dim=1)
        return stacked_obs

    def collect_rollouts(self):
        """
        Collect rollouts from multiple environments.
        """
        obs, actions, rewards, values, dones, hxs = [], [], [], [], []
        for _ in range(self.n_steps):
            obs.append(self.obs)
            policy_logits, value, self.hx = self.model((self.obs, self.hx))
            action = torch.multinomial(F.softmax(policy_logits, dim=-1), num_samples=1)
            
            # Handle Gym's new API with 'terminated' and 'truncated'
            next_obs, reward, terminated, truncated, info = self.envs.step(action.cpu().numpy().squeeze())
            done = terminated | truncated  # Combine termination and truncation as 'done'
            self.hx = self.hx * (1 - done.unsqueeze(1))


            # Record rollouts
            actions.append(action)
            values.append(value)
            hxs.append(self.hx)
            rewards.append(torch.tensor(reward, dtype=torch.float32, device=self.device))
            dones.append(torch.tensor(done, dtype=torch.float32, device=self.device))
            self.obs = self._preprocess_obs(next_obs)  # Preprocess and move to the correct device

        # Stack rollouts
        hxs = torch.stack(hxs)
        obs = torch.stack(obs)
        actions = torch.cat(actions)
        rewards = torch.stack(rewards)
        values = torch.stack(values).squeeze(-1)
        dones = torch.stack(dones)
        return obs, hxs, actions, rewards, values, dones

    def compute_returns_and_advantages(self, rewards, values, dones):
        """
        Compute discounted returns and advantages.
        """
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        next_value = 0
        for t in reversed(range(self.n_steps)):
            next_non_terminal = 1.0 - dones[t]
            next_return = rewards[t] + self.gamma * next_value * next_non_terminal
            next_advantage = next_return - values[t]
            returns[t] = next_return
            advantages[t] = next_advantage
            next_value = values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def update(self, obs, actions, returns, advantages):
        """
        Perform a single A2C update.
        """
        policy_logits, values = self.model(obs)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        log_action_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        policy_loss = -(log_action_probs * advantages).mean()

        entropy = -(log_probs * torch.exp(log_probs)).sum(-1).mean()
        value_loss = F.mse_loss(values.squeeze(-1), returns)

        loss = policy_loss - self.ent_coef * entropy + self.vf_coef * value_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return policy_loss.item(), value_loss.item(), entropy.item()

    def train(self, total_timesteps=1e6):
        timesteps = 0
        print(f"DEBUG: Starting training for worker with log_dir {self.log_dir}", flush=True)
        while True:
            obs, hxs, actions, rewards, values, dones = self.collect_rollouts()
            returns, advantages = self.compute_returns_and_advantages(rewards, values, dones)

            batch_obs = obs.view(-1, *obs.shape[2:])
            batch_actions = actions.view(-1)
            batch_returns = returns.view(-1)
            batch_advantages = advantages.view(-1)

            # Compute gradients for local model
            self.optimizer.zero_grad()
            policy_logits, values = self.local_model(batch_obs)
            log_probs = F.log_softmax(policy_logits, dim=-1)
            log_action_probs = log_probs.gather(1, batch_actions.unsqueeze(-1)).squeeze(-1)
            policy_loss = -(log_action_probs * batch_advantages).mean()
            value_loss = F.mse_loss(values.squeeze(-1), batch_returns)
            loss = policy_loss + 0.5 * value_loss - self.ent_coef * (-(log_probs * torch.exp(log_probs)).sum(-1).mean())
            loss.backward()

            # Clip gradients and apply to global model
            torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), self.max_grad_norm)
            for local_param, global_param in zip(self.local_model.parameters(), self.global_model.parameters()):
                global_param._grad = local_param.grad
            self.optimizer.step()

            # Sync local model with global model
            self.local_model.load_state_dict(self.global_model.state_dict())
            if timesteps % self.log_rate == 0 and len(rewards) >= 100:
                mean_reward = np.mean(rewards[-100:])
                self.mean_rewards.append(mean_reward)
                self.log_progress(timesteps, policy_loss, value_loss, -1)
                print(f"Worker logging: Mean Reward (last 100 episodes): {mean_reward:.2f}")

            timesteps += 1
            


    def log_progress(self, timesteps, policy_loss, value_loss, entropy):
        """
        Log training progress to files and plots.
        """
        log_path = os.path.join(self.log_dir, 'training_logs.txt')
        with open(log_path, 'a') as log_file:
            log_file.write(f"Timesteps: {timesteps}, Policy Loss: {policy_loss:.4f}, "
                           f"Value Loss: {value_loss:.4f}, Entropy: {entropy:.4f}, "
                           f"Mean Reward: {self.mean_rewards[-1]:.2f}\n")
        self.plot_rewards()

    def plot_rewards(self):
        """
        Plot the reward curve, using a moving average to smooth the graph.
        """
        plt.figure(figsize=(12, 5))
        plt.title("Reward vs. Episode")
        plt.xlabel("Episodes")
        plt.ylabel("Mean Reward")
        plt.plot(self.mean_rewards)
        plt.savefig(os.path.join(self.log_dir, "reward_plot.png"))
        plt.close()

    def save_model(self):
        """
        Save the trained model.
        """
        torch.save(self.model.state_dict(), os.path.join(self.log_dir, "a2c_model.pth"))
        
    def test(self, env, total_episodes=100, model_path=None, render=False, record_video=False):
        """
        Test the trained ActorCritic model.
        Args:
            env: The environment to test on.
            total_episodes: Number of episodes to test.
            model_path: Path to the model weights.
            render: Whether to render the environment during testing.
            record_video: Whether to save the testing run as a video.
        """
        self.model.eval()  # Set the model to evaluation mode
        rewards = []

        # Load model if the path is defined
        if model_path:
            print(f"Loading model from: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("Model successfully loaded.")

        vid = None
        if record_video:
            # Use the first environment in the vectorized environment for video recording
            vid = video_recorder.VideoRecorder(env=env.envs[0].env, path="test_vid.mp4")

        for episode in tqdm(range(total_episodes), desc="Testing episodes"):
            episode_reward = 0.0
            obs, _ = env.reset()
            hx = torch.zeros((1, 512), device=self.device)  # Initialize hidden state for single env
            obs = self._preprocess_obs(obs)
            done = False

            while not done:
                with torch.no_grad():
                    policy_logits, _, hx = self.model((obs, hx))
                    action = torch.argmax(F.softmax(policy_logits, dim=-1), dim=-1)

                next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
                done = terminated | truncated

                if render:
                    env.envs[0].render()

                if record_video:
                    vid.capture_frame()

                episode_reward += reward
                obs = self._preprocess_obs(next_obs)

            rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

        if record_video:
            vid.close()

        env.close()

        avg_reward = np.mean(rewards)
        print(f"Average Reward over {total_episodes} episodes: {avg_reward}")

        # Save results to a file
        results_path = os.path.join(self.log_dir, 'test_results.txt')
        with open(results_path, 'a') as log_file:
            log_file.write(f"Test Results - Average Reward: {avg_reward}\n")

        return avg_reward
