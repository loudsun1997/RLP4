import argparse
import numpy as np
from environment import Environment
import time
from gymnasium.wrappers.monitoring import video_recorder
from tqdm import tqdm

seed = 11037

def test(agent, env, total_episodes=100, record_video=False):
    rewards = []
    env.seed(seed)

    vid = None  # Initialize vid to None to ensure it's accessible outside the if block
    if record_video:
        vid = video_recorder.VideoRecorder(env=env.env, path="test_vid.mp4")
        
    start_time = time.time()
    
    for _ in tqdm(range(total_episodes)):
        episode_reward = 0.0
        truncated = False
        
        for life in range(3):  # Run each episode for 3 lives (Ms. Pac-Man life limit)
            state = env.reset()
            agent.init_game_setting()
            terminated = False

            # Playing one game (1 life)
            while not terminated and not truncated:
                action = agent.make_action(state, test=True)
                state, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                if record_video:
                    vid.capture_frame()

            # End episode if Ms. Pac-Man has lost all lives
            if terminated and life == 2:  # Final life lost on the third life
                truncated = True

        rewards.append(episode_reward)

    if record_video:
        vid.close()  # Ensure the video recorder is properly closed

    env.close()

    print('Run %d episodes with 3 lives each' % (total_episodes))
    print('Mean:', np.mean(rewards))
    print('Rewards:', rewards)
    print('Running time:', time.time() - start_time)

# Old code to test environment initally
# import gym

# # Initialize the environment without rendering
# env = gym.make("ALE/MsPacman-v5")

# print("Environment initialized:", env)
# print("Action space:", env.action_space)
# print("Observation space:", env.observation_space)

# # Reset the environment and print the initial state
# initial_state = env.reset()
# print("Initial state:", initial_state)

# # Step through the environment with random actions
# is_done = False
# step = 0
# total_reward = 0

# while not is_done:
#     action = env.action_space.sample()  # Choose a random action
#     new_state, reward, terminated, truncated, info = env.step(action)
#     is_done = terminated or truncated
#     # Update total reward and step count
#     total_reward += reward
#     step += 1

#     # Print statements to monitor progress
#     print(f"Step {step}: Action taken: {action}")
#     print(f"New state shape: {new_state.shape} | Reward: {reward} | Done: {is_done}")

# print("Episode finished.")
# print("Total steps taken:", step)
# print("Total reward accumulated:", total_reward)

# env.close()

# had to run pip install -U nptyping ro fix bool issue now just get warning 
