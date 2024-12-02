import os
from ale_py import ALEInterface
import gymnasium as gym

# Set the ROM path directly
rom_path = "/home/zyang12/hw/RL/p4/RLP4/Project4/Roms/ROMS/ms_pacman.bin"

# Load the ROM using ALEInterface
ale = ALEInterface()
if not os.path.exists(rom_path):
    raise FileNotFoundError(f"ROM not found at {rom_path}")
ale.loadROM(rom_path)
print("ROM loaded successfully!")

# Pass the ROM to Gymnasium (if needed for gym environments)
os.environ["ALE_ROM_PATH"] = "/home/zyang12/hw/RL/p4/RLP4/Project4/Roms/ROMS"

# Create the environment
env = gym.make("ALE/MsPacman-v5")
obs = env.reset()
print("Environment created and reset successfully!")
env.close()