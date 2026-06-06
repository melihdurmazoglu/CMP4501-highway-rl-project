import gymnasium as gym
import highway_env  # noqa: F401
from src.config import TRAIN_CONFIG, ENV_CONFIG

env = gym.make(TRAIN_CONFIG["env_id"])
env.unwrapped.configure(ENV_CONFIG)
obs, _ = env.reset()

print("Observation shape:", obs.shape)
print("Observation:\n", obs)
print("\nRoad info:", env.unwrapped.road)
print("Ego vehicle:", env.unwrapped.vehicle)