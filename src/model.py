import gymnasium as gym
import highway_env  # noqa: F401
from highway_env.vehicle.behavior import IDMVehicle
from stable_baselines3 import PPO
from src.config import TRAIN_CONFIG, ENV_CONFIG

from src.wrapper import HighwayRewardWrapper

IDMVehicle.SPEED_MIN = 10
IDMVehicle.SPEED_MAX = 17
IDMVehicle.target_speeds = [10, 12, 15, 17]

def make_env() -> gym.Env:
    """Highway ortamını oluşturur ve yapılandırır."""
    env = gym.make(TRAIN_CONFIG["env_id"])
    env.unwrapped.configure(ENV_CONFIG)
    env = HighwayRewardWrapper(env)
    return env


def create_model(env: gym.Env) -> PPO:
    """PPO modelini oluşturur."""
    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=TRAIN_CONFIG["learning_rate"],
        n_steps=TRAIN_CONFIG["n_steps"],
        batch_size=TRAIN_CONFIG["batch_size"],
        n_epochs=TRAIN_CONFIG["n_epochs"],
        gamma=TRAIN_CONFIG["gamma"],
        verbose=1,
    )


def load_model(path: str, env: gym.Env) -> PPO:
    """Kaydedilmiş bir modeli yükler."""
    return PPO.load(path, env=env)