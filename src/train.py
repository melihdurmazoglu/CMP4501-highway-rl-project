import os
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from src.model import make_env, create_model
from src.utils import save_reward_plot, ensure_dirs
from src.config import TRAIN_CONFIG


def train() -> None:
    """Ajanı eğitir ve checkpoint'leri kaydeder."""
    ensure_dirs()

    print("Parallel environments (4 processes) are being initialized...")
    env = make_vec_env(make_env, n_envs=4, vec_env_cls=SubprocVecEnv)
    model = create_model(env)

    os.makedirs("models", exist_ok=True)

    print("Eğitim başlıyor...")

    half_steps = TRAIN_CONFIG["total_timesteps"] // 2
    model.learn(total_timesteps=half_steps, reset_num_timesteps=True)
    model.save("models/ppo_highway_half")
    print("Yarı eğitimli model kaydedildi.")

    model.learn(total_timesteps=half_steps, reset_num_timesteps=False)
    model.save("models/ppo_highway_full")
    print("Tam eğitimli model kaydedildi.")

    env.close()
    print("Eğitim tamamlandı!")


if __name__ == "__main__":
    train()