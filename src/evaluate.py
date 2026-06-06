import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import cv2
import os
from stable_baselines3 import PPO
from src.config import TRAIN_CONFIG, ENV_CONFIG
from src.model import make_env
from src.renderer import CustomRenderer


def record_agent_custom(model_path: str | None, output_path: str,
                         n_episodes: int = 3) -> None:
    """Custom renderer ile ajanı kaydeder."""
    env = make_env()

    if model_path is not None:
        model = PPO.load(model_path, env=env)
    else:
        model = None

    renderer = CustomRenderer()
    frames = []
    total_reward = 0.0

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        episode_reward = 0.0

        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += float(reward)
            done = terminated or truncated
            step += 1

            frame = renderer.render(env, episode_reward, episode + 1, step)
            frames.append(frame)

        total_reward += episode_reward
        print(f"Episode {episode + 1}/{n_episodes} – Reward: {episode_reward:.2f}")

    renderer.close()
    env.close()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    height, width, _ = frames[0].shape
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (width, height),
    )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Video kaydedildi: {output_path}")


import argparse


def create_evolution_video() -> None:
    """3 aşamanın custom render videolarını oluşturur."""
    print("Eğitimsiz ajan kaydediliyor...")
    record_agent_custom(None, "videos/untrained.mp4")

    print("Yarı eğitimli ajan kaydediliyor...")
    record_agent_custom("models/ppo_highway_half", "videos/half_trained.mp4")

    print("Tam eğitimli ajan kaydediliyor...")
    record_agent_custom("models/ppo_highway_full", "videos/full_trained.mp4")

    print("Tüm videolar kaydedildi!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ajan değerlendirme ve video kaydetme betiği.")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "untrained", "half", "full"],
        help="Değerlendirilecek model: 'all' (hepsi), 'untrained' (eğitimsiz), 'half' (yarı eğitimli) veya 'full' (tam eğitimli)"
    )
    args = parser.parse_args()

    if args.model == "all":
        create_evolution_video()
    elif args.model == "untrained":
        print("Eğitimsiz ajan kaydediliyor...")
        record_agent_custom(None, "videos/untrained.mp4")
    elif args.model == "half":
        print("Yarı eğitimli ajan kaydediliyor...")
        record_agent_custom("models/ppo_highway_half", "videos/half_trained.mp4")
    elif args.model == "full":
        print("Tam eğitimli ajan kaydediliyor...")
        record_agent_custom("models/ppo_highway_full", "videos/full_trained.mp4")