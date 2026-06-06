import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from src.model import make_env


def plot_rewards() -> None:
    """Eğitimli modeli çalıştırıp ödülleri ölçer ve grafiğe döker."""
    env = make_env()
    model = PPO.load("models/ppo_highway_full", env=env)

    rewards = []
    n_episodes = 50

    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
        print(f"Episode {episode + 1}/{n_episodes} – Reward: {total_reward:.2f}")

    env.close()

    window = 10
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(12, 5))
    plt.plot(rewards, alpha=0.3, color="steelblue", label="Episode Reward")
    plt.plot(range(window - 1, len(rewards)), smoothed,
             color="steelblue", linewidth=2, label=f"Moving Avg ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Evaluation Reward – Highway-v0 (PPO)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs("assets", exist_ok=True)
    plt.savefig("assets/reward_plot.png", dpi=150)
    plt.close()
    print("Grafik kaydedildi: assets/reward_plot.png")


if __name__ == "__main__":
    plot_rewards()