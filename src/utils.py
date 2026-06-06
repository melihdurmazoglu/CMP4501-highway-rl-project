import os
import matplotlib.pyplot as plt
from typing import List


def save_reward_plot(rewards: List[float], save_path: str = "assets/reward_plot.png") -> None:
    """Eğitim boyunca alınan ödülleri grafik olarak kaydeder."""
    os.makedirs("assets", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward Over Episodes")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Reward plot saved to {save_path}")


def ensure_dirs() -> None:
    """Gerekli klasörlerin var olduğundan emin olur."""
    for folder in ["models", "assets", "videos"]:
        os.makedirs(folder, exist_ok=True)