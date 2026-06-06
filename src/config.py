TRAIN_CONFIG: dict = {
    "env_id": "highway-v0",
    "total_timesteps": 200_000,
    "learning_rate": 5e-4,
    "n_steps": 256,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "model_save_path": "models/ppo_highway",
}

ENV_CONFIG: dict = {
    "lanes_count": 4,
    "vehicles_count": 20,
    "duration": 60,
    "reward_speed_range": [27, 35],
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "controlled_vehicles": 1,
    "initial_spacing": 2,
    "ego_spacing": 1.5,
    "collision_reward": -5,
    "high_speed_reward": 2.5,
    "right_lane_reward": 0.0,
    "lane_change_reward": 0.0,
}