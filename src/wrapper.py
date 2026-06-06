import gymnasium as gym
import numpy as np
from highway_env.vehicle.controller import ControlledVehicle

class HighwayRewardWrapper(gym.Wrapper):
    """
    Custom reward wrapper for Highway-env to encourage:
    1. High speed (with a smooth gradient).
    2. Keeping a safe distance from leading vehicles (tailgating penalty).
    3. Avoiding unnecessary or frequent lane changes.
    4. Staying on the road.
    5. A large and clear collision penalty.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.last_lane_index = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        ego = self.env.unwrapped.vehicle
        self.last_lane_index = ego.lane_index
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        unwrapped = self.env.unwrapped
        ego = unwrapped.vehicle
        road = unwrapped.road
        
        custom_reward = 0.0
        
        if ego.crashed:
            custom_reward += -25.0
            
        min_speed = 20.0
        max_speed = 30.0
        
        if ego.speed <= min_speed:
            speed_ratio = 0.0
        elif ego.speed >= max_speed:
            speed_ratio = 1.0
        else:
            speed_ratio = (ego.speed - min_speed) / (max_speed - min_speed)
            
        custom_reward += 1.5 * speed_ratio
        
        front_vehicle, _ = road.neighbour_vehicles(ego, ego.lane_index)
        if front_vehicle is not None:
            distance = front_vehicle.position[0] - ego.position[0]
            safe_distance = max(ego.speed * 1.0, 15.0)
            
            if distance < safe_distance:
                closeness = 1.0 - (distance / safe_distance)
                custom_reward += -1.5 * closeness
                
        current_lane_index = ego.lane_index
        if self.last_lane_index is not None and current_lane_index != self.last_lane_index:
            custom_reward += -0.15
        self.last_lane_index = current_lane_index
        
        if current_lane_index is not None:
            lane_id = current_lane_index[2]
            right_lane_bonus = 0.1 * (lane_id / 3.0)
            custom_reward += right_lane_bonus
            
        if not ego.on_road:
            custom_reward += -3.0
            
        info["custom_rewards"] = {
            "collision": -25.0 if ego.crashed else 0.0,
            "speed": 1.5 * speed_ratio,
            "tailgating": -1.5 * (1.0 - (distance / safe_distance)) if (front_vehicle is not None and distance < safe_distance) else 0.0,
            "lane_change": -0.15 if (self.last_lane_index is not None and current_lane_index != self.last_lane_index) else 0.0,
            "right_lane": right_lane_bonus if current_lane_index is not None else 0.0,
            "off_road": -3.0 if not ego.on_road else 0.0,
        }
        
        return obs, custom_reward, terminated, truncated, info

