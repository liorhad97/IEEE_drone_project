import torch
from api import API
import time
class Reward:
    def __init__(self, accel_threshold=0.1, gyro_threshold=0.1, teta_threshold=0.1, 
                 num_samples=4, sample_delay=0.001):
        self.accel_threshold = accel_threshold
        self.gyro_threshold = gyro_threshold
        self.teta_threshold = teta_threshold
        self.num_samples = num_samples
        self.sample_delay = sample_delay
    
    def _get_reward_sensors_aspect(self, data_points):
        """Calculate reward based on drone stability"""
        if not data_points:
            return -1.0
        
        # Extract data into tensors efficiently
        accel_tensor = torch.tensor([data.accel for data in data_points])
        gyro_tensor = torch.tensor([data.gyro for data in data_points])
        teta_tensor = torch.tensor([data.teta for data in data_points])
        
        # Calculate deviations in one step
        accel_deviation = torch.std(accel_tensor).item()
        gyro_deviation = torch.std(gyro_tensor).item()
        teta_deviation = torch.std(teta_tensor).item()
        
        # Determine reward based on configurable thresholds
        if (accel_deviation < self.accel_threshold and 
            gyro_deviation < self.gyro_threshold and 
            teta_deviation < self.teta_threshold):
            reward = 0.5
        else:
            reward = -1.0
        
        return reward
    def _get_reward_time_aspect(self, drone_data=None):
        #each time step, give a negative reward
        return -0.01
    
    def _get_reward_find_target_aspect(self):
        """Calculate reward based on whether the target person is found"""
        found = API.is_drone_find_objective()
        if found:
            return 30
        else:
            return 0

    def get_reward_time_aspect(self, data_points):
        #each time step, give a negative reward
        return -0.01

    def _get_reward_find_target_aspect(self, data_points):
        """Calculate reward based on whether the target person is found"""
        found = API.is_drone_find_objective()

    def get_reward(self, drone_data=None):
        """Calculate the total reward based on multiple aspects"""
        # Collect all data points once for all reward calculations
        data_points = []
        try:
            for _ in range(self.num_samples):
                data_points.append(API.get_drone_data())
                if _ < self.num_samples - 1:  # Don't sleep after the last sample
                    time.sleep(self.sample_delay)
        except Exception as e:
            print(f"Error fetching drone data: {e}")
            data_points = None
        
        # Pass the same data to all reward aspects
        sensors_reward = self.get_reward_sensors_aspect(data_points)
        time_reward = self.get_reward_time_aspect(data_points)
        target_reward = self.get_reward_find_target_aspect(data_points)

        total_reward = sensors_reward + time_reward + target_reward
        return total_reward