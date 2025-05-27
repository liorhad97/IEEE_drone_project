import torch
from api import API  # Ensure API class is imported
import time

class Reward:
    def __init__(self, api_client: API, accel_threshold=0.1, gyro_threshold=0.1, teta_threshold=0.1, 
                 num_samples=4, sample_delay=0.001):
        self.api_client = api_client  # Store the API client instance
        self.accel_threshold = accel_threshold
        self.gyro_threshold = gyro_threshold
        self.teta_threshold = teta_threshold
        self.num_samples = num_samples
        self.sample_delay = sample_delay
    
    def get_reward_sensors_aspect(self, data_points): # Renamed from _get_reward_sensors_aspect
        """Calculate reward based on drone stability"""
        if not data_points:
            return -1.0  # Penalty if no data points
        
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
    
    def get_reward_time_aspect(self, data_points):
        #each time step, give a negative reward
        return -0.01

    def get_reward_find_target_aspect(self, data_points): # data_points unused, kept for signature consistency
        """Calculate reward based on whether the target person is found"""
        # Uses the API client instance
        if not self.api_client:
            print("Reward Error: API client not initialized in Reward class.")
            return 0.0 # Or some other default/error value
        found = self.api_client.is_drone_find_objective() 
        if found:
            return 30
        else:
            return 0

    def get_reward(self): # Removed unused drone_data parameter
        """Calculate the total reward based on multiple aspects"""
        data_points = []
        if not self.api_client:
            print("Reward Error: API client not initialized in Reward class for get_reward.")
            return -100.0 # Example error reward

        try:
            for _ in range(self.num_samples):
                # Use the API client instance to get data
                # IMPORTANT: Ensure get_drone_data() returns a valid SensorsData object or None
                point = self.api_client.get_drone_data()
                if point is not None: # Only append valid data points
                    data_points.append(point) 
                else:
                    # Optionally, log or handle the case where get_drone_data() returns None
                    print("Warning: get_drone_data() returned None, skipping this sample.")
                if _ < self.num_samples - 1:  # Don't sleep after the last sample
                    time.sleep(self.sample_delay)
        except Exception as e:
            print(f"Error fetching drone data for reward calculation: {e}")
            # Return a default highly negative reward or handle error appropriately
            return -100.0 # Example error reward

        if not data_points: # Safeguard if loop finishes with no data (e.g., num_samples = 0 or all samples were None)
             print("Warning: No valid data points collected for reward calculation.")
             return -100.0 

        # Pass the same data to all reward aspects
        sensors_reward = self.get_reward_sensors_aspect(data_points)
        time_reward = self.get_reward_time_aspect(data_points)
        target_reward = self.get_reward_find_target_aspect(data_points)

        total_reward = sensors_reward + time_reward + target_reward
        return total_reward