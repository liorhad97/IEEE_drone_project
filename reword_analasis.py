import torch
from api import API  # Ensure API class is imported
import time
import matplotlib.pyplot as plt # Added import

class Reward:
    def __init__(self, api_client: API, accel_threshold=0.1, gyro_threshold=0.1, teta_threshold=0.1, 
                 num_samples=4, sample_delay=0.001):
        self.api_client = api_client  # Store the API client instance
        self.accel_threshold = accel_threshold
        self.gyro_threshold = gyro_threshold
        self.teta_threshold = teta_threshold
        self.num_samples = num_samples
        self.sample_delay = sample_delay
        
        # Track previous rewards for smoother transition
        self.prev_reward = 0.0
        self.reward_history = []
        self.sensors_reward_history = [] # New
        self.time_reward_history = []    # New
        self.target_reward_history = []  # New
        self.people_reward_history = []  # New
        
        # Reward smoothing factor (0 = no smoothing, 1 = complete smoothing)
        self.reward_smoothing = 0.7
        
        # Timer to track episode length
        self.start_time = time.time()
        self.episode_timer = 0
    
    def get_reward_sensors_aspect(self, data_points):
        """Calculate reward based on drone stability - more continuous reward signal"""
        if not data_points:
            return -1.0  # Penalty if no data points
        
        # Extract data into tensors efficiently
        accel_tensor = torch.stack([data.accel for data in data_points])
        gyro_tensor = torch.stack([data.gyro for data in data_points])
        teta_tensor = torch.stack([data.teta for data in data_points])
        
        # Calculate deviations in one step
        accel_deviation = torch.std(accel_tensor).item()
        gyro_deviation = torch.std(gyro_tensor).item()
        teta_deviation = torch.std(teta_tensor).item()
        
        # Calculate continuous rewards based on deviation from threshold
        # The closer to 0, the better the reward (more stable)
        accel_reward = max(0, 1.0 - (accel_deviation / self.accel_threshold))
        gyro_reward = max(0, 1.0 - (gyro_deviation / self.gyro_threshold))
        teta_reward = max(0, 1.0 - (teta_deviation / self.teta_threshold))
        
        # Combined stability reward - weighted average
        stability_reward = (accel_reward * 0.3 + gyro_reward * 0.4 + teta_reward * 0.3) 
        
        # Scale to appropriate range
        scaled_reward = stability_reward - 0.5  # Range from -0.5 to 0.5
        
        return scaled_reward
    
    def get_reward_time_aspect(self, data_points):
        """Time-based reward to encourage faster completion"""
        # Update episode timer
        current_time = time.time()
        self.episode_timer = current_time - self.start_time
        
        # Implement decreasing reward over time to encourage faster completion
        # Base small negative reward
        base_time_penalty = -0.01
        
        # Additional penalty that increases with time (capped)
        time_factor = min(1.0, self.episode_timer / 300.0)  # Cap at 5 minutes
        additional_penalty = -0.02 * time_factor
        
        return base_time_penalty + additional_penalty

    def get_reward_find_target_aspect(self, data_points):
        """Calculate reward based on whether the target person is found"""
        # Uses the API client instance
        if not self.api_client:
            print("Reward Error: API client not initialized in Reward class.")
            return 0.0
            
        found = self.api_client.is_drone_find_objective()
        
        if found:
            # Big bonus for finding target, plus time efficiency bonus
            time_efficiency_bonus = max(0, 30 - min(30, self.episode_timer / 10))
            
            # Reset episode timer for next attempt
            self.start_time = time.time()
            self.episode_timer = 0
            
            return 30.0 + time_efficiency_bonus
        else:
            # Potential reward can be adjusted based on proximity to target if available
            return 0.0
        
    def get_reward_people_aspect(self, data_points):
        """Calculate reward based on the number of people detected"""
        # Uses the API client instance
        if not self.api_client:
            print("Reward Error: API client not initialized in Reward class.")
            return 0.0
            
        num_people = self.api_client.get_num_people()
        
        if num_people > 0:
            # Increasing returns for first few people, diminishing returns after that
            if num_people <= 3:
                return float(num_people * 1.5)  # Higher reward for initial detections
            else:
                return 4.5 + float((num_people - 3) * 0.5)  # Diminishing returns
        else:
            return -0.1  # Small penalty for not detecting any people
        
    def get_reward(self):
        """Calculate the total reward based on multiple aspects with smoothing"""
        data_points = []
        if not self.api_client:
            print("Reward Error: API client not initialized in Reward class for get_reward.")
            return -10.0  # Reduced penalty to avoid extreme values

        try:
            for _ in range(self.num_samples):
                # Use the API client instance to get data
                point = self.api_client.get_sensors_data_from_api()
                if point is not None:  # Only append valid data points
                    data_points.append(point) 
                else:
                    print("Warning: get_sensors_data_from_api() returned None, skipping this sample.")
                if _ < self.num_samples - 1:  # Don't sleep after the last sample
                    time.sleep(self.sample_delay)
        except Exception as e:
            print(f"Error fetching drone data for reward calculation: {e}")
            return -10.0  # Reduced penalty

        if not data_points:
             print("Warning: No valid data points collected for reward calculation.")
             return -10.0  # Reduced penalty

        # Pass the same data to all reward aspects
        sensors_reward = self.get_reward_sensors_aspect(data_points)
        time_reward = self.get_reward_time_aspect(data_points)
        target_reward = self.get_reward_find_target_aspect(data_points)
        people_reward = self.get_reward_people_aspect(data_points)

        # Calculate weighted reward sum
        raw_reward = (
            sensors_reward * 0.4 +  # Stability is important
            time_reward * 0.1 +     # Time pressure is less important
            target_reward * 0.3 +   # Finding the target is important
            people_reward * 0.2     # Finding people is moderately important
        )
        
        # Apply smoothing to prevent drastic reward changes
        smoothed_reward = (self.reward_smoothing * self.prev_reward + 
                           (1 - self.reward_smoothing) * raw_reward)
        
        # Apply clipping to prevent extremely large values
        final_reward = max(-10.0, min(40.0, smoothed_reward))
        
        # Store for next iteration smoothing
        self.prev_reward = final_reward
        
        # Store in history for potential debugging/visualization
        self.reward_history.append(final_reward)
        self.sensors_reward_history.append(sensors_reward)
        self.time_reward_history.append(time_reward)
        self.target_reward_history.append(target_reward)
        self.people_reward_history.append(people_reward)

        history_cap = 1000
        if len(self.reward_history) > history_cap:  # Prevent unbounded growth
            self.reward_history.pop(0)
            self.sensors_reward_history.pop(0) # Keep synchronized
            self.time_reward_history.pop(0)
            self.target_reward_history.pop(0)
            self.people_reward_history.pop(0)
            
        # Log occasionally
        if len(self.reward_history) % 50 == 0:
            avg_recent_reward = sum(self.reward_history[-50:]) / min(50, len(self.reward_history))
            print(f"Average recent reward: {avg_recent_reward:.4f}")
        
        return final_reward

    def plot_reward_components(self):
        """Plots different reward components over time on separate subplots within the same figure."""
        if not self.reward_history: # Check if there's any data
            print("No reward data to plot.")
            return

        timesteps = range(len(self.reward_history))

        # Create a figure with 3 subplots, sharing the x-axis
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True) # 3 rows, 1 column
        
        fig.suptitle('Reward Components Over Time', fontsize=16) # Main title for the figure

        # Plot Sensors Reward
        if self.sensors_reward_history:
            axs[0].plot(timesteps, self.sensors_reward_history, label='Sensors Reward', color='blue')
            axs[0].set_ylabel('Sensors Reward Value')
            axs[0].set_title('Sensors Reward Component')
            axs[0].legend()
            axs[0].grid(True)
        else:
            axs[0].set_title('Sensors Reward Component (No data)')

        # Plot Time Reward
        if self.time_reward_history:
            axs[1].plot(timesteps, self.time_reward_history, label='Time Reward', color='green')
            axs[1].set_ylabel('Time Reward Value')
            axs[1].set_title('Time Reward Component')
            axs[1].legend()
            axs[1].grid(True)
        else:
            axs[1].set_title('Time Reward Component (No data)')

        # Plot Target Reward
        if self.target_reward_history:
            axs[2].plot(timesteps, self.target_reward_history, label='Target Reward', color='red')
            axs[2].set_ylabel('Target Reward Value')
            axs[2].set_title('Target Reward Component')
            axs[2].legend()
            axs[2].grid(True)
        else:
            axs[2].set_title('Target Reward Component (No data)')
        
        # Common X-axis label
        plt.xlabel('Timestep')
        
        # Adjust layout to prevent overlapping titles/labels
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for suptitle
        
        plt.show()