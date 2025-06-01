import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
import numpy as np

from data_models.drone_data_model import DroneData
from models.drone_agent_model import DroneModel
from config import hyperparameters as hp
from api import API
from reword_analasis import Reward

import matplotlib.pyplot as plt # Added for plotting

class PPOTrainer:
    def __init__(self, model: DroneModel, device, use_generated_data: bool = True):
        self.model = model.to(device)
        self.device = device
        self.use_generated_data = use_generated_data
        
        # Initialize API with Gaussian distribution parameters for simulation
        # Use normal/Gaussian distributions for all sensor data
        self.api_client = API(
            stream_url=hp.SSE_STREAM_URL,
            simulate=True,
            sim_objective_find_interval_seconds=hp.SIM_OBJECTIVE_FIND_INTERVAL,
            sim_objective_reset_interval_seconds=hp.SIM_OBJECTIVE_RESET_INTERVAL,
            sim_listener_tick_seconds=hp.SIM_LISTENER_TICK,
            sim_gps_drift_scale=hp.SIM_GPS_DRIFT_SCALE,
            sim_accel_range=(-hp.SIM_ACCEL_RANGE, hp.SIM_ACCEL_RANGE),
            sim_gyro_range=(-hp.SIM_GYRO_RANGE, hp.SIM_GYRO_RANGE),
            sim_teta_yaw_range=(-hp.SIM_TETA_YAW_RANGE, hp.SIM_TETA_YAW_RANGE),
            sim_teta_pitch_range=(-hp.SIM_TETA_PITCH_RANGE, hp.SIM_TETA_PITCH_RANGE)
        )
        
        self.reward_calculator = Reward(
            api_client=self.api_client,
            accel_threshold=hp.REWARD_ACCEL_THRESHOLD,
            gyro_threshold=hp.REWARD_GYRO_THRESHOLD,
            teta_threshold=hp.REWARD_TETA_THRESHOLD,
            num_samples=hp.REWARD_NUM_SAMPLES,
            sample_delay=hp.REWARD_SAMPLE_DELAY
        )

        self.actor_optimizer = optim.Adam(self.model.actor_head.parameters(), lr=hp.LEARNING_RATE_ACTOR)
        self.critic_optimizer = optim.Adam(self.model.critic_head.parameters(), lr=hp.LEARNING_RATE_CRITIC)
        
    def _get_current_drone_data(self):
        # Get data from API (simulation or real based on API configuration)
        try:
            # Get sensor data with Gaussian distributions from API
            sensor_data = self.api_client.get_sensors_data_from_api()
            camera_image = self.api_client.get_camera_image_from_api().to(self.device)
            lidar_image = self.api_client.get_lidar_image_from_api().to(self.device)
            
            if sensor_data is not None and camera_image is not None and lidar_image is not None:
                return DroneData(
                    timestamp=time.time(),
                    camera_image=camera_image,
                    lidar_image=lidar_image,
                    gps_data=sensor_data.gps.to(self.device),
                    accelerometer_data=sensor_data.accel.to(self.device),
                    gyroscope_data=sensor_data.gyro.to(self.device),
                    attitude_data=sensor_data.teta.to(self.device)
                )
        except Exception as e:
            print(f"Error fetching data from API: {e}")
        
        # If we reach here, there was an issue with the API data
        print("Warning: Failed to fetch complete data from API. Creating fallback data.")
        
        # Create fallback data with Gaussian distributions
        # Use torch.normal for Gaussian distributions
        camera_img = torch.normal(mean=0.5, std=0.2, size=(3, hp.IMG_SIZE, hp.IMG_SIZE)).to(self.device)
        lidar_img = torch.normal(mean=0.3, std=0.15, size=(1, hp.IMG_SIZE, hp.IMG_SIZE)).to(self.device)
        gps = torch.normal(mean=torch.tensor([32.0, 34.0, 50.0]), std=torch.tensor([0.01, 0.01, 0.5])).to(self.device)
        accel = torch.normal(mean=torch.tensor([0.0]), std=torch.tensor([0.1])).to(self.device)
        gyro = torch.normal(mean=torch.tensor([0.0, 0.0, 0.0]), std=torch.tensor([0.05, 0.05, 0.05])).to(self.device)
        teta = torch.normal(mean=torch.tensor([0.0, 0.0]), std=torch.tensor([0.1, 0.1])).to(self.device)
        
        return DroneData(time.time(), camera_img, lidar_img, gps, accel, gyro, teta)

    def _perform_update(self, state, action, old_log_prob, old_value, reward_tensor, next_state, done_tensor):
        # Get value of next_state for target calculation
        with torch.no_grad():
            _, _, next_value_nograd_full = self.model.act(next_state) 
            next_value_nograd = next_value_nograd_full.squeeze()
            # Ensure next_value_nograd is at least 1D
            if next_value_nograd.ndim == 0:
                 next_value_nograd = next_value_nograd.unsqueeze(0)

        # Calculate target for critic
        target_value = reward_tensor + hp.GAMMA * next_value_nograd * (1.0 - done_tensor)
        
        # Normalize rewards for more stable training
        if reward_tensor.shape[0] > 1:  # Only normalize if batch size > 1
            reward_mean = reward_tensor.mean()
            reward_std = reward_tensor.std() + 1e-8  # Avoid division by zero
            normalized_rewards = (reward_tensor - reward_mean) / reward_std
        else:
            normalized_rewards = reward_tensor
        
        # Squeeze old_value if it's [1] or [1,1] to match target_value shape
        old_value_squeezed = old_value.squeeze()
        if old_value_squeezed.ndim == 0:
            old_value_squeezed = old_value_squeezed.unsqueeze(0)

        # Calculate advantage
        advantage = target_value - old_value_squeezed

        # Evaluate the action taken to get new log_prob, current state value, and entropy
        # Ensure action is in the correct shape for model.evaluate
        new_log_prob, current_value_evaluated, entropy = self.model.evaluate(state, action)
        
        current_value_evaluated_squeezed = current_value_evaluated.squeeze()
        if current_value_evaluated_squeezed.ndim == 0:
            current_value_evaluated_squeezed = current_value_evaluated_squeezed.unsqueeze(0)
        
        if new_log_prob.ndim == 0: # Ensure log_prob is at least 1D
            new_log_prob = new_log_prob.unsqueeze(0)


        # PPO Surrogate Loss for Actor
        # Ensure old_log_prob is detached and has compatible shape
        log_ratio = new_log_prob - old_log_prob.detach()
        ratio = torch.exp(log_ratio)

        surr1 = ratio * advantage.detach() # Detach advantage here
        surr2 = torch.clamp(ratio, 1.0 - hp.CLIP_EPSILON, 1.0 + hp.CLIP_EPSILON) * advantage.detach()
        actor_loss = -torch.min(surr1, surr2).mean()
            
        # Critic Loss
        critic_loss = nn.MSELoss()(current_value_evaluated_squeezed, target_value.detach()) # Detach target_value

        # Total Loss
        loss = actor_loss + hp.CRITIC_LOSS_COEFF * critic_loss - hp.ENTROPY_COEFF * entropy

        # Perform optimization step
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5) 
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        return actor_loss.item(), critic_loss.item(), entropy.item()

    def run_training_loop(self):
        print(f"Starting simplified online training on {self.device}...")
        self.model.train()
        
        current_drone_state = self._get_current_drone_data()

        # --- Plotting setup ---
        plt.ion() # Turn on interactive mode
        fig, ax = plt.subplots()
        actor_losses = []
        critic_losses = []
        total_losses = []
        steps_list = []

        line_actor, = ax.plot(steps_list, actor_losses, label='Actor Loss')
        line_critic, = ax.plot(steps_list, critic_losses, label='Critic Loss')
        line_total, = ax.plot(steps_list, total_losses, label='Total Loss')
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Loss")
        ax.set_yscale('linear') # Set y-axis to linear scale
        ax.set_title("Training Loss Over Time (Log Scale)")
        ax.legend()
        # --- End Plotting setup ---

        for step in range(hp.MAX_TRAINING_STEPS):
            # Act: Get action, log_prob, and value from the current policy
            with torch.no_grad():
                action_raw, log_prob_raw, value_raw = self.model.act(current_drone_state)
            
            # Ensure action, log_prob, value are correctly shaped tensors
            action = action_raw.squeeze() 
            if action.ndim == 0: # if it became a scalar, unsqueeze to 1D for evaluate
                action = action.unsqueeze(0)
            # If action is still scalar after squeeze (e.g. discrete action space)
            # and model.evaluate expects a 1D tensor, this should be fine.
            # If model.evaluate expects a specific shape (e.g. [1, N]), adjust accordingly.

            log_prob = log_prob_raw.squeeze()
            if log_prob.ndim == 0:
                 log_prob = log_prob.unsqueeze(0)

            value = value_raw.squeeze() # This is V(s_t)
            if value.ndim == 0:
                 value = value.unsqueeze(0)

            # Interact with environment
            next_drone_state = self._get_current_drone_data()
            reward_val = self.reward_calculator.get_reward() # scalar
            done_val = self.api_client.is_drone_find_objective() # boolean

            reward_tensor = torch.tensor([reward_val], dtype=torch.float32, device=self.device)
            done_tensor = torch.tensor([float(done_val)], dtype=torch.float32, device=self.device)

            # Perform update using the collected transition
            actor_loss, critic_loss, entropy = self._perform_update(
                current_drone_state, 
                action,         
                log_prob,       
                value,          
                reward_tensor, 
                next_drone_state, 
                done_tensor
            )
            
            current_drone_state = next_drone_state
            if done_val:
                print(f"Objective found or episode ended at step {step}. Resetting.")
                current_drone_state = self._get_current_drone_data()
            
            # --- Update Plot Data ---
            total_loss_value = actor_loss + hp.CRITIC_LOSS_COEFF * critic_loss # Entropy is a bonus, not part of primary loss for plotting
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            total_losses.append(total_loss_value)
            steps_list.append(step)

            if step % 10 == 0: # Update plot every 10 steps
                line_actor.set_xdata(steps_list)
                line_actor.set_ydata(actor_losses)
                line_critic.set_xdata(steps_list)
                line_critic.set_ydata(critic_losses)
                line_total.set_xdata(steps_list)
                line_total.set_ydata(total_losses)
                
                ax.relim() # Recalculate limits
                ax.autoscale_view(True,True,True) # Autoscale
                fig.canvas.draw()
                fig.canvas.flush_events()
                print(f"Step: {step}/{hp.MAX_TRAINING_STEPS}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, Entropy: {entropy:.4f}, Total Loss (approx): {total_loss_value:.4f}")
            # --- End Update Plot Data ---

        print("Training finished.")
        plt.ioff() # Turn off interactive mode
        plt.show() # Keep plot open after training

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DroneModel(
        fusion_dim=hp.FUSION_DIM,
        scalar_input_dim=hp.SCALAR_INPUT_DIM,
        vit_image_size=hp.IMG_SIZE,
        vit_patch_size=hp.VIT_PATCH_SIZE,
        num_control_outputs=hp.NUM_CONTROL_OUTPUTS
    ).to(device)
    print(model)
    
    trainer = PPOTrainer(model, device, use_generated_data=True)
    try:
        trainer.run_training_loop()
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        if hasattr(trainer, 'api_client') and trainer.api_client:
            trainer.api_client.stop_listening()
        print("Cleanup complete.")