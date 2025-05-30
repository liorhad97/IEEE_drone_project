import torch
import torch.nn as nn
import torch.optim as optim
import time
from collections import deque
import random
import numpy as np

from data_models.drone_data_model import DroneData
from models.drone_agent_model import DroneModel
from config import hyperparameters as hp
from api import API
from reword_analasis import Reward

class TestDataGenerator:
    def __init__(self, device):
        self.device = device

    def generate_drone_data(self):
        camera = torch.randn(3, hp.IMG_SIZE, hp.IMG_SIZE, device=self.device)
        lidar = torch.randn(1, hp.IMG_SIZE, hp.IMG_SIZE, device=self.device)
        gps = torch.randn(3, device=self.device)
        accel = torch.tensor([torch.randn(1).item()], device=self.device)
        gyro = torch.randn(3, device=self.device)
        teta = torch.randn(2, device=self.device)
        return DroneData(time.time(), camera, lidar, gps, accel, gyro, teta)

class PPOTrainer:
    def __init__(self, model: DroneModel, device, use_generated_data: bool = True):
        self.model = model.to(device)
        self.device = device
        self.use_generated_data = use_generated_data
        
        self.api_client = API(stream_url=hp.SSE_STREAM_URL) 
        self.reward_calculator = Reward(
            api_client=self.api_client,
            accel_threshold=hp.REWARD_ACCEL_THRESHOLD,
            gyro_threshold=hp.REWARD_GYRO_THRESHOLD,
            teta_threshold=hp.REWARD_TETA_THRESHOLD,
            num_samples=hp.REWARD_NUM_SAMPLES,
            sample_delay=hp.REWARD_SAMPLE_DELAY
        )

        if self.use_generated_data:
            self.data_generator = TestDataGenerator(device=self.device)

        self.actor_optimizer = optim.Adam(self.model.actor_head.parameters(), lr=hp.LEARNING_RATE_ACTOR)
        self.critic_optimizer = optim.Adam(self.model.critic_head.parameters(), lr=hp.LEARNING_RATE_CRITIC)
        
        self.rollout_buffer = {
            'states': [], 'actions': [], 'log_probs': [], 
            'rewards': [], 'values': [], 'dones': [], 'advantages': [], 'returns': []
        }
        self.current_step = 0

    def _clear_rollout_buffer(self):
        for key in self.rollout_buffer:
            self.rollout_buffer[key].clear()

    def _get_current_drone_data(self):
        if self.use_generated_data:
            if hasattr(self, 'data_generator'):
                return self.data_generator.generate_drone_data()
            else:
                print("Error: TestDataGenerator not initialized even though use_generated_data is True. Falling back to dummy data.")
                temp_data_generator = TestDataGenerator(device=self.device)
                return temp_data_generator.generate_drone_data()
        else:
            print("Fetching real drone data from API...")
            sensor_data = self.api_client.get_sensors_data_from_api() 
            camera_image = self.api_client.get_camera_image_from_api().to(self.device) 
            lidar_image = self.api_client.get_lidar_image_from_api().to(self.device)   
            
            if sensor_data and camera_image is not None and lidar_image is not None:
                return DroneData(
                    timestamp=time.time(), 
                    camera_image=camera_image,
                    lidar_image=lidar_image,
                    gps_data=sensor_data.gps.to(self.device),
                    accelerometer_data=sensor_data.accel.to(self.device),
                    gyroscope_data=sensor_data.gyro.to(self.device),
                    attitude_data=sensor_data.teta.to(self.device)
                )
            else:
                print("Warning: Failed to fetch complete data from API. Falling back to dummy data generation.")
                if not hasattr(self, 'data_generator_fallback'):
                    self.data_generator_fallback = TestDataGenerator(device=self.device)
                return self.data_generator_fallback.generate_drone_data()

    def _compute_gae_and_returns(self, next_value, next_done):
        rewards = torch.tensor(self.rollout_buffer['rewards'], dtype=torch.float32, device=self.device)
        values = torch.tensor(self.rollout_buffer['values'], dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.rollout_buffer['dones'], dtype=torch.float32, device=self.device)
        
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t+1]
                nextvalues = values[t+1]
            delta = rewards[t] + hp.GAMMA * nextvalues * nextnonterminal - values[t]
            advantages[t] = last_gae_lam = delta + hp.GAMMA * hp.GAE_LAMBDA * nextnonterminal * last_gae_lam
        
        returns = advantages + values
        self.rollout_buffer['advantages'] = advantages
        self.rollout_buffer['returns'] = returns

    def train_ppo_epoch(self):
        old_actions = torch.stack(self.rollout_buffer['actions']).detach()
        old_log_probs = torch.stack(self.rollout_buffer['log_probs']).detach()
        advantages = self.rollout_buffer['advantages'].detach()
        returns = self.rollout_buffer['returns'].detach()
        states_data_list = self.rollout_buffer['states'] 

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_samples = len(states_data_list)
        indices = np.arange(num_samples)

        for _ in range(hp.PPO_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, num_samples, hp.PPO_MINI_BATCH_SIZE):
                end = start + hp.PPO_MINI_BATCH_SIZE
                minibatch_indices = indices[start:end]
                new_log_probs_list = []
                values_list = []
                entropy_list = []
                for idx in minibatch_indices:
                    log_prob, value, entropy = self.model.evaluate(states_data_list[idx], old_actions[idx])
                    new_log_probs_list.append(log_prob)
                    values_list.append(value)
                    entropy_list.append(entropy)
                
                new_log_probs = torch.stack(new_log_probs_list)
                values = torch.stack(values_list).squeeze()
                entropy = torch.stack(entropy_list).mean()

                log_ratio = new_log_probs - old_log_probs[minibatch_indices]
                ratio = torch.exp(log_ratio)

                surr1 = ratio * advantages[minibatch_indices]
                surr2 = torch.clamp(ratio, 1.0 - hp.CLIP_EPSILON, 1.0 + hp.CLIP_EPSILON) * advantages[minibatch_indices]
                actor_loss = -torch.min(surr1, surr2).mean()
                    
                critic_loss = nn.MSELoss()(values, returns[minibatch_indices])

                loss = actor_loss + hp.CRITIC_LOSS_COEFF * critic_loss - hp.ENTROPY_COEFF * entropy

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        
        return actor_loss.item(), critic_loss.item(), entropy.item()

    def run_training_loop(self):
        print(f"Starting PPO training on {self.device}...")
        self.model.train()
        
        current_drone_state = self._get_current_drone_data()

        for step in range(hp.MAX_TRAINING_STEPS):
            self.current_step = step
            
            for _ in range(hp.ROLLOUT_LENGTH):
                with torch.no_grad():
                    action, log_prob, value = self.model.act(current_drone_state)

                next_drone_state = self._get_current_drone_data()
                reward = self.reward_calculator.get_reward()
                done = self.api_client.is_drone_find_objective()

                self.rollout_buffer['states'].append(current_drone_state)
                self.rollout_buffer['actions'].append(action.squeeze())
                self.rollout_buffer['log_probs'].append(log_prob)
                self.rollout_buffer['rewards'].append(reward)
                self.rollout_buffer['values'].append(value.squeeze())
                self.rollout_buffer['dones'].append(done)
                
                current_drone_state = next_drone_state
                if done:
                    print(f"Objective found or episode ended at step {step}. Resetting.")
                    current_drone_state = self._get_current_drone_data()
            
            with torch.no_grad():
                _, _, next_value = self.model.act(current_drone_state)
            next_done_tensor = torch.tensor([float(self.api_client.is_drone_find_objective())], device=self.device)
            self._compute_gae_and_returns(next_value.squeeze(), next_done_tensor)
            
            actor_loss, critic_loss, entropy = self.train_ppo_epoch()
            
            self._clear_rollout_buffer()
            
            if step % 10 == 0:
                print(f"Step: {step}/{hp.MAX_TRAINING_STEPS}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, Entropy: {entropy:.4f}")

        print("Training finished.")

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