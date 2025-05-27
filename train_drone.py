import torch
import torch.nn as nn # Ensure nn is imported
import torch.optim as optim # Import optimizer
import time
from collections import deque
import random
import numpy as np # For GAE calculation

from data_models.drone_data_model import DroneData
from models.drone_agent_model import DroneModel
from config import hyperparameters as hp
from api import API # Import the API class
from reword_analasis import Reward # Import the Reward class

class PPOTrainer:
    def __init__(self, model: DroneModel, device):
        self.model = model.to(device)
        self.device = device
        
        # Initialize API client and Reward calculator
        # Ensure SSE_STREAM_URL is correctly set in hyperparameters
        self.api_client = API(stream_url=hp.SSE_STREAM_URL) 
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
        
        # PPO specific storage for rollouts
        self.rollout_buffer = {
            'states': [], 'actions': [], 'log_probs': [], 
            'rewards': [], 'values': [], 'dones': [], 'advantages': [], 'returns': []
        }
        self.current_step = 0

    def _clear_rollout_buffer(self):
        for key in self.rollout_buffer:
            self.rollout_buffer[key].clear()

    def _generate_dummy_drone_data(self): # Simulates getting data from environment
        camera = torch.randn(3, hp.IMG_SIZE, hp.IMG_SIZE, device=self.device)
        lidar = torch.randn(1, hp.IMG_SIZE, hp.IMG_SIZE, device=self.device)
        # Scalar data should also be on the correct device and correctly shaped
        gps = torch.randn(3, device=self.device)
        accel = torch.tensor([torch.randn(1).item()], device=self.device) # Ensure accel is a 1D tensor
        gyro = torch.randn(3, device=self.device)
        teta = torch.randn(2, device=self.device)
        return DroneData(time.time(), camera, lidar, gps, accel, gyro, teta)

    def _compute_gae_and_returns(self, next_value, next_done):
        # Convert lists to tensors for computation
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
        # Convert rollout data to tensors
        # Note: States are DroneData objects, handle them appropriately when batching
        # For simplicity, we'll assume states can be processed one by one or need a custom collate_fn if batched directly
        old_actions = torch.stack(self.rollout_buffer['actions']).detach()
        old_log_probs = torch.stack(self.rollout_buffer['log_probs']).detach()
        advantages = self.rollout_buffer['advantages'].detach()
        returns = self.rollout_buffer['returns'].detach()
        # States are kept as a list of DroneData objects
        states_data_list = self.rollout_buffer['states'] 

        # Normalize advantages (optional but often helpful)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_samples = len(states_data_list)
        indices = np.arange(num_samples)

        for _ in range(hp.PPO_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, num_samples, hp.PPO_MINI_BATCH_SIZE):
                end = start + hp.PPO_MINI_BATCH_SIZE
                minibatch_indices = indices[start:end]

                # Process DroneData objects for the minibatch
                # This part needs careful handling if you batch DroneData objects directly
                # For now, let's assume we re-evaluate the model for each state in the minibatch
                # This is less efficient but simpler to implement without a custom collate_fn
                
                # Placeholder for minibatch processing - this needs to be efficient
                # A better way is to batch the forward pass if possible
                # For simplicity, this example might be slow due to iterating for new_log_probs, values, entropy
                
                # Actor Loss
                # Re-evaluate actions for the current policy on minibatch states
                # This is where you'd ideally batch process states_data_list[minibatch_indices]
                # For simplicity, let's iterate. This is NOT efficient for actual training.
                new_log_probs_list = []
                values_list = []
                entropy_list = []
                for idx in minibatch_indices:
                    log_prob, value, entropy = self.model.evaluate(states_data_list[idx], old_actions[idx])
                    new_log_probs_list.append(log_prob)
                    values_list.append(value)
                    entropy_list.append(entropy)
                
                new_log_probs = torch.stack(new_log_probs_list)
                values = torch.stack(values_list).squeeze() # Ensure values is 1D if it comes out as [N,1]
                entropy = torch.stack(entropy_list).mean() # Mean entropy for the batch

                log_ratio = new_log_probs - old_log_probs[minibatch_indices]
                ratio = torch.exp(log_ratio)

                surr1 = ratio * advantages[minibatch_indices]
                surr2 = torch.clamp(ratio, 1.0 - hp.CLIP_EPSILON, 1.0 + hp.CLIP_EPSILON) * advantages[minibatch_indices]
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic Loss
                # Ensure values and returns[minibatch_indices] are correctly shaped for MSELoss
                # values should be [minibatch_size], returns[minibatch_indices] should be [minibatch_size]
                critic_loss = nn.MSELoss()(values, returns[minibatch_indices])

                # Total Loss
                loss = actor_loss + hp.CRITIC_LOSS_COEFF * critic_loss - hp.ENTROPY_COEFF * entropy

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                # Gradient clipping (optional but often helpful)
                # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        
        # Return average losses for logging, if needed
        # For simplicity, not calculating average losses here, but you might want to.
        return actor_loss.item(), critic_loss.item(), entropy.item()

    def run_training_loop(self):
        print(f"Starting PPO training on {self.device}...")
        self.model.train() # Set model to training mode
        
        # Initial state (dummy for now, replace with actual environment interaction)
        current_drone_state = self._generate_dummy_drone_data()

        for step in range(hp.MAX_TRAINING_STEPS):
            self.current_step = step
            
            # Collect ROLLOUT_LENGTH transitions
            for _ in range(hp.ROLLOUT_LENGTH):
                with torch.no_grad():
                    action, log_prob, value = self.model.act(current_drone_state)
                
                # Simulate environment step: get next_state, reward, done
                # This is where you interact with your drone environment/API
                # For now, using dummy data and reward calculation
                next_drone_state = self._generate_dummy_drone_data() # Simulate next state
                reward = self.reward_calculator.get_reward() # Get reward from your Reward class
                done = self.api_client.is_drone_find_objective() # Check if objective found (episode ends)
                # Or, done might be true if max episode length is reached, etc.

                # Store transition
                self.rollout_buffer['states'].append(current_drone_state)
                self.rollout_buffer['actions'].append(action.squeeze()) # Ensure action is stored correctly
                self.rollout_buffer['log_probs'].append(log_prob)
                self.rollout_buffer['rewards'].append(reward)
                self.rollout_buffer['values'].append(value.squeeze()) # Ensure value is scalar
                self.rollout_buffer['dones'].append(done)
                
                current_drone_state = next_drone_state
                if done:
                    # If episode ends, reset environment and get new initial state
                    # For dummy data, just regenerate. In a real env, this means a reset call.
                    print(f"Objective found or episode ended at step {step}. Resetting.")
                    current_drone_state = self._generate_dummy_drone_data()
                    # Potentially break from rollout collection if an episode ends early
                    # and handle partial rollouts, or ensure rollouts are always full.
                    # For simplicity, we continue filling the rollout buffer here.
            
            # Compute GAE and returns for the collected rollout
            with torch.no_grad():
                _, _, next_value = self.model.act(current_drone_state) # Value of the last state in rollout
            next_done_tensor = torch.tensor([float(self.api_client.is_drone_find_objective())], device=self.device)
            self._compute_gae_and_returns(next_value.squeeze(), next_done_tensor)
            
            # Train PPO for PPO_EPOCHS using the collected rollout data
            actor_loss, critic_loss, entropy = self.train_ppo_epoch()
            
            # Clear rollout buffer for next iteration
            self._clear_rollout_buffer()
            
            if step % 10 == 0: # Log progress periodically
                print(f"Step: {step}/{hp.MAX_TRAINING_STEPS}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, Entropy: {entropy:.4f}")

        print("Training finished.")
        # Save the model (optional)
        # torch.save(self.model.state_dict(), "ppo_drone_model.pth")
        # self.api_client.stop_listening() # Clean up API client listener

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure hyperparameters are loaded, especially scalar_input_dim and num_control_outputs
    model = DroneModel(
        fusion_dim=hp.FUSION_DIM,
        scalar_input_dim=hp.SCALAR_INPUT_DIM, # Make sure this matches your DroneData scalar components
        vit_image_size=hp.IMG_SIZE,
        vit_patch_size=hp.VIT_PATCH_SIZE,
        num_control_outputs=hp.NUM_CONTROL_OUTPUTS
    ).to(device)
    
    trainer = PPOTrainer(model, device)
    try:
        trainer.run_training_loop()
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        if hasattr(trainer, 'api_client') and trainer.api_client:
            trainer.api_client.stop_listening() # Ensure SSE listener is stopped
        print("Cleanup complete.")