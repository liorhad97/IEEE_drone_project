import torch
import torch.nn as nn
import time
from collections import deque
import random

from data_models.drone_data_model import DroneData
from models.drone_agent_model import DroneModel
from config import hyperparameters as hp

class Trainer:
    def __init__(self, model :DroneModel, device):
        self.model = model
        self.device = device
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = deque(maxlen=hp.REPLAY_BUFFER_SIZE)

    def generate_dummy_drone_experience(self):
        camera = torch.randn(3, hp.IMG_SIZE, hp.IMG_SIZE)
        lidar = torch.randn(1, hp.IMG_SIZE, hp.IMG_SIZE)
        gps = torch.randn(3)
        accel = torch.tensor([torch.randn(1).item()])
        gyro = torch.randn(3)
        teta = torch.randn(2)
        current_drone_data = DroneData(time.time(), camera, lidar, gps, accel, gyro, teta)
        expert_controls = torch.rand(hp.NUM_CONTROL_OUTPUTS) * 2 - 1
        return (current_drone_data, expert_controls)

    def populate_replay_buffer(self):
        for _ in range(hp.REPLAY_BUFFER_SIZE // 10):
            self.replay_buffer.append(self.generate_dummy_drone_experience())

    def train_epoch(self):
        if len(self.replay_buffer) < hp.BATCH_SIZE:
            return None

        batch_experiences = random.sample(self.replay_buffer, hp.BATCH_SIZE)
        states_data_list = [exp[0] for exp in batch_experiences]
        target_controls_list = [exp[1] for exp in batch_experiences]

        predicted_controls_batch = []
        for i in range(hp.BATCH_SIZE):
            current_state_data = states_data_list[i]
            predicted_controls = self.model(current_state_data)
            predicted_controls_batch.append(predicted_controls.squeeze(0))
        
        predicted_controls_tensor = torch.stack(predicted_controls_batch)
        target_controls_tensor = torch.stack(target_controls_list).to(self.device)

        loss = self.loss_fn(predicted_controls_tensor, target_controls_tensor)
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()
        return loss.item()

    def run_training_loop(self):
        self.populate_replay_buffer()

        if len(self.replay_buffer) < hp.BATCH_SIZE:
            return

        for epoch in range(hp.EPOCHS):
            loss_item = self.train_epoch()
            if loss_item is not None:
                print(f"Epoch {epoch+1}, Batch Loss (MSE): {loss_item:.4f}")
            else:
                continue
            
            if epoch < hp.EPOCHS -1:
                for _ in range(hp.BATCH_SIZE):
                    if len(self.replay_buffer) < hp.REPLAY_BUFFER_SIZE:
                         self.replay_buffer.append(self.generate_dummy_drone_experience())

    def evaluate_model(self):
        self.model.eval()
        example_state_data, example_expert_controls = self.generate_dummy_drone_experience()
        with torch.no_grad():
            predicted_joystick_outputs = self.model(example_state_data)

        inference_mse = self.loss_fn(predicted_joystick_outputs.squeeze(), example_expert_controls.to(self.device))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DroneModel(
        fusion_dim=hp.FUSION_DIM,
        scalar_input_dim=hp.SCALAR_INPUT_DIM,
        vit_image_size=hp.IMG_SIZE,
        vit_patch_size=hp.VIT_PATCH_SIZE,
        num_control_outputs=hp.NUM_CONTROL_OUTPUTS,
        learning_rate=hp.LEARNING_RATE
    ).to(device)
    
    trainer = Trainer(model, device)
    trainer.run_training_loop()
    trainer.evaluate_model()