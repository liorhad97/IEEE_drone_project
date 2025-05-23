# train_drone.py
import torch
import torch.nn as nn
# import torch.optim as optim # Not strictly needed here as optimizer is in model
import time
from collections import deque
import random # For sampling from replay buffer

from drone_data import DroneData
from drone_agent_model import DroneModel

if __name__ == "__main__":
    print("--- Drone Behavioral Cloning Training Simulation ---")

    # --- Hyperparameters ---
    FUSION_DIM = 128       # Smaller for quicker example
    VIT_INTERNAL_DIM = 128
    VIT_MLP_DIM = 256
    NUM_CONTROL_OUTPUTS = 4 # Roll, Pitch, Yaw, Throttle
    LEARNING_RATE = 0.001
    REPLAY_BUFFER_SIZE = 1000
    BATCH_SIZE = 4 # Increased slightly
    EPOCHS = 3
    SCALAR_INPUT_DIM = 3 + 1 + 3 + 2 # GPS(3) + Accel(1) + Gyro(3) + Teta(2) = 9
    IMG_SIZE = 224 # Using smaller images for faster dummy data generation/processing
    VIT_PATCH_SIZE = 16 # Ensure IMG_SIZE is divisible by VIT_PATCH_SIZE

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Model Initialization ---
    model = DroneModel(
        fusion_dim=FUSION_DIM,
        scalar_input_dim=SCALAR_INPUT_DIM,
        vit_internal_dim=VIT_INTERNAL_DIM,
        vit_mlp_dim=VIT_MLP_DIM,
        vit_image_size=IMG_SIZE,      # Use smaller image size
        vit_lidar_image_size=IMG_SIZE,# Use smaller image size
        vit_patch_size=VIT_PATCH_SIZE,            # Adjust patch size for smaller images
        num_control_outputs=NUM_CONTROL_OUTPUTS,
        learning_rate=LEARNING_RATE
    ).to(device)
    model.train()

    loss_fn = nn.MSELoss() # Mean Squared Error for continuous value regression

    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    print("Model initialized for behavioral cloning.")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # --- Dummy Data Generation Function ---
    def generate_dummy_drone_experience(img_size=IMG_SIZE, num_controls=NUM_CONTROL_OUTPUTS): # Function definition
        camera = torch.randn(3, img_size, img_size)
        lidar = torch.randn(1, img_size, img_size)
        gps = torch.randn(3)
        accel = torch.tensor([torch.randn(1).item()])
        gyro = torch.randn(3)
        teta = torch.randn(2)
        
        current_drone_data = DroneData(time.time(), camera, lidar, gps, accel, gyro, teta)
        
        expert_controls = torch.rand(num_controls) * 2 - 1 # Random values in [-1, 1]
        
        experience = (current_drone_data, expert_controls)
        return experience

    for _ in range(REPLAY_BUFFER_SIZE // 10):
        replay_buffer.append(generate_dummy_drone_experience())

    # --- Training Loop (Behavioral Cloning) ---
    if len(replay_buffer) >= BATCH_SIZE:

        for epoch in range(EPOCHS):
            
            if len(replay_buffer) < BATCH_SIZE:
                continue
            
            batch_experiences = random.sample(replay_buffer, BATCH_SIZE)

            states_data_list = [exp[0] for exp in batch_experiences]
            target_controls_list = [exp[1] for exp in batch_experiences]

            predicted_controls_batch = []
            
            for i in range(BATCH_SIZE):
                current_state_data = states_data_list[i]
                predicted_controls = model(current_state_data)
                predicted_controls_batch.append(predicted_controls.squeeze(0))
            
            predicted_controls_tensor = torch.stack(predicted_controls_batch)
            target_controls_tensor = torch.stack(target_controls_list).to(device)

            loss = loss_fn(predicted_controls_tensor, target_controls_tensor)
            print(f"Batch Loss (MSE): {loss.item()}")

            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            print(f"Epoch {epoch+1}: Weights updated.")
            
            if epoch < EPOCHS -1:
                for _ in range(BATCH_SIZE):
                    if len(replay_buffer) < REPLAY_BUFFER_SIZE:
                         replay_buffer.append(generate_dummy_drone_experience()) # CORRECTED CALL

        print("\n--- Training Simulation Complete ---")

    else:
        print(f"Not enough experiences in replay buffer ({len(replay_buffer)}) to start training (min: {BATCH_SIZE}).")

    # --- Example: Inference with the trained model ---
    print("\n--- Example: Inference with the 'trained' model ---")
    model.eval()
    
    example_state_data, example_expert_controls = generate_dummy_drone_experience() # CORRECTED CALL
    
    with torch.no_grad():
        predicted_joystick_outputs = model(example_state_data)
    
    print(f"Example State's Expert Controls: {example_expert_controls.tolist()}")
    print(f"Predicted Joystick Outputs:      {predicted_joystick_outputs.squeeze().tolist()}")
    inference_mse = loss_fn(predicted_joystick_outputs.squeeze(), example_expert_controls.to(device))
    print(f"MSE for this example: {inference_mse.item()}")