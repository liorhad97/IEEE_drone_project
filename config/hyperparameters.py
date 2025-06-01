FUSION_DIM = 128
VIT_INTERNAL_DIM = 128
VIT_MLP_DIM = 256
NUM_CONTROL_OUTPUTS = 4
LEARNING_RATE = 0.0005  # Reduced learning rate for better convergence
REPLAY_BUFFER_SIZE = 2000  # Increased buffer size for more diverse experiences
BATCH_SIZE = 64  # Increased batch size for more stable learning
EPOCHS = 20  # Increased epochs for more training iterations
SCALAR_INPUT_DIM = 9 # GPS(3) + Accel(1) + Gyro(3) + Teta(2)
IMG_SIZE = 224
VIT_PATCH_SIZE = 16

# PPO Specific Hyperparameters
GAMMA = 0.99  # Discount factor for rewards
GAE_LAMBDA = 0.95  # Lambda for Generalized Advantage Estimation
CLIP_EPSILON = 0.1  # Reduced epsilon for more conservative policy updates
PPO_EPOCHS = 15  # Increased epochs for more policy updates per batch
PPO_MINI_BATCH_SIZE = 16  # Increased mini-batch size for more stable learning
CRITIC_LOSS_COEFF = 0.5  # Coefficient for the critic loss
ENTROPY_COEFF = 0.02  # Slightly increased entropy coefficient to encourage exploration
MAX_TRAINING_STEPS = 2000  # Increased total training steps
ROLLOUT_LENGTH = 256  # Increased rollout length for better reward estimation
LEARNING_RATE_ACTOR = 1e-4  # Reduced actor learning rate for more stable learning
LEARNING_RATE_CRITIC = 5e-4  # Reduced critic learning rate

# Gradient clipping parameters
GRAD_CLIP_NORM = 0.5  # Maximum norm for gradient clipping
# API and Reward related
SSE_STREAM_URL = "http://localhost:5000/stream" # Placeholder, replace with your actual SSE stream URL
REWARD_ACCEL_THRESHOLD = 0.1
REWARD_GYRO_THRESHOLD = 0.1
REWARD_TETA_THRESHOLD = 0.1
REWARD_NUM_SAMPLES = 4
REWARD_SAMPLE_DELAY = 0.001

# Simulation parameters for Gaussian distributions
SIM_OBJECTIVE_FIND_INTERVAL = 30.0  # Time interval between finding objectives
SIM_OBJECTIVE_RESET_INTERVAL = 15.0  # Time after finding objective to reset
SIM_LISTENER_TICK = 0.5  # Tick rate for the simulation
SIM_GPS_DRIFT_SCALE = 0.001  # Controls Gaussian variance for GPS drift
SIM_ACCEL_RANGE = 0.2  # Range for acceleration (converted to [-value, value])
SIM_GYRO_RANGE = 0.05  # Range for gyroscope (converted to [-value, value])
SIM_TETA_YAW_RANGE = 3.14  # Range for yaw (converted to [-value, value])
SIM_TETA_PITCH_RANGE = 1.57  # Range for pitch (converted to [-value, value])