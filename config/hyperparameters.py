FUSION_DIM = 128
VIT_INTERNAL_DIM = 128
VIT_MLP_DIM = 256
NUM_CONTROL_OUTPUTS = 4
LEARNING_RATE = 0.001
REPLAY_BUFFER_SIZE = 1000
BATCH_SIZE = 4
EPOCHS = 3
SCALAR_INPUT_DIM = 9 # GPS(3) + Accel(1) + Gyro(3) + Teta(2)
IMG_SIZE = 224
VIT_PATCH_SIZE = 16

# PPO Specific Hyperparameters
GAMMA = 0.99  # Discount factor for rewards
GAE_LAMBDA = 0.95  # Lambda for Generalized Advantage Estimation
CLIP_EPSILON = 0.2  # Epsilon for PPO clipping
PPO_EPOCHS = 10  # Number of epochs to train on a collected batch of data
PPO_MINI_BATCH_SIZE = 2 # Number of mini-batches to split a batch into for PPO updates
CRITIC_LOSS_COEFF = 0.5  # Coefficient for the critic loss
ENTROPY_COEFF = 0.01  # Coefficient for the entropy bonus
MAX_TRAINING_STEPS = 10000  # Total number of training steps (environment interactions)
ROLLOUT_LENGTH = 128  # Number of steps to collect in each rollout before updating
LEARNING_RATE_ACTOR = 3e-4 # Learning rate for the actor
LEARNING_RATE_CRITIC = 1e-3 # Learning rate for the critic
# API and Reward related
SSE_STREAM_URL = "http://localhost:5000/stream" # Placeholder, replace with your actual SSE stream URL
REWARD_ACCEL_THRESHOLD = 0.1
REWARD_GYRO_THRESHOLD = 0.1
REWARD_TETA_THRESHOLD = 0.1
REWARD_NUM_SAMPLES = 4
REWARD_SAMPLE_DELAY = 0.001