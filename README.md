# IEEE Drone Project

## Project Overview
In this project, we will be implementing a solution to the following problem:

- An unknown person is located in an environment
- We only have a single photo of the person
- Our goal is to find this person using Reinforcement Learning techniques

## Data Collection
The drone will provide the following data:

### Scalar Data
- **A**: Accelerometer data from the drone
- **V**: Drone speed
- **θ**: Drone orientation relative to magnetic north
- **G**: GPS coordinates (x, y, z) of the drone's position

### Tensor Data
- **I**: Image from the drone camera at timestamp t
- **L**: LIDAR sensor data

## Agent Output Options

### Option 1: Orientation and Acceleration Control
- **θ_hat**: Degree of change relative to current orientation (θ_hat - θ)
- **A_hat**: Relative acceleration where A(t) = A_hat(t-1) + A(t-1)

### Option 2: Thrust Control
- **Y**: 4-element vector representing acceleration of each thrust in the quadcopter 
- Y(t) = Y(t-1) + Y_hat(t-1)
- *Note: This approach is more abstract and may be harder to learn but easier to implement*

### Option 3: Controller Emulation
Estimating joystick positions:

- **R_L_hat**: Radius of left joystick from center
- **θ_L_hat**: Angle of left joystick relative to forward position
- **R_R_hat**: Radius of right joystick from center
- **θ_R_hat**: Angle of right joystick relative to forward position
- **H_L_hat**: Height of left trigger/button
- **H_R_hat**: Height of right trigger/button

# we will do option 3 (:






