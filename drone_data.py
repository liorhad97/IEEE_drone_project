import torch
class DroneData:
    """
    A class representing the data for the drone.
    """
    def __init__(self, timestamp: float, gps_position: torch.Tensor, lidar_data: torch.Tensor, camera_data: torch.Tensor, accelerometer_data: float, gyroscope_data: torch.Tensor, teta: torch.Tensor):
        """
        Initializes the DroneData object.

        Args:
            timestamp (float): A single float value representing the time of data capture.
            gps_position (torch.Tensor): A 3-element tensor for GPS coordinates (e.g., [X, Y, Z] or [latitude, longitude, altitude]).
            lidar_data (torch.Tensor): A tensor containing Lidar scan data, typically 1080 data points (e.g., distance measurements).
            camera_data (torch.Tensor): A tensor containing camera data, specified as 1080 pixels (this could represent a flattened image segment or a feature vector).
            accelerometer_data (float): A single float value from the accelerometer.
            gyroscope_data (torch.Tensor): A 3-element tensor where each element represents the rotational speed around the X, Y, and Z axes, respectively.
            teta (torch.Tensor): A tensor representing the drone's angular position, such as its angle relative to North and its angle relative to the ground.
        """
        # Convert scalar float inputs to scalar tensors.
        # The original `torch.Tensor(float_value)` would cause a TypeError for a single float.
        # `torch.tensor(float_value)` correctly creates a scalar tensor.
        self.timestamp = torch.tensor(timestamp) 
        self.GPS_position = gps_position
        self.Lidar_data = lidar_data
        self.camera_data = camera_data
        self.accelerometer_data = torch.tensor(accelerometer_data)
        self.gyroscope_data = gyroscope_data
        self.teta = teta