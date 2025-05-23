# drone_data.py
import torch
import time
from typing import Any

class DroneData:
    """
    A simple class to hold drone sensor data.
    """
    def __init__(self,
                 timestamp: float,
                 camera_data: torch.Tensor,
                 lidar_data: torch.Tensor,
                 gps_position: torch.Tensor,
                 accelerometer_data: torch.Tensor, # Or float, will be converted to tensor in model
                 gyroscope_data: torch.Tensor,
                 teta: torch.Tensor):
        self.timestamp = timestamp
        self.camera_data = camera_data
        self.Lidar_data = lidar_data # Matches attribute name used in original DroneModel's forward
        self.GPS_position = gps_position # Matches attribute name used in original DroneModel's forward
        self.accelerometer_data = accelerometer_data
        self.gyroscope_data = gyroscope_data
        self.teta = teta

    def __repr__(self):
        return (f"DroneData(timestamp={self.timestamp}, "
                f"camera_data_shape={self.camera_data.shape}, "
                f"lidar_data_shape={self.Lidar_data.shape}, "
                f"gps_position={self.GPS_position}, "
                f"accelerometer_data={self.accelerometer_data}, "
                f"gyroscope_data={self.gyroscope_data}, "
                f"teta={self.teta})")

if __name__ == '__main__':
    # Example usage:
    current_camera_data = torch.randn(3, 1080, 1080)
    current_lidar_data = torch.randn(1, 1080, 1080) # Assuming single channel for LiDAR
    current_gps = torch.randn(3)
    current_accel = torch.tensor([0.1, 0.2, 0.3]) # Example, model expects tensor
    current_gyro = torch.randn(3)
    current_teta = torch.randn(2)

    drone_instance_data = DroneData(
        timestamp=time.time(),
        camera_data=current_camera_data,
        lidar_data=current_lidar_data,
        gps_position=current_gps,
        accelerometer_data=current_accel, # Pass as tensor
        gyroscope_data=current_gyro,
        teta=current_teta
    )
    print(drone_instance_data)