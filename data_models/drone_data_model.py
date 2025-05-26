import time
import torch

class DroneData:
    def __init__(self, timestamp, camera_image, lidar_image, gps_data, accelerometer_data, gyroscope_data, attitude_data):
        self.timestamp = timestamp
        self.camera_image = camera_image
        self.lidar_image = lidar_image
        self.gps_data = gps_data
        self.accelerometer_data = accelerometer_data
        self.gyroscope_data = gyroscope_data
        self.attitude_data = attitude_data

    def to_tensor_tuple(self):
        return (
            self.camera_image,
            self.lidar_image,
            self.gps_data,
            self.accelerometer_data,
            self.gyroscope_data,
            self.attitude_data
        )

    @classmethod
    def from_dict(cls, data_dict):
        timestamp = data_dict.get("timestamp", time.time())
        camera_image = torch.tensor(data_dict["camera_image"]) if "camera_image" in data_dict else torch.randn(3, 224, 224) # Default if not provided
        lidar_image = torch.tensor(data_dict["lidar_image"]) if "lidar_image" in data_dict else torch.randn(1, 224, 224)
        gps_data = torch.tensor(data_dict.get("gps_data", [0.0,0.0,0.0]))
        accelerometer_data = torch.tensor(data_dict.get("accelerometer_data", [0.0]))
        gyroscope_data = torch.tensor(data_dict.get("gyroscope_data", [0.0,0.0,0.0]))
        attitude_data = torch.tensor(data_dict.get("attitude_data", [0.0,0.0]))
        return cls(timestamp, camera_image, lidar_image, gps_data, accelerometer_data, gyroscope_data, attitude_data)