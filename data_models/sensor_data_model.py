import torch

class SensorsData:
    def __init__(self, gps, accel, gyro, teta):
        self.gps = gps
        self.accel = accel
        self.gyro = gyro
        self.teta = teta

    def to_tensor_tuple(self):
        return (self.gps, self.accel, self.gyro, self.teta)
    
    # create a static method to create an instance from a dictionary
    @staticmethod
    def from_dict(data_dict : dict):
        gps = torch.tensor(data_dict.get("gps", [0.0, 0.0, 0.0]))
        accel = torch.tensor(data_dict.get("accel", [0.0]))
        gyro = torch.tensor(data_dict.get("gyro", [0.0, 0.0, 0.0]))
        teta = torch.tensor(data_dict.get("teta", [0.0, 0.0]))
        return SensorsData(gps, accel, gyro, teta)
    



