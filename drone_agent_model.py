# drone_agent_model.py
import torch
import torch.nn as nn
import torch.optim as optim

from drone_data import DroneData # Assuming it's in the same directory or python path
from vit_components import ImageVitModel, LidarVitModel # Assuming it's in the same directory

class DroneModel(nn.Module):
    """
    A multimodal transformer model for drone control, outputting continuous
    joystick-like signals.
    """
    def __init__(self,
                 fusion_dim=768,
                 scalar_input_dim=9, # GPS (3) + Accel (1) + Gyro (3) + Teta (2) = 9
                 vit_internal_dim=768,
                 vit_depth=12,
                 vit_heads=12,
                 vit_mlp_dim=3072,
                 vit_patch_size=30,
                 vit_image_size=1080,
                 vit_lidar_image_size=1080,
                 fusion_depth=3,
                 fusion_heads=8,
                 fusion_mlp_dim_factor=4,
                 fusion_dropout=0.1,
                 num_control_outputs=4, # e.g., Roll, Pitch, Yaw, Throttle
                 optimizer_class=optim.Adam,
                 learning_rate=0.001):

        super(DroneModel, self).__init__()
        self.learning_rate = learning_rate
        self.fusion_dim = fusion_dim
        self.num_control_outputs = num_control_outputs

        self.imageVit = ImageVitModel(
            image_size=vit_image_size,
            patch_size=vit_patch_size,
            num_classes=fusion_dim,
            dim=vit_internal_dim,
            depth=vit_depth,
            heads=vit_heads,
            mlp_dim=vit_mlp_dim,
            channels=3
        )

        self.lidarVit = LidarVitModel(
            image_size=vit_lidar_image_size,
            patch_size=vit_patch_size,
            num_classes=fusion_dim,
            dim=vit_internal_dim,
            depth=vit_depth,
            heads=vit_heads,
            mlp_dim=vit_mlp_dim,
            channels=1
        )

        self.scalar_embedder = nn.Sequential(
            nn.Linear(scalar_input_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, fusion_dim)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, fusion_dim))
        self.cls_token.requires_grad = True
        self.pos_embedding_fusion = nn.Parameter(torch.randn(1, 4, fusion_dim)) # CLS, Img, Lidar, Scalar

        fusion_transformer_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=fusion_heads,
            dim_feedforward=fusion_dim * fusion_mlp_dim_factor,
            dropout=fusion_dropout,
            batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(
            fusion_transformer_layer,
            num_layers=fusion_depth
        )

        # Output head for continuous control signals
        self.control_head = nn.Linear(fusion_dim, num_control_outputs)
        self.output_activation = nn.Tanh() # To scale outputs to [-1, 1]

        self.optimizer = optimizer_class(self.parameters(), lr=self.learning_rate)


    def forward(self, data: DroneData) -> torch.Tensor:
        device = self.cls_token.device # Use device of one of the model's parameters

        # 1. Process Image Data
        camera_input = data.camera_data.to(device)
        if camera_input.ndim == 3:
            camera_input = camera_input.unsqueeze(0)
        image_embedding = self.imageVit(camera_input)

        # 2. Process LiDAR Data
        lidar_input = data.Lidar_data.to(device) # Corrected attribute name based on DroneData
        if lidar_input.ndim == 3:
            lidar_input = lidar_input.unsqueeze(0)
        elif lidar_input.ndim == 2:
            lidar_input = lidar_input.unsqueeze(0).unsqueeze(0)
        lidar_embedding = self.lidarVit(lidar_input)

        # 3. Process Scalar Data
        gps = data.GPS_position # Corrected attribute name
        accel = data.accelerometer_data
        gyro = data.gyroscope_data
        teta_val = data.teta

        if not isinstance(gps, torch.Tensor): gps = torch.tensor(gps, dtype=torch.float32)
        if not isinstance(accel, torch.Tensor): accel = torch.tensor([accel] if isinstance(accel, float) else accel, dtype=torch.float32)
        if not isinstance(gyro, torch.Tensor): gyro = torch.tensor(gyro, dtype=torch.float32)
        if not isinstance(teta_val, torch.Tensor): teta_val = torch.tensor(teta_val, dtype=torch.float32)

        gps = gps.to(device).float().view(1, -1)
        accel = accel.to(device).float().view(1, -1)
        gyro = gyro.to(device).float().view(1, -1)
        teta_val = teta_val.to(device).float().view(1, -1)

        scalar_features_cat = torch.cat([gps, accel, gyro, teta_val], dim=1)
        scalar_embedding = self.scalar_embedder(scalar_features_cat)

        # 4. Prepare tokens for Fusion Transformer
        image_token = image_embedding.unsqueeze(1)
        lidar_token = lidar_embedding.unsqueeze(1)
        scalar_token = scalar_embedding.unsqueeze(1)

        batch_size = image_token.shape[0] # Should be 1 for single DroneData processing
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        tokens = torch.cat([cls_tokens, image_token, lidar_token, scalar_token], dim=1)
        
        if batch_size > 1 and self.pos_embedding_fusion.shape[0] == 1:
            pos_embedding = self.pos_embedding_fusion.expand(batch_size, -1, -1)
        else:
            pos_embedding = self.pos_embedding_fusion
        
        tokens = tokens + pos_embedding.to(device)


        fused_output = self.fusion_transformer(tokens)
        final_representation = fused_output[:, 0]  # Get the CLS token representation [B, fusion_dim]

        # 5. Control Head
        control_outputs = self.control_head(final_representation) # [B, num_control_outputs]
        activated_control_outputs = self.output_activation(control_outputs) # [B, num_control_outputs] in [-1, 1]

        return activated_control_outputs

if __name__ == '__main__':
    print("Testing DroneModel for continuous control output...")

    dummy_cam = torch.randn(3, 1080, 1080)
    dummy_lid = torch.randn(1, 1080, 1080)
    dummy_gps = torch.randn(3)
    dummy_accel = torch.tensor([0.5]) # scalar
    dummy_gyro = torch.randn(3)
    dummy_teta = torch.randn(2)

    test_data = DroneData(
        timestamp=0.0,
        camera_data=dummy_cam,
        lidar_data=dummy_lid,
        gps_position=dummy_gps,
        accelerometer_data=dummy_accel,
        gyroscope_data=dummy_gyro,
        teta=dummy_teta
    )
    
    # scalar_input_dim: GPS(3) + Accel(1) + Gyro(3) + Teta(2) = 9
    model = DroneModel(
        scalar_input_dim=9, 
        num_control_outputs=4, # e.g., Roll, Pitch, Yaw, Throttle
        fusion_dim=128, vit_internal_dim=128, vit_mlp_dim=256 # Smaller for faster test
    )
    model.eval()

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    with torch.no_grad():
        control_signals = model(test_data)

    print(f"Output control signals shape: {control_signals.shape}") # Expected: [1, num_control_outputs]
    print(f"Control signals (Roll, Pitch, Yaw, Throttle): {control_signals.squeeze().tolist()}")
    assert control_signals.shape[1] == 4
    assert control_signals.min() >= -1.0 and control_signals.max() <= 1.0
    print("Test successful: Output shape and range are correct.")