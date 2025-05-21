import torch
import torch.nn as nn
import torch.optim as optim
from drone_data import DroneData
from vit_pytorch import ViT
import time


class DroneModel(nn.Module):
    """
    A class representing a drone in the IEEE Drone Project, using a multimodal
    transformer architecture for state representation and decision making.
    """
    def __init__(self, 
                 fusion_dim=768, 
                 scalar_input_dim=9, # GPS (3) + Accel (1) + Gyro (3) + Teta (2)
                 vit_internal_dim=768, # Internal dimension for Image/LIDAR ViTs
                 vit_depth=12,
                 vit_heads=12,
                 vit_mlp_dim=3072,
                 vit_patch_size=30,
                 fusion_depth=3, 
                 fusion_heads=8, 
                 fusion_mlp_dim_factor=4,
                 fusion_dropout=0.1,
                 num_actions=6, # Example number of actions for policy head
                 optimizer_class=optim.Adam, 
                 learning_rate=0.001):
        
        super(DroneModel, self).__init__()
        self.learning_rate = learning_rate
        self.fusion_dim = fusion_dim

        # Instantiate ViT for Camera Data
        # Output dimension (num_classes) is set to fusion_dim
        self.imageVit = ImageVitModel(
            image_size=1080, 
            patch_size=vit_patch_size, 
            num_classes=fusion_dim, # Output will be (batch, fusion_dim)
            dim=vit_internal_dim, 
            depth=vit_depth, 
            heads=vit_heads, 
            mlp_dim=vit_mlp_dim, 
            channels=3
        )
        
        # Instantiate ViT for LIDAR Data
        # Output dimension (num_classes) is set to fusion_dim
        self.lidarVit = LidarVitModel(
            image_size=1080, 
            patch_size=vit_patch_size, 
            num_classes=fusion_dim, # Output will be (batch, fusion_dim)
            dim=vit_internal_dim, 
            depth=vit_depth, 
            heads=vit_heads, 
            mlp_dim=vit_mlp_dim, 
            channels=1 # LIDAR data is single channel
        )

        # Scalar Embedding Module
        self.scalar_embedder = nn.Sequential(
            nn.Linear(scalar_input_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, fusion_dim)
        )

        # Learnable CLS token for the fusion transformer
        self.cls_token = nn.Parameter(torch.randn(1, 1, fusion_dim))

        # Positional embedding for the sequence of 4 tokens (CLS, Camera, LIDAR, Scalar)
        self.pos_embedding_fusion = nn.Parameter(torch.randn(1, 4, fusion_dim))

        # Fusion Transformer
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

        # Output heads (example for Reinforcement Learning)
        self.policy_head = nn.Linear(fusion_dim, num_actions)
        self.value_head = nn.Linear(fusion_dim, 1)
        
        # Optimizer (can be initialized here or outside)
        # self.optimizer = optimizer_class(self.parameters(), lr=self.learning_rate)


    def forward(self, data: DroneData) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the multimodal drone model.

        Args:
            data (DroneData): Input DroneData object containing all state components.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Policy logits and value estimate.
        """
        # Infer batch size, assuming camera_data is always batched
        batch_size = data.camera_data.shape[0]

        # 1. Process Camera Image
        # Expected output shape: (batch_size, fusion_dim)
        image_embedding = self.imageVit(data.camera_data)

        # 2. Process LIDAR Image
        # Expected output shape: (batch_size, fusion_dim)
        lidar_embedding = self.lidarVit(data.Lidar_data)

        # 3. Process Scalar Data
        # Ensure scalar components are correctly shaped (batch_size, num_features)
        gps = data.GPS_position
        accel = data.accelerometer_data
        gyro = data.gyroscope_data
        teta_val = data.teta

        # Handle potential unbatched scalar inputs from DroneData if not careful during creation
        if gps.ndim == 1: gps = gps.unsqueeze(0).expand(batch_size, -1)
        if accel.ndim == 0: accel = accel.view(1).expand(batch_size) # scalar to (B)
        if accel.ndim == 1: accel = accel.unsqueeze(1) # (B) to (B,1)
        if gyro.ndim == 1: gyro = gyro.unsqueeze(0).expand(batch_size, -1)
        if teta_val.ndim == 1: teta_val = teta_val.unsqueeze(0).expand(batch_size, -1)
        
        scalar_features = torch.cat([gps, accel, gyro, teta_val], dim=1)
        scalar_embedding = self.scalar_embedder(scalar_features) # (batch_size, fusion_dim)

        # 4. Prepare tokens for Fusion Transformer
        # Reshape embeddings to (batch_size, 1, fusion_dim) to form a sequence
        image_token = image_embedding.unsqueeze(1)
        lidar_token = lidar_embedding.unsqueeze(1)
        scalar_token = scalar_embedding.unsqueeze(1)

        # Expand CLS token for the batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # (batch_size, 1, fusion_dim)

        # Concatenate tokens: [CLS, Camera, LIDAR, Scalar]
        tokens = torch.cat([cls_tokens, image_token, lidar_token, scalar_token], dim=1) # (batch_size, 4, fusion_dim)

        # Add positional embedding
        tokens = tokens + self.pos_embedding_fusion # Broadcasting pos_embedding_fusion

        # 5. Pass through Fusion Transformer
        fused_output = self.fusion_transformer(tokens) # (batch_size, 4, fusion_dim)

        # 6. Get the representation from the CLS token
        final_representation = fused_output[:, 0] # (batch_size, fusion_dim)

        # 7. Pass to output heads
        policy_logits = self.policy_head(final_representation)
        value_estimate = self.value_head(final_representation)

        return policy_logits, value_estimate

class ImageVitModel(nn.Module):
    def __init__(self, image_size=1080, patch_size=30, num_classes=768, # Default num_classes to typical fusion_dim
                 dim=768, depth=12, heads=12, mlp_dim=3072, pool='cls', 
                 channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.vit = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes, # This is the output dim of this module
            dim=dim,                 # This is the internal working dim of ViT
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            pool=pool,
            channels=channels,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout
        )
    
    def forward(self, img) -> torch.Tensor:
        return self.vit(img)
    

class LidarVitModel(nn.Module):
    def __init__(self, image_size=1080, patch_size=30, num_classes=768, # Default num_classes to typical fusion_dim
                 dim=768, depth=12, heads=12, mlp_dim=3072, pool='cls', 
                 channels=1, # LIDAR is typically single channel
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.vit = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes, # This is the output dim of this module
            dim=dim,                 # This is the internal working dim of ViT
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            pool=pool,
            channels=channels,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout
        )

    def forward(self, img) -> torch.Tensor:
        return self.vit(img)


if  __name__ == "__main__":
    # Example usage
    # Initialize the model
    model = DroneModel(fusion_dim=768, num_actions=6) # Match fusion_dim with ViTs' num_classes
    
    # Create dummy batched data for DroneData
    batch_s = 2 # Example batch size
    timestamp = time.time() # Single timestamp for the batch or list of timestamps
    
    # Ensure scalar data is (batch_size, num_features) or (batch_size) for single features
    gps_position = torch.randn(batch_s, 3) 
    # For DroneData, accelerometer_data is expected as a scalar or 1D tensor per instance
    # If passing a batch of scalars, it should be a 1D tensor of length batch_s
    accelerometer_data_batch = torch.randn(batch_s) # Example: tensor([0.5, -0.2])

    # camera_data and lidar_data are already batched
    camera_data = torch.randn(batch_s, 3, 1080, 1080) 
    lidar_data = torch.randn(batch_s, 1, 1080, 1080) 
    
    gyroscope_data = torch.randn(batch_s, 3) 
    teta = torch.randn(batch_s, 2) 

    # DroneData might need adjustment if it doesn't inherently support batched scalar types well.
    # For simplicity, let's assume we create a list of DroneData objects for a batch,
    # or adapt DroneData/forward pass if DroneData is to hold a batch directly.
    # The current forward pass tries to adapt, but DroneData init is key.

    # Let's assume DroneData is for a single instance, and we'd loop for a batch,
    # or modify DroneData to accept batched inputs directly.
    # For this example, let's make one DroneData instance with first element of batch
    # (The model's forward pass is designed for batched Tensors within DroneData)

    print(f"Simulating for batch size: {batch_s}")
    print(f"GPS shape: {gps_position.shape}")
    print(f"Accelerometer (batch) shape: {accelerometer_data_batch.shape}")
    print(f"Gyro shape: {gyroscope_data.shape}")
    print(f"Teta shape: {teta.shape}")
    print(f"Camera shape: {camera_data.shape}")
    print(f"LIDAR shape: {lidar_data.shape}")

    # To make DroneData directly usable with the batched forward pass,
    # all its tensor fields should be batched.
    # The DroneData class currently converts single float accelerometer_data to a scalar tensor.
    # If accelerometer_data_batch (a 1D tensor of len batch_s) is passed,
    # torch.tensor(accelerometer_data_batch) will create a copy. This is fine.
    
    drone_data_instance = DroneData(
        timestamp=timestamp, # Single timestamp
        gps_position=gps_position, # (B, 3)
        lidar_data=lidar_data,     # (B, 1, 1080, 1080)
        camera_data=camera_data,   # (B, 3, 1080, 1080)
        accelerometer_data=accelerometer_data_batch, # (B) -> DroneData makes it (B)
        gyroscope_data=gyroscope_data, # (B, 3)
        teta=teta                      # (B, 2)
    )
    
    print(f"DroneData accelerometer_data shape: {drone_data_instance.accelerometer_data.shape}")


    policy_logits, value_estimate = model(drone_data_instance)
    print("\\nModel Output:")
    print("Policy Logits Shape:", policy_logits.shape) # Expected: (batch_s, num_actions)
    print("Value Estimate Shape:", value_estimate.shape) # Expected: (batch_s, 1)
    print("Policy Logits (example):", policy_logits)
    print("Value Estimate (example):", value_estimate)



