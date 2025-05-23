# import torch
# import torch.nn as nn
# import torch.optim as optim
# from drone_data import DroneData
# from vit_pytorch import ViT
# import time
# from collections import deque
# import time

# class DroneModel(nn.Module):
#     """
#     A class representing a drone in the IEEE Drone Project, using a multimodal
#     transformer architecture for state representation and decision making.
#     """
#     def __init__(self, 
#                  fusion_dim=768, 
#                  scalar_input_dim=9, 
#                  vit_internal_dim=768,
#                  vit_depth=12,
#                  vit_heads=12,
#                  vit_mlp_dim=3072,
#                  vit_patch_size=30,
#                  fusion_depth=3, 
#                  fusion_heads=8, 
#                  fusion_mlp_dim_factor=4,
#                  fusion_dropout=0.1,
#                  num_actions=6, 
#                  optimizer_class=optim.Adam, 
#                  learning_rate=0.001):
        
#         super(DroneModel, self).__init__()
#         self.learning_rate = learning_rate
#         self.fusion_dim = fusion_dim

#         self.imageVit = ImageVitModel(
#             image_size=1080, 
#             patch_size=vit_patch_size, 
#             num_classes=fusion_dim,
#             dim=vit_internal_dim, 
#             depth=vit_depth, 
#             heads=vit_heads, 
#             mlp_dim=vit_mlp_dim, 
#             channels=3
#         )
       
#         self.lidarVit = LidarVitModel(
#             image_size=1080, 
#             patch_size=vit_patch_size, 
#             num_classes=fusion_dim, 
#             dim=vit_internal_dim, 
#             depth=vit_depth, 
#             heads=vit_heads, 
#             mlp_dim=vit_mlp_dim, 
#             channels=1 
#         )
#         self.optimizer = optimizer_class(self.parameters(), lr=self.learning_rate) # Corrected optimizer initialization

#         self.scalar_embedder = nn.Sequential(
#             nn.Linear(scalar_input_dim, fusion_dim // 2),
#             nn.ReLU(),
#             nn.Linear(fusion_dim // 2, fusion_dim)
#         )

#         self.cls_token = nn.Parameter(torch.randn(1, 1, fusion_dim))
#         self.cls_token.requires_grad = True
#         self.pos_embedding_fusion = nn.Parameter(torch.randn(1, 4, fusion_dim))

#         fusion_transformer_layer = nn.TransformerEncoderLayer(
#             d_model=fusion_dim, 
#             nhead=fusion_heads, 
#             dim_feedforward=fusion_dim * fusion_mlp_dim_factor, 
#             dropout=fusion_dropout, 
#             batch_first=True
#         )
#         self.fusion_transformer = nn.TransformerEncoder(
#             fusion_transformer_layer, 
#             num_layers=fusion_depth
#         )

#         self.policy_head = nn.Linear(fusion_dim, num_actions)
#         self.value_head = nn.Linear(fusion_dim, 1)
        


#     def forward(self, data: DroneData) -> torch.Tensor: # Changed return type
#         camera_input = data.camera_data
#         if camera_input.ndim == 3:
#             camera_input = camera_input.unsqueeze(0)
#         image_embedding = self.imageVit(camera_input)
#         lidar_input = data.Lidar_data # Corrected attribute name based on DroneData
#         if lidar_input.ndim == 3: # Lidar data might be [C, H, W]
#             lidar_input = lidar_input.unsqueeze(0)
#         elif lidar_input.ndim == 2: # Or [H, W], so add channel and batch dim
#             lidar_input = lidar_input.unsqueeze(0).unsqueeze(0)
#         lidar_embedding = self.lidarVit(lidar_input)
#         # Assuming lidar_embedding is [1, fusion_dim]

#         gps = data.GPS_position # Corrected attribute name based on DroneData
#         accel = data.accelerometer_data
#         gyro = data.gyroscope_data
#         teta_val = data.teta

#         if not isinstance(gps, torch.Tensor): gps = torch.tensor(gps)
#         if not isinstance(accel, torch.Tensor): accel = torch.tensor([accel]) # accel is float, make it tensor
#         if not isinstance(gyro, torch.Tensor): gyro = torch.tensor(gyro)
#         if not isinstance(teta_val, torch.Tensor): teta_val = torch.tensor(teta_val)

#         device = self.cls_token.device
#         gps = gps.to(device).float()
#         accel = accel.to(device).float()
#         gyro = gyro.to(device).float()
#         teta_val = teta_val.to(device).float()
#         image_embedding = image_embedding.to(device).float()
#         lidar_embedding = lidar_embedding.to(device).float()


#         scalar_features_flat = torch.cat([
#             gps.view(-1), accel.view(-1), gyro.view(-1), teta_val.view(-1)
#         ], dim=0)
#         scalar_features_batched = scalar_features_flat.unsqueeze(0)
#         scalar_embedding = self.scalar_embedder(scalar_features_batched)

#         image_token = image_embedding.unsqueeze(1)
#         lidar_token = lidar_embedding.unsqueeze(1)
#         scalar_token = scalar_embedding.unsqueeze(1)

#         batch_size = image_token.shape[0]
#         cls_tokens = self.cls_token.expand(batch_size, -1, -1)

#         tokens = torch.cat([cls_tokens, image_token, lidar_token, scalar_token], dim=1)
        
#         if batch_size > 1 and self.pos_embedding_fusion.shape[0] == 1:
#             pos_embedding = self.pos_embedding_fusion.expand(batch_size, -1, -1)
#         else:
#             pos_embedding = self.pos_embedding_fusion
        
#         tokens = tokens + pos_embedding

#         fused_output = self.fusion_transformer(tokens)
#         final_representation = fused_output[:, 0]  # Get the CLS token representation

#         policy_logits = self.policy_head(final_representation) # Shape [batch_size, num_actions]

#         return policy_logits # Return Q-values (logits)

# class ImageVitModel(nn.Module):
#     def __init__(self, image_size=1080, patch_size=30, num_classes=768,
#                  dim=768, depth=12, heads=12, mlp_dim=3072, pool='cls', 
#                  channels=3, dim_head=64, dropout=0., emb_dropout=0.):
#         super().__init__()
#         self.vit = ViT(
#             image_size=(image_size, image_size),
#             patch_size=patch_size,
#             num_classes=num_classes,
#             dim=dim,                
#             depth=depth,
#             heads=heads,
#             mlp_dim=mlp_dim,
#             pool=pool,
#             channels=channels,
#             dim_head=dim_head,
#             dropout=dropout,
#             emb_dropout=emb_dropout
#         )
    
#     def forward(self, img) -> torch.Tensor:
#         return self.vit(img)
    

# class LidarVitModel(nn.Module):
#     def __init__(self, image_size=1080, patch_size=30, num_classes=768, 
#                  dim=768, depth=12, heads=12, mlp_dim=3072, pool='cls', 
#                  channels=1,
#                  dim_head=64, dropout=0., emb_dropout=0.
#                  ):
#         super().__init__()
#         self.vit = ViT(
#             image_size=(image_size, image_size),
#             patch_size=patch_size,
#             num_classes=num_classes, 
#             dim=dim,               
#             depth=depth,
#             heads=heads,
#             mlp_dim=mlp_dim,
#             pool=pool,
#             channels=channels,
#             dim_head=dim_head,
#             dropout=dropout,
#             emb_dropout=emb_dropout
#         )

#     def forward(self, img) -> torch.Tensor:
#         return self.vit(img)

# if __name__ == "__main__":

#     model = DroneModel(fusion_dim=768, num_actions=6, learning_rate=0.001)
#     model.train()
#     loss_fn = nn.MSELoss()

#     replay_buffer = deque(maxlen=100)

#     # Generate random input data for current state
#     current_camera_data = torch.randn(3, 1080, 1080)
#     current_lidar_data = torch.randn(1, 1080, 1080)
#     current_gps = torch.randn(3)
#     current_accel = torch.randn(1)
#     current_gyro = torch.randn(3)
#     current_teta = torch.randn(2)

#     current_drone_data = DroneData(
#         timestamp=time.time(),
#         camera_data=current_camera_data,
#         lidar_data=current_lidar_data,
#         gps_position=current_gps,
#         accelerometer_data=current_accel,
#         gyroscope_data=current_gyro,
#         teta=current_teta
#     )

#     q_values = model(current_drone_data) 
#     print("Q-values:", q_values)
#     action = torch.argmax(q_values, dim=1).item()
#     print("Selected action:", action)

#     # Simulate reward and next state
#     reward = 1.0
#     done = False  # Assume episode continues
#     gamma = 0.99  # Discount factor

#     next_camera_data = torch.randn(3, 1080, 1080)
#     next_lidar_data = torch.randn(1, 1080, 1080)
#     next_gps = torch.randn(3)
#     next_accel = torch.randn(1)
#     next_gyro = torch.randn(3)
#     next_teta = torch.randn(2)

#     next_drone_data = DroneData(
#         timestamp=time.time(),
#         camera_data=next_camera_data,
#         lidar_data=next_lidar_data,
#         gps_position=next_gps,
#         accelerometer_data=next_accel,
#         gyroscope_data=next_gyro,
#         teta=next_teta
#     )
    
#     experience = (current_drone_data, action, reward, next_drone_data, done)
#     replay_buffer.append(experience)
#     print("Experience stored in deque")    
#     state, action, reward, next_state, done = replay_buffer[0]

#     q_values = model(state) 
#     predicted_q = q_values[0, action].unsqueeze(0) 

#     with torch.no_grad():
#         next_q_values = model(next_state) 
#         max_next_q_tensor = torch.max(next_q_values, dim=1)[0] # Shape (1,)

#         if done:
#             target = torch.tensor(reward, device=max_next_q_tensor.device, dtype=max_next_q_tensor.dtype)
#         else:
#             target = reward + gamma * max_next_q_tensor.squeeze()
#     print("Target Q-value:", target.item())

#     # Compute loss
#     loss = loss_fn(predicted_q, target)
#     print("Loss:", loss.item())

#     model.optimizer.zero_grad()
#     loss.backward()
#     model.optimizer.step()
#     print("Weights updated successfully!")