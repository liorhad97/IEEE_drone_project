import torch
import torch.nn as nn
import torchvision.models as models
from torch.distributions import Normal

class DroneModel(nn.Module): # Actor-Critic Model
    def __init__(self, fusion_dim, scalar_input_dim, vit_image_size, vit_patch_size, 
                 num_control_outputs, vit_model_dim=768): # Removed learning_rate
        super(DroneModel, self).__init__()

        # --- Shared Feature Extraction Backbone ---
        self.camera_vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        # Freeze parameters of the pre-trained ViT
        # for param in self.camera_vit.parameters():
        #     param.requires_grad = False
        self.camera_vit.heads.head = nn.Linear(vit_model_dim, fusion_dim) # Replace the classifier head
        
        self.lidar_vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        # Freeze parameters of the pre-trained ViT
        # for param in self.lidar_vit.parameters():
        #    param.requires_grad = False
            
        original_conv_proj = self.lidar_vit.conv_proj
        self.lidar_vit.conv_proj = nn.Conv2d(
            in_channels=1, 
            out_channels=original_conv_proj.out_channels, 
            kernel_size=original_conv_proj.kernel_size, 
            stride=original_conv_proj.stride,
            padding=original_conv_proj.padding,
            dilation=original_conv_proj.dilation,
            groups=original_conv_proj.groups,
            bias=(original_conv_proj.bias is not None)
        )
        if original_conv_proj.bias is not None:
            # Initialize bias if it exists
            nn.init.zeros_(self.lidar_vit.conv_proj.bias)
        # Initialize weights (example: Kaiming uniform)
        # For the new conv layer, initialize its weights. 
        # We are not copying mean of weights from original conv_proj, as it might not be optimal for single channel.
        nn.init.kaiming_uniform_(self.lidar_vit.conv_proj.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.lidar_vit.heads.head = nn.Linear(vit_model_dim, fusion_dim) # Replace the classifier head

        self.scalar_processor = nn.Linear(scalar_input_dim, fusion_dim)
        # Adjusted fusion_layer input to match the sum of individual fusion_dims
        self.fusion_layer = nn.Linear(fusion_dim * 3, vit_model_dim) # Assuming fusion_dim for each of 3 modalities
        self.relu = nn.ReLU()

        # --- Actor Head (Policy) ---
        self.actor_head = nn.Sequential(
            nn.Linear(vit_model_dim, vit_model_dim // 2),
            nn.ReLU(),
            nn.Linear(vit_model_dim // 2, num_control_outputs) # Outputs mean of the action distribution
        )
        # Learnable log standard deviation for the action distribution
        self.log_std = nn.Parameter(torch.zeros(num_control_outputs))

        # --- Critic Head (Value Function) ---
        self.critic_head = nn.Sequential(
            nn.Linear(vit_model_dim, vit_model_dim // 2),
            nn.ReLU(),
            nn.Linear(vit_model_dim // 2, 1) # Outputs a single value for the state
        )

    def _extract_features(self, drone_data_obj):
        camera_img, lidar_img, gps, accel, gyro, teta = drone_data_obj.to_tensor_tuple()
        # Ensure tensors are on the same device as the model parameters
        device = next(self.parameters()).device
        
        # Add batch dimension if not present (assuming single instance processing for now)
        camera_img = camera_img.to(device).unsqueeze(0) if camera_img.ndim == 3 else camera_img.to(device)
        lidar_img = lidar_img.to(device).unsqueeze(0) if lidar_img.ndim == 3 else lidar_img.to(device)
        
        # Ensure scalar inputs are 2D [batch_size, num_features]
        scalar_inputs = torch.cat([gps, accel, gyro, teta], dim=-1).to(device)
        if scalar_inputs.ndim == 1:
            scalar_inputs = scalar_inputs.unsqueeze(0)
        # If scalar_inputs was [N] (e.g. for accel which is 1D) after cat, it might become [M] where M is sum of dims.
        # It needs to be [Batch, Sum_of_dims]. If batch is 1, then [1, Sum_of_dims].
        # The unsqueeze(0) above handles the case where the initial individual tensors were 0D or 1D without batch.
        # If they are already [Batch, Dim], cat will produce [Batch, Sum_of_dims].

        cam_embedding = self.camera_vit(camera_img)
        lidar_embedding = self.lidar_vit(lidar_img)
        scalar_embedding = self.scalar_processor(scalar_inputs)

        fused = torch.cat((cam_embedding, lidar_embedding, scalar_embedding), dim=1)
        shared_features = self.relu(self.fusion_layer(fused))
        return shared_features

    def act(self, drone_data_obj, deterministic=False):
        shared_features = self._extract_features(drone_data_obj)
        action_mean = self.actor_head(shared_features)
        action_std = torch.exp(self.log_std.expand_as(action_mean)) # Ensure log_std is expanded correctly
        dist = Normal(action_mean, action_std)

        if deterministic:
            action = action_mean
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(axis=-1) # Sum log_probs for multi-dimensional actions
        state_value = self.critic_head(shared_features)
        
        # Ensure outputs are detached if they are not used for gradient computation directly here
        return action.detach(), log_prob.detach(), state_value.detach()

    def evaluate(self, drone_data_obj, action_taken):
        shared_features = self._extract_features(drone_data_obj)
        action_mean = self.actor_head(shared_features)
        action_std = torch.exp(self.log_std.expand_as(action_mean))
        dist = Normal(action_mean, action_std)

        log_prob = dist.log_prob(action_taken).sum(axis=-1)
        dist_entropy = dist.entropy().sum(axis=-1)
        state_value = self.critic_head(shared_features)

        return log_prob, state_value, dist_entropy