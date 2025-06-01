import torch
import torch.nn as nn
import torchvision.models as models
from torch.distributions import Normal

class DroneModel(nn.Module): # Actor-Critic Model
    def __init__(self, fusion_dim, scalar_input_dim, vit_image_size, vit_patch_size, 
                 num_control_outputs, vit_model_dim=768): # Removed learning_rate
        super(DroneModel, self).__init__()

        # Input normalization parameters (learnable)
        self.register_buffer('scalar_mean', torch.zeros(scalar_input_dim))
        self.register_buffer('scalar_std', torch.ones(scalar_input_dim))
        
        # --- Shared Feature Extraction Backbone ---
        # Use ViT but smaller and with more frozen layers for efficiency
        self.camera_vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        # Freeze parameters of the pre-trained ViT - most layers should be frozen
        for name, param in self.camera_vit.named_parameters():
            # Only keep the last few transformer blocks trainable
            if 'blocks.11' not in name and 'heads' not in name:
                param.requires_grad = False
        self.camera_vit.heads.head = nn.Linear(vit_model_dim, fusion_dim) # Replace the classifier head
        
        self.lidar_vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        # Freeze parameters of the pre-trained ViT
        for name, param in self.lidar_vit.named_parameters():
            # Only keep the last few transformer blocks trainable
            if 'blocks.11' not in name and 'heads' not in name:
                param.requires_grad = False
            
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
        nn.init.kaiming_normal_(self.lidar_vit.conv_proj.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.lidar_vit.heads.head = nn.Linear(vit_model_dim, fusion_dim) # Replace the classifier head

        self.scalar_processor = nn.Linear(scalar_input_dim, fusion_dim)
        # Adjusted fusion_layer input to match the sum of individual fusion_dims
        self.fusion_layer = nn.Linear(fusion_dim * 3, vit_model_dim) # Assuming fusion_dim for each of 3 modalities
        self.fusion_layer_2 = nn.Linear(vit_model_dim, vit_model_dim) # Added layer
        self.fusion_layer_3 = nn.Linear(vit_model_dim, vit_model_dim) # Added another layer
        self.relu = nn.ReLU()
        
        # --- Actor Head (Policy) ---
        self.actor_head = nn.Sequential(
            nn.Linear(vit_model_dim, vit_model_dim // 2),
            nn.ReLU(),
            nn.Linear(vit_model_dim // 2, vit_model_dim // 2), 
            nn.ReLU(), 
            nn.Linear(vit_model_dim // 2, vit_model_dim // 2), # Added another layer
            nn.ReLU(), # Added activation
            nn.Linear(vit_model_dim // 2, num_control_outputs) # Outputs mean of the action distribution
        )
        # Learnable log standard deviation for the action distribution with better initialization
        self.log_std = nn.Parameter(torch.ones(num_control_outputs) * -0.5)  # Start with smaller stdev

        # --- Critic Head (Value Function) ---
        self.critic_head = nn.Sequential(
            nn.Linear(vit_model_dim, vit_model_dim // 2),
            nn.ReLU(),
            nn.Linear(vit_model_dim // 2, vit_model_dim // 2), 
            nn.ReLU(), 
            nn.Linear(vit_model_dim // 2, vit_model_dim // 2), # Added another layer
            nn.ReLU(), # Added activation
            nn.Linear(vit_model_dim // 2, 1) # Outputs a single value for the state
        )

    def _extract_features(self, drone_data_obj):
        camera_img, lidar_img, gps, accel, gyro, teta = drone_data_obj.to_tensor_tuple()
        # Ensure tensors are on the same device as the model parameters
        device = next(self.parameters()).device
        
        # Add batch dimension if not present (assuming single instance processing for now)
        if camera_img.dim() == 3:
            camera_img = camera_img.unsqueeze(0)
        if lidar_img.dim() == 3:
            lidar_img = lidar_img.unsqueeze(0)
            
        # For scalar inputs, ensure they all have the same dimensions before concatenating
        scalar_inputs = [t.float() for t in [gps, accel, gyro, teta]]
        
        # Add batch dimension if not present
        scalar_inputs = [s.unsqueeze(0) if s.dim() == 1 else s for s in scalar_inputs]
            
        # Concatenate scalar inputs along dimension 1 (feature dimension)
        scalar_inputs = torch.cat(scalar_inputs, dim=1)
        
        # Apply input normalization to scalar data (running average)
        with torch.no_grad():
            if self.training:
                # Update normalization parameters during training
                batch_mean = scalar_inputs.mean(dim=0) if scalar_inputs.size(0) > 1 else scalar_inputs[0]
                batch_std = scalar_inputs.std(dim=0) + 1e-8 if scalar_inputs.size(0) > 1 else torch.ones_like(scalar_inputs[0])
                
                # Exponential moving average updates for mean and std
                momentum = 0.01
                self.scalar_mean = (1 - momentum) * self.scalar_mean + momentum * batch_mean
                self.scalar_std = (1 - momentum) * self.scalar_std + momentum * batch_std
        
        # Normalize scalar inputs
        scalar_inputs = (scalar_inputs - self.scalar_mean) / self.scalar_std
        
        cam_embedding = self.camera_vit(camera_img)
        lidar_embedding = self.lidar_vit(lidar_img)
        scalar_embedding = self.scalar_processor(scalar_inputs)

        fused = torch.cat((cam_embedding, lidar_embedding, scalar_embedding), dim=1)
 
        fusedd = self.fusion_layer(fused)
        fusedd = self.relu(fusedd) # Activation for first fusion layer
        fusedd = self.fusion_layer_2(fusedd) # Pass through second fusion layer
        fusedd = self.relu(fusedd) # Activation for second fusion layer
        fusedd = self.fusion_layer_3(fusedd) # Pass through third fusion layer
        shared_features = self.relu(fusedd) # Activation for third fusion layer
        return shared_features

    def act(self, drone_data_obj, deterministic=False):
        shared_features = self._extract_features(drone_data_obj)
        action_mean = self.actor_head(shared_features)
        # Clamp action means to prevent extreme values
        action_mean = torch.tanh(action_mean)  # Constrain to [-1, 1]
        
        # Get action standard deviation with minimum value to prevent collapse
        action_std = torch.exp(self.log_std.expand_as(action_mean)).clamp(min=1e-3, max=1.0)
        
        dist = Normal(action_mean, action_std)
        if deterministic:
            action = action_mean
        else:
            action = dist.sample()
            # Limit debugging output to not overwhelm logs
            if torch.rand(1).item() < 0.01:  # Only print occasionally
                print(f"Action mean: {action_mean.detach().cpu().numpy()}, std: {action_std.detach().cpu().numpy()}")
        
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
    
    # if printed it will show the model architecture
    def __repr__(self):
        parts = [f"{self.__class__.__name__}("]

        # Camera ViT
        parts.append(f"  (camera_vit): ViT_B_16 (pretrained)")
        parts.append(f"    (heads.head): Linear(in_features={self.camera_vit.heads.head.in_features}, out_features={self.camera_vit.heads.head.out_features})")

        # Lidar ViT
        parts.append(f"  (lidar_vit): ViT_B_16 (pretrained)")
        parts.append(f"    (conv_proj): Conv2d(in_channels={self.lidar_vit.conv_proj.in_channels}, out_channels={self.lidar_vit.conv_proj.out_channels}, kernel_size={self.lidar_vit.conv_proj.kernel_size}, stride={self.lidar_vit.conv_proj.stride}, padding={self.lidar_vit.conv_proj.padding})")
        parts.append(f"    (heads.head): Linear(in_features={self.lidar_vit.heads.head.in_features}, out_features={self.lidar_vit.heads.head.out_features})")

        # Scalar Processor
        parts.append(f"  (scalar_processor): {self.scalar_processor}")

        # Fusion Layer
        parts.append(f"  (fusion_layer): {self.fusion_layer}")
        parts.append(f"  (relu): ReLU()")

        # Actor Head
        actor_head_parts = ["  (actor_head): Sequential("]
        for i, layer in enumerate(self.actor_head):
            actor_head_parts.append(f"    ({i}): {layer}")
        actor_head_parts.append("  )")
        parts.extend(actor_head_parts)

        # Log Std
        parts.append(f"  (log_std): Parameter(shape={list(self.log_std.shape)})")

        # Critic Head
        critic_head_parts = ["  (critic_head): Sequential("]
        for i, layer in enumerate(self.critic_head):
            critic_head_parts.append(f"    ({i}): {layer}")
        critic_head_parts.append("  )")
        parts.extend(critic_head_parts)
        
        parts.append(")")
        return "\n\n".join(parts)