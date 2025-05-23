# vit_components.py
import torch
import torch.nn as nn
from vit_pytorch import ViT # Make sure this library is installed: pip install vit-pytorch

class ImageVitModel(nn.Module):
    def __init__(self, image_size=1080, patch_size=30, num_classes=768,
                 dim=768, depth=12, heads=12, mlp_dim=3072, pool='cls',
                 channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.vit = ViT(
            image_size=(image_size, image_size) if isinstance(image_size, int) else image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            pool=pool,
            channels=channels,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.vit(img)


class LidarVitModel(nn.Module):
    def __init__(self, image_size=1080, patch_size=30, num_classes=768,
                 dim=768, depth=12, heads=12, mlp_dim=3072, pool='cls',
                 channels=1, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.vit = ViT(
            image_size=(image_size, image_size) if isinstance(image_size, int) else image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            pool=pool,
            channels=channels,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.vit(img)

if __name__ == '__main__':
    # Example Usage
    dummy_image_data = torch.randn(1, 3, 1080, 1080) # Batch, Channels, Height, Width
    image_vit = ImageVitModel()
    image_features = image_vit(dummy_image_data)
    print(f"ImageViT output shape: {image_features.shape}") # Expected: [1, fusion_dim]

    dummy_lidar_data = torch.randn(1, 1, 1080, 1080) # Batch, Channels, Height, Width
    lidar_vit = LidarVitModel()
    lidar_features = lidar_vit(dummy_lidar_data)
    print(f"LidarViT output shape: {lidar_features.shape}") # Expected: [1, fusion_dim]