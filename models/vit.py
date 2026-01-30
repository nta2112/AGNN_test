import torch
import torch.nn as nn
import torchvision.models as models
from .models import register

@register('vit')
class ViTWrapper(nn.Module):
    def __init__(self, model_name='vit_b_16', pretrained=True, z_dim=128, image_size=84):
        super(ViTWrapper, self).__init__()
        
        try:
            # Modern torchvision
            if model_name == 'vit_b_16':
                weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
                self.model = models.vit_b_16(weights=weights)
            elif model_name == 'vit_b_32':
                weights = models.ViT_B_32_Weights.DEFAULT if pretrained else None
                self.model = models.vit_b_32(weights=weights)
            elif model_name == 'vit_l_16':
                weights = models.ViT_L_16_Weights.DEFAULT if pretrained else None
                self.model = models.vit_l_16(weights=weights)
            else:
                 # Fallback or error
                 weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
                 self.model = models.vit_b_16(weights=weights)
        except AttributeError:
            # Older torchvision or simplified fallback
            self.model = models.vit_b_16(pretrained=pretrained)

        # ViT B-16 dim is 768.
        # We need to replace the head.
        self.num_ftrs = self.model.heads.head.in_features
        self.model.heads.head = nn.Identity()

        self.projection = nn.Linear(self.num_ftrs, z_dim)

        self.image_size = image_size
        self.resize = None
        if image_size != 224:
             self.resize = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

        self.out_dim = z_dim

    def forward(self, x):
        if self.resize is not None:
            x = self.resize(x)
        
        x = self.model(x)
        x = self.projection(x)
        return x
