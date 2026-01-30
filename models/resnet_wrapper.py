import torch
import torch.nn as nn
import torchvision.models as models
from .models import register

@register('resnet50')
class ResNet50Wrapper(nn.Module):
    def __init__(self, pretrained=True, z_dim=128, image_size=84):
        super(ResNet50Wrapper, self).__init__()
        # Load ResNet50
        # Use 'weights' parameter for newer torchvision, but 'pretrained' for compatibility if needed.
        # We'll try the newer 'weights' syntax if available, else fallback or use 'pretrained=True'
        try:
            # Modern torchvision
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.model = models.resnet50(weights=weights)
        except AttributeError:
            # Older torchvision
            self.model = models.resnet50(pretrained=pretrained)

        # ResNet50 output is 2048. We need to replace the fc layer or add a projection.
        # AGNN expects a feature embedding. 
        # The original model typically outputs 'x_cls'. 
        
        # We can remove the original fc and add our own projection.
        self.num_ftrs = self.model.fc.in_features # 2048
        self.model.fc = nn.Identity() # Remove the original classification head

        # Projection to z_dim (128 for AGNN)
        self.projection = nn.Linear(self.num_ftrs, z_dim)
        
        # Optional: Internal resize if we want to ensure 224x224 input behavior
        # But for efficiency on 84x84, we might skip it or make it optional.
        # If user wants best performance from pre-trained, resizing to 224 is better.
        # We'll add a flag or logic. Let's assume we want to match pre-training distribution.
        self.image_size = image_size
        self.resize = None
        if image_size < 224:
            self.resize = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
            
        self.out_dim = z_dim

    def forward(self, x):
        if self.resize is not None:
            x = self.resize(x)
            
        x = self.model(x) # Output is (N, 2048)
        x = self.projection(x) # (N, z_dim)
        return x
