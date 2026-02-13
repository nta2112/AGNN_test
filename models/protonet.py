import torch
import torch.nn as nn
import torch.nn.functional as F
import models
from .models import register

@register('protonet')
class ProtoNet(nn.Module):
    def __init__(self, encoder, encoder_args={}, n_way=5, scale_cls=10.0, learn_scale=False):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        if learn_scale:
            self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(scale_cls), requires_grad=True)
        else:
            self.scale_cls = scale_cls

    def forward(self, x_shot, x_query, tr_label=None):
        # x_shot: (local_batch, n_way, n_shot, C, H, W)
        # x_query: (local_batch, n_way * n_query, C, H, W)
        
        shot_shape = x_shot.shape[:-3] # (B, n_way, n_shot)
        query_shape = x_query.shape[:-3] # (B, n_query_total)
        img_shape = x_shot.shape[-3:]

        # Flatten for encoder
        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        
        # Combine to save one forward pass if needed, or just run separately
        # AGNN runs them together, let's follow that pattern to be safe with batch norms etc
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
        
        # Split back
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        
        # Reshape to (B, n_way, n_shot, D)
        x_shot = x_shot.view(*shot_shape, -1)
        # Reshape to (B, n_query_total, D)
        x_query = x_query.view(*query_shape, -1)
        
        # Compute Prototypes: Mean over n_shot (dim 2)
        # prototypes: (B, n_way, D)
        prototypes = x_shot.mean(dim=2)
        
        # Compute distances
        # We need distance between every query and every prototype
        # x_query: (B, N_Q, D)
        # prototypes: (B, N_Way, D)
        
        # Expand for broadcasting
        # query: (B, N_Q, 1, D)
        # proto: (B, 1, N_Way, D)
        
        # Only support batch size of 1 for simplicity first? 
        # No, the code likely supports batch size > 1 (ep_per_batch)
        
        num_query = x_query.size(1)
        num_proto = prototypes.size(1)
        
        query_exp = x_query.unsqueeze(2).expand(-1, -1, num_proto, -1)
        proto_exp = prototypes.unsqueeze(1).expand(-1, num_query, -1, -1)
        
        # Squared Euclidean Distance
        logits = -torch.pow(query_exp - proto_exp, 2).sum(3) # (B, N_Q, N_Way)
        
        if self.scale_cls != 0:
             logits = logits * self.scale_cls
        
        return logits
