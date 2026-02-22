import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from .models import register

class Graph_conv_block(nn.Module):
    def __init__(self, input_dim, output_dim, use_bn=True):
        super(Graph_conv_block, self).__init__()

        self.weight = nn.Linear(input_dim, output_dim)
        if use_bn:
            self.bn = nn.BatchNorm1d(output_dim)
        else:
            self.bn = None

    def forward(self, x, A):
        # x: (b, N, input_dim)
        # A: (b, N, N)
        x_next = torch.matmul(A, x) # (b, N, input_dim)
        x_next = self.weight(x_next) # (b, N, output_dim)

        if self.bn is not None:
            x_next = torch.transpose(x_next, 1, 2) # (b, output_dim, N)
            x_next = x_next.contiguous()
            x_next = self.bn(x_next)
            x_next = torch.transpose(x_next, 1, 2) # (b, N, output_dim)

        return x_next

class Adjacency_layer(nn.Module):
    def __init__(self, input_dim, hidden_dim, ratio=[2, 1]):
        super(Adjacency_layer, self).__init__()

        module_list = []
        for i in range(len(ratio)):
            if i == 0:
                module_list.append(nn.Conv2d(input_dim, hidden_dim * ratio[i], 1, 1))
            else:
                module_list.append(nn.Conv2d(hidden_dim * ratio[i-1], hidden_dim * ratio[i], 1, 1))
            module_list.append(nn.BatchNorm2d(hidden_dim * ratio[i]))
            module_list.append(nn.LeakyReLU())

        module_list.append(nn.Conv2d(hidden_dim * ratio[-1], 1, 1, 1))
        self.module_list = nn.ModuleList(module_list)

    def forward(self, x):
        X_i = x.unsqueeze(2) # (b, N , 1, input_dim)
        X_j = torch.transpose(X_i, 1, 2) # (b, 1, N, input_dim)

        phi = torch.abs(X_i - X_j) # (b, N, N, input_dim)
        phi = torch.transpose(phi, 1, 3) # (b, input_dim, N, N)

        A = phi
        for l in self.module_list:
            A = l(A)
        # (b, 1, N, N)

        A = torch.transpose(A, 1, 3) # (b, N, N, 1)
        A = F.softmax(A, 2) # normalize
        return A.squeeze(3) # (b, N, N)

class GNN_module(nn.Module):
    def __init__(self, nway, input_dim, hidden_dim, num_layers, feature_type='dense'):
        super(GNN_module, self).__init__()
        self.feature_type = feature_type

        adjacency_list = []
        graph_conv_list = []
        ratio = [2, 1]

        if self.feature_type == 'dense':
            for i in range(num_layers):
                adjacency_list.append(Adjacency_layer(
                    input_dim=input_dim + hidden_dim // 2 * i, 
                    hidden_dim=hidden_dim, 
                    ratio=ratio))

                graph_conv_list.append(Graph_conv_block(
                    input_dim=input_dim + hidden_dim // 2 * i, 
                    output_dim=hidden_dim // 2))

            # last layer
            self.last_adjacency = Adjacency_layer(
                        input_dim=input_dim + hidden_dim // 2 * num_layers, 
                        hidden_dim=hidden_dim, 
                        ratio=ratio)

            self.last_conv = Graph_conv_block(
                    input_dim=input_dim + hidden_dim // 2 * num_layers, 
                    output_dim=nway, 
                    use_bn=False)
            
        else: # 'forward' - simplified for brevity, default is dense
            raise NotImplementedError

        self.adjacency_list = nn.ModuleList(adjacency_list)
        self.graph_conv_list = nn.ModuleList(graph_conv_list)

    def forward(self, x):
        for i, _ in enumerate(self.adjacency_list):
            adjacency_layer = self.adjacency_list[i]
            conv_block = self.graph_conv_list[i]

            A = adjacency_layer(x)
            x_next = conv_block(x, A)
            x_next = F.leaky_relu(x_next, 0.1)

            if self.feature_type == 'dense':
                x = torch.cat([x, x_next], dim=2)

        A = self.last_adjacency(x)
        out = self.last_conv(x, A)   
        return out

@register('baseline_gnn')
class baseline_gnn(nn.Module):
    """
    Garcia & Bruna (2018) Few-Shot Learning with Graph Neural Networks Baseline.
    Uses Adjacency_layer with Conv2D MLPs to learn the metric for generating edge weights.
    """
    def __init__(self, encoder, encoder_args={}, n_way=5, temp=10, temp_learnable=False):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        
        # input_features = 128 (backbone) + n_way (one-hot encode support labels)
        self.gnn_model = GNN_module(nway=n_way, input_dim=128 + n_way, hidden_dim=128, num_layers=3, feature_type='dense')
        
        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward(self, x_shot, x_query, tr_label):
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        
        # 1. Extract Backbone Features
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))

        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        x_shot = x_shot.view(*shot_shape, -1)
        x_query = x_query.view(*query_shape, -1)

        [a1, a2, a3, a4] = x_shot.size()
        x_support = x_shot.reshape(a1, a2*a3, a4)
        
        # Combine support and query nodes
        x_node = torch.cat([x_support, x_query], dim=1)         
        [b, n, _] = x_query.size()
        num_support = a2 * a3 
        
        # 2. Construct Label Features
        tr_label = tr_label.unsqueeze(1)
        n_way = a2 
        
        one_hot = torch.zeros((tr_label.size()[0], n_way), device=x_shot.device)
        one_hot.scatter_(1, tr_label, 1)
        one_hot_fin = one_hot.reshape(b, tr_label.size()[0]//b, n_way)     
        
        zero_pad = torch.zeros((b, n, n_way), device=x_query.device).fill_(1.0/n_way)
        label_fea = torch.cat([one_hot_fin, zero_pad], dim=1)

        # 3. Concatenate node features and label features
        xx = torch.cat([x_node, label_fea], dim=2) 

        # 4. Standard GNN Forward Pass (Garcia & Bruna, 2018)
        out_fea = self.gnn_model(xx) 

        # 5. Extract Query Predictions
        logits = out_fea[:, num_support:, :] * self.temp
        return logits
