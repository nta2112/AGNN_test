import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import models
import utils
from .models import register

def gmul(input):
    W, x = input
    # x is a tensor of size (bs, N, num_features)
    # W is a tensor of size (bs, N, N, J)
    x_size = x.size()
    W_size = W.size()
    N = W_size[-2]
    W = W.split(1, 3)
    W = torch.cat(W, 1).squeeze(3) # W is now a tensor of size (bs, J*N, N)
    output = torch.bmm(W, x) # output has size (bs, J*N, num_features)
    output = output.split(N, 1)
    output = torch.cat(output, 2) # output has size (bs, N, J*num_features)
    return output

class Gconv(nn.Module):
    def __init__(self, nf_input, nf_output, J, bn_bool=True):
        super(Gconv, self).__init__()
        self.J = J
        self.num_inputs = J*nf_input
        self.num_outputs = nf_output
        self.fc = nn.Linear(self.num_inputs, self.num_outputs)

        self.bn_bool = bn_bool
        if self.bn_bool:
            self.bn = nn.BatchNorm1d(self.num_outputs)

    def forward(self, input):
        W = input[0]
        x = gmul(input) # out has size (bs, N, num_inputs)
        #if self.J == 1:
        #    x = torch.abs(x)
        x_size = x.size()
        x = x.contiguous()
        x = x.view(-1, self.num_inputs)
        x = self.fc(x) # has size (bs*N, num_outputs)
        if self.bn_bool:
            x = self.bn(x)

        x = x.view(*x_size[:-1], self.num_outputs)
        return W, x

class Wcompute(nn.Module):
    def __init__(self, input_features, nf, operator='J2', activation='softmax', ratio=[2,2,1,1], num_operators=1, drop=False):
        super(Wcompute, self).__init__()
        self.num_features = nf
        self.operator = operator
        self.conv2d_1 = nn.Conv2d(input_features, int(nf * ratio[0]), 1, stride=1)
        self.bn_1 = nn.BatchNorm2d(int(nf * ratio[0]))
        self.drop = drop
        if self.drop:
            self.dropout = nn.Dropout(0.3)
        self.conv2d_2 = nn.Conv2d(int(nf * ratio[0]), int(nf * ratio[1]), 1, stride=1)
        self.bn_2 = nn.BatchNorm2d(int(nf * ratio[1]))
        self.conv2d_3 = nn.Conv2d(int(nf * ratio[1]), nf*ratio[2], 1, stride=1)
        self.bn_3 = nn.BatchNorm2d(nf*ratio[2])
        self.conv2d_4 = nn.Conv2d(nf*ratio[2], nf*ratio[3], 1, stride=1)
        self.bn_4 = nn.BatchNorm2d(nf*ratio[3])
        self.conv2d_last = nn.Conv2d(nf, num_operators, 1, stride=1)
        self.activation = activation

    def forward(self, x, W_id):
        W1 = x.unsqueeze(2)
        W2 = torch.transpose(W1, 1, 2) #size: bs x N x N x num_features
        W_new = torch.abs(W1 - W2) #size: bs x N x N x num_features
        W_new = torch.transpose(W_new, 1, 3) #size: bs x num_features x N x N

        W_new = self.conv2d_1(W_new)
        W_new = self.bn_1(W_new)
        W_new = F.leaky_relu(W_new)
        if self.drop:
            W_new = self.dropout(W_new)

        W_new = self.conv2d_2(W_new)
        W_new = self.bn_2(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_3(W_new)
        W_new = self.bn_3(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_4(W_new)
        W_new = self.bn_4(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_last(W_new)
        W_new = torch.transpose(W_new, 1, 3) #size: bs x N x N x 1
      
        if self.activation == 'softmax':
            W_new = W_new - W_id.expand_as(W_new) * 1e8          
            W_new = torch.transpose(W_new, 2, 3)         
            # Applying Softmax
            W_new = W_new.contiguous()     

        ### old version
            W_new_size = W_new.size()
            W_new = W_new.view(-1, W_new.size(3)) #size: (bs x N) x N  

            p = 0.5
            mvalue = int(x.size(1)*p)
            [kval,_] = torch.kthvalue(W_new, mvalue, dim=1) 
            W_index_drop = torch.ge(W_new, kval.unsqueeze(1).expand_as(W_new), out=None)
            W_index_drop = W_index_drop.cuda(device=W_new.device)
            W_new = W_new.mul(W_index_drop.type(torch.float32)) 
            W_new = W_new + torch.mul(torch.ones_like(W_index_drop.type(torch.float32), device= W_new.device),-1e3)
            W_new = W_new - torch.mul(W_index_drop.type(torch.float32),-1e3)
     
            W_new = F.softmax(W_new, dim=1)                        
            W_new = W_new.view(W_new_size)
            #Softmax applied
            W_new = torch.transpose(W_new, 2, 3)
            
        elif self.activation == 'sigmoid':
            W_new = F.sigmoid(W_new)
            W_new *= (1 - W_id)
        elif self.activation == 'none':
            W_new *= (1 - W_id)
        else:
            raise (NotImplementedError)

        if self.operator == 'laplace':
            W_new = W_id - W_new
        elif self.operator == 'J2':
            W_new = torch.cat([W_id, W_new], 3)
        else:
            raise(NotImplementedError)

        return W_new

class GNN_nl(nn.Module):
    def __init__(self, input_features, nf, n_way=5):
        super(GNN_nl, self).__init__()
        self.train_N_way = n_way
        self.input_features = input_features
        self.nf = nf
        self.num_layers = 3

        for i in range(self.num_layers):
            if i == 0:
                module_w = Wcompute(self.input_features, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
                module_l = Gconv(self.input_features, int(nf / 2), 2)
            else:
                module_w = Wcompute(self.input_features + int(nf / 2) * i, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
                module_l = Gconv(self.input_features + int(nf / 2) * i, int(nf / 2), 2)
            self.add_module('layer_w{}'.format(i), module_w)
            self.add_module('layer_l{}'.format(i), module_l)

        self.w_comp_last = Wcompute(self.input_features + int(self.nf / 2) * self.num_layers, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
        self.layer_last = Gconv(self.input_features + int(self.nf / 2) * self.num_layers, self.train_N_way, 2, bn_bool=False)

    def forward(self, x):
        
        W_init = Variable(torch.eye(x.size(1),device=x.device).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3))
           
        for i in range(self.num_layers):
            Wi = self._modules['layer_w{}'.format(i)](x, W_init)               
            x_new = F.leaky_relu(self._modules['layer_l{}'.format(i)]([Wi, x])[1])
            x = torch.cat([x, x_new], 2)
           
        Wl=self.w_comp_last(x, W_init)
        out = self.layer_last([Wl, x])[1]

        return out

@register('node_att')
class node_att(nn.Module):

    def __init__(self, encoder, encoder_args={}, n_way=5, temp=10, temp_learnable=False, ablation_no_graph=False):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.gnn_model = GNN_nl(input_features=128 + n_way, nf=128, n_way=n_way)    #133,128
        # fusion and alpha have been removed as part of ablation (no node self-attention)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp
        
        self.ablation_no_graph = ablation_no_graph
        if self.ablation_no_graph:
            print("WARNING: Model is running in ABLATION MODE (No Graph). GNN modules will be skipped.")

    def forward(self, x_shot, x_query, tr_label):
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))

        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        x_shot = x_shot.view(*shot_shape, -1)
        x_query = x_query.view(*query_shape, -1)

        # --- ABLATION: NO GRAPH ---
        # If this is True, even the edge GNN is skipped (Prototypical Net mode)
        if self.ablation_no_graph:
            prototypes = x_shot.mean(dim=2) # (B, n_way, D)
            num_query = x_query.size(1)
            num_proto = prototypes.size(1)
            query_exp = x_query.unsqueeze(2).expand(-1, -1, num_proto, -1) # (B, N_Q, N_Way, D)
            proto_exp = prototypes.unsqueeze(1).expand(-1, num_query, -1, -1) # (B, N_Q, N_Way, D)
            logits = -torch.pow(query_exp - proto_exp, 2).sum(3) * self.temp # (B, N_Q, N_Way)
            return logits

        [a1,a2,a3,a4]=x_shot.size()
        x_node = torch.cat([x_shot.reshape(a1,a2*a3,a4),x_query], dim=1)         
        [b,n,_] = x_query.size()
        
        # Calculate number of support nodes to slice later
        num_support = a2 * a3 
        
        tr_label = tr_label.unsqueeze(1)
        
        # Dynamic N-way from input shape
        n_way = a2
        
        one_hot = torch.zeros((tr_label.size()[0], n_way), device=x_shot.device)
        one_hot.scatter_(1, tr_label, 1)
        one_hot_fin = one_hot.reshape(b, tr_label.size()[0]//b, n_way)     
        
        zero_pad = torch.zeros((b, n, n_way), device=x_query.device).fill_(1.0/n_way)
        label_fea = torch.cat([one_hot_fin,zero_pad], dim=1)
        
        # --- ABLATION: NO NODE SELF-ATTENTION ---
        # In the original AGNN, self-attention and fusion block are used here to compute new_fea and lab_new.
        # We skip that completely and concatenate the raw node features with the label features.
        xx = torch.cat([x_node, label_fea], dim=2)

        out_fea = self.gnn_model(xx) #b*M*d

        # Slice the queries from the output
        logits = out_fea[:, num_support:, :] * self.temp
        
        return logits
