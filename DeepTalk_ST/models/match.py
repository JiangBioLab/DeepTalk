from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
from scipy import spatial
import time
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                # layers.append(nn.BatchNorm1d(channels[i]))
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image         print(desc0.shape)
image_shape"""
    #print(image_shape)
    #_, _, height, width = image_shape
    height, width = image_shape.numpy()[0],image_shape.numpy()[1]
    #height, width = image_shape.cpu().numpy()[0][0],image_shape.cpu().numpy()[0][1]
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]

class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([2] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts):
        #inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        inputs = [kpts.transpose(1, 2)]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class My_loss_new(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, M, S, T):

        SSS = (torch.mm((M[0,:,:].softmax(dim=-1)).transpose(0,1),S[:,0,:].transpose(0,1)))
        T1 = T[:,0,:].transpose(0,1)

        CT_COR = F.cosine_similarity(SSS,T1,dim=0)
        CT_COR_mean0 = torch.mean((CT_COR))
        CT_COR_mean0 = torch.reshape(CT_COR_mean0, (1, -1))

        CT_COR1 = F.cosine_similarity(SSS,T1,dim=1)
        CT_COR_mean1 = torch.mean((CT_COR1))
        CT_COR_mean1 = torch.reshape(CT_COR_mean1, (1, -1))

        loss = - 1*CT_COR_mean0[0] +0*CT_COR_mean1[0]
        #loss = - CT_COR_mean[0]
        score0 = CT_COR_mean0[0]
        score1 = CT_COR_mean1[0]
        return loss, score0, score1

class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value)
        self.prob.append(prob)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 1)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            layer.attn.prob = []
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class Matchsc_st(nn.Module):

    default_config = {
        'keypoint_encoder': [8,16,32],
        'GNN_layers': ['self','cross'] * 1,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data['descrip0Scale'], data['descrip1Scale']
        #desc0, desc1 = data['descriptors0'], data['descriptors1']

        kpts0, kpts1 = data['keypoints0'], data['keypoints1']

        desc0 = desc0.transpose(0,1)
        desc1 = desc1.transpose(0,1)

        kpts0 = torch.reshape(kpts0, (1, -1, 2))
        kpts1 = torch.reshape(kpts1, (1, -1, 2))

        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['sc_location'])
        kpts1 = normalize_keypoints(kpts1, data['st_location'])
        #print(data['sc_location'].shape)
        #print(kpts0.shape)
        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0)
        desc1 = desc1 + self.kenc(kpts1)
        #print(desc1.shape)
        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        #desc0, desc1 = self.final_proj(desc0), self.final_proj(desc1)
        scores = torch.einsum('bdn,bdm->bnm', desc0, desc1)
        #scores = scores / self.config['descriptor_dim']**.5
        scores = scores / self.config['descriptor_dim']

        return {
            'scores':scores
            }

    def get_loss(self,scores,data):

        SC = data['descriptors0'].float()
        ST = data['descriptors1'].float()

        loss_fun = My_loss_new()

        loss,score0,score1 = loss_fun(scores['scores'].float(), SC, ST)

        return loss,score0,score1




