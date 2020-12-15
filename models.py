import logging
import math

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.init import xavier_normal_

import data

from layers import GraphConvolution, WeightedGraphConvolutionNetwork


# GCN
class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, num_relations, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.num_relations = num_relations
        self.alpha = torch.nn.Embedding(num_relations + 1, 1, padding_idx=0)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        alp = self.alpha(adj[1]).t()[0]
        A = torch.sparse_coo_tensor(adj[0], alp, torch.Size([adj[2], adj[2]]), requires_grad=True)
        A = A + A.transpose(0, 1)
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(A, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class WGCN(nn.Module):
    def __init__(self, triples, num_entities, init_emb_size, ndim1, ndim2, ndim3, nrelation, device, dropout=0.25):

        super(WGCN, self).__init__()

        self.emb_e = torch.nn.Embedding(num_entities, init_emb_size, padding_idx=0)   # 初始化实体 embedding
        self.re_attention_weight = Parameter(torch.FloatTensor(nrelation))  # 存储的是每一个关系的权重

        self.u = nn.Parameter(torch.Tensor(ndim3 * 3))
        self.en_weight = nn.Parameter(torch.Tensor(ndim3, ndim3))
        self.re_weight = nn.Parameter(torch.Tensor(ndim3, ndim3))
        self.re_specific_attention = nn.Parameter(torch.Tensor(ndim3))
        self.wgc1 = WeightedGraphConvolutionNetwork(ndim0, ndim1, self.re_attention_weight, triples)
        self.wgc2 = WeightedGraphConvolutionNetwork(ndim1, ndim2, self.re_attention_weight, triples)
        self.wgc3 = WeightedGraphConvolutionNetwork(ndim2, ndim3, self.re_attention_weight, triples)
        self.dropout = dropout

    def forward(self):
        x = self.wgc1(self.em_entity)
        x = self.wgc2(x)
        x = self.wgc3(x)
        output_em = torch.Tensor(self.nentity, self.ndim3)
        for triple in self.triples:
            output_em[triple[0]] += torch.dot(self.get_softmax(triple[0], triple[2]), x)
        return output_em

    def get_unnormalized_attention_weight(self, h_i, h_j):
        unnormalized_attention_weight = torch.nn.LeakyReLU(torch.dot(self.u,
                                                                     torch.cat(torch.mm(self.en_weight, h_i),
                                                                                       torch.mm(self.re_weight, self.re_specific_attention),
                                                                                       torch.mm(self.en_weight, h_j))))
        return unnormalized_attention_weight

    def get_softmax(self, h_i, h_j):
        up = torch.exp(self.get_unnormalized_attention_weight(h_i=h_i, h_j=h_j))
        down = 0
        for triple in self.triples:
            down += torch.exp(self.get_unnormalized_attention_weight(triple[0], triple[2]))
        return up / down


class StructureAwareLayer(nn.Module):
    def __init__(self):
        super(StructureAwareLayer, self).__init__()


class DistMult(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(DistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel, X, A): # X and A haven't been used here.
        e1_embedded= self.emb_e(e1)
        rel_embedded= self.emb_rel(rel)
        e1_embedded = e1_embedded.squeeze()
        rel_embedded = rel_embedded.squeeze()

        e1_embedded = self.inp_drop(e1_embedded)
        rel_embedded = self.inp_drop(rel_embedded)

        pred = torch.mm(e1_embedded*rel_embedded, self.emb_e.weight.transpose(1,0))
        pred = torch.sigmoid(pred)

        return pred

