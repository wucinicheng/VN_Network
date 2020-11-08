import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from layers import GraphConvolution, WeightedGraphConvolutionNetwork

def opts2params(opts, dictionary: data.Dictionary):
    """Convert command line options to a dictionary to construct a model"""
    params = {
        "rnn_type" : opts.rnn_type,
        "direction" : opts.direction,
        "tok_len" : dictionary.tok_len(),
        "tok_emb" : opts.tok_emb,
        "tok_hid" : opts.tok_hid,
        "char_len" : dictionary.char_len(),
        "char_emb" : opts.char_emb,
        "char_hid" : opts.char_hid,
        "char_kmin" : opts.char_kmin,
        "char_kmax" : opts.char_kmax,
        "wo_char" : opts.wo_char,
        "wo_tok" : opts.wo_tok,
        "nlayers" : opts.nlayers,
        "dropout" : opts.dropout,
        "init_range" : opts.init_range,
        "tied" : opts.tied
    }
    return params

class WGCN(nn.Module):
    def __init__(self, triples, nentity, ndim0, ndim1, ndim2, ndim3, nrelation, dropout):
        super(WGCN, self).__init__()
        self.triples = triples
        self.nentity = nentity
        self.ndim0 = ndim0
        self.ndim1 = ndim1
        self.ndim2 = ndim2
        self.ndim3 = ndim3
        self.em_entity = nn.Embedding(nentity, ndim0)   # 初始化实体 embedding
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
