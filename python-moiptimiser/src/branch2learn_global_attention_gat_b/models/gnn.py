"""
GNN models.

File adapted from https://github.com/ds4dm/ecole
by Lars Sandberg @Sandbergo
May 2021
"""

import torch

import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import global_mean_pool, MessagePassing, GlobalAttention, global_add_pool
from torch_geometric.utils import remove_self_loops

from utilities.model import BipartiteGraphConvolution, BipartiteGIN

import math
import random
import numpy as np
import torch.nn as nn



class PositionwiseFeedForward(nn.Module):
    """ Implements a two layer feed-forward network.
    """

    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)
    

    def forward(self, x):
        """ x: (batch_size, seq_len, d_model)
        """
        x = self.dropout(F.relu(self.w_1(x)))  # (batch_size, seq_len, d_ff)
        x = self.w_2(x)  # (batch_size, seq_len, d_model)

        # x: (batch_size, seq_len, d_model)
        return x




class ScaledDotProductAttention(nn.Module):
    """ Computes scaled dot product attention
    """

    def __init__(self, scale, dropout_rate):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = scale
        self.dropout_rate = dropout_rate
        
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, query, key, value, mask=None):
        """ query: (batch_size, n_heads, query_len, head_dim)
            key: (batch_size, n_heads, key_len, head_dim)
            value: (batch_size, n_heads, value_len, head_dim)
            mask: (batch_size, 1, 1, source_seq_len) for source mask
                  (batch_size, 1, target_seq_len, target_seq_len) for target mask
        """
        # calculate alignment scores
        scores = torch.matmul(query, key.transpose(-2, -1))  # (batch_size, n_heads, query_len, value_len)
        scores = scores / self.scale  # (batch_size, num_heads, query_len, value_len)

        # mask out invalid positions
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))  # (batch_size, n_heads, query_len, value_len)

        # calculate the attention weights (prob) from alignment scores
        attn_probs = F.softmax(scores, dim=-1)  # (batch_size, n_heads, query_len, value_len)
        
        # calculate context vector
        output = torch.matmul(self.dropout(attn_probs), value)  # (batch_size, n_heads, query_len, head_dim)

        # output: (batch_size, n_heads, query_len, head_dim)
        # attn_probs: (batch_size, n_heads, query_len, value_len)
        return output, attn_probs




class MultiHeadAttention(nn.Module):
    """ Implements Multi-Head Self-Attention proposed by Vaswani et al., 2017.
        refer https://arxiv.org/abs/1706.03762
    """

    def __init__(self, d_model, n_heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "`d_model` should be a multiple of `n_heads`"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = self.d_v = d_model // n_heads  # head_dim
        self.dropout_rate = dropout_rate

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(np.sqrt(self.d_k), dropout_rate)
    

    def split_heads(self, x):
        """ x: (batch_size, seq_len, d_model)
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (batch_size, n_heads, seq_len, d_k)

        # x: (batch_size, n_heads, seq_len, head_dim)
        return x


    def group_heads(self, x):
        """ x: (batch_size, n_heads, seq_len, head_dim)
        """
        batch_size = x.size(0)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)

        # x: (batch_size, seq_len, d_model)
        return x


    def forward(self, query, key, value, mask=None):
        """ query: (batch_size, query_len, d_model)
            key: (batch_size, key_len, d_model)
            value: (batch_size, value_len, d_model)
            mask: (batch_size, 1, source_seq_len) for source mask
                  (batch_size, target_seq_len, target_seq_len) for target mask
        """
        # apply linear projections to query, key and value
        Q = self.split_heads(self.W_q(query))  # (batch_size, n_heads, query_len, head_dim)
        K = self.split_heads(self.W_k(key))  # (batch_size, n_heads, key_len, head_dim)
        V = self.split_heads(self.W_v(value))  # (batch_size, n_heads, value_len, head_dim)

        if mask is not None:
            # apply same mask for all the heads
            mask = mask.unsqueeze(1)

            # mask: (batch_size, 1, 1, source_seq_len) for source mask
            #       (batch_size, 1, target_seq_len, target_seq_len) for target mask
        
        # calculate attention weights and context vector for each of the heads
        x, attn = self.attention(Q, K, V, mask)

        # x: (batch_size, n_heads, query_len, head_dim)
        # attn: (batch_size, n_heads, query_len, value_len)

        # concatenate context vector of all the heads
        x = self.group_heads(x)  # (batch_size, query_len, d_model)

        # apply linear projection to concatenated context vector
        x = self.W_o(x)  # (batch_size, query_len, d_model)

        # x: (batch_size, query_len, d_model)
        # attn: (batch_size, n_heads, query_len, value_len)
        return x, attn


class DecoderLayer(nn.Module):
    """ Decoder is made up of a self-attention layer, a encoder-decoder attention 
        layer and a feed-forward layer.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate

        self.attn_layer = MultiHeadAttention(d_model, n_heads, dropout_rate)
        self.attn_layer_norm = nn.BatchNorm1d(d_model, affine=True)

        self.enc_attn_layer = MultiHeadAttention(d_model, n_heads, dropout_rate)
        self.enc_attn_layer_norm = nn.BatchNorm1d(d_model, affine=True)

        self.ff_layer = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.ff_layer_norm = nn.BatchNorm1d(d_model, affine=True)

        self.dropout = nn.Dropout(dropout_rate)

    
    def forward(self, x, tgt_mask):
        """ x: (batch_size, target_seq_len, d_model)
            memory: (batch_size, source_seq_len, d_model)
            tgt_mask: (batch_size, target_seq_len, target_seq_len)
        """
        # apply self-attention
        x1, _ = self.attn_layer(x, x, x, tgt_mask)  # (batch_size, target_seq_len, d_model)

        # apply residual connection followed by layer normalization
        x = self.attn_layer_norm((x + self.dropout(x1)).reshape(-1, x.size(-1))).reshape(*x.size())  # (batch_size, target_seq_len, d_model)
        
        # apply position-wise feed-forward
        x1 = self.ff_layer(x)  # (batch_size, target_seq_len, d_model)

        # apply residual connection followed by layer normalization
        x = self.ff_layer_norm((x + self.dropout(x1)).reshape(-1, x.size(-1))).reshape(*x.size())  # (batch_size, target_seq_len, d_model)

        # x: (batch_size, target_seq_len, d_model)
        # attn: (batch_size, n_heads, target_seq_len, source_seq_len)
        return x




class Decoder(nn.Module):
    """ Decoder block is a stack of N identical decoder layers.
    """

    def __init__(self, d_model, n_layers, n_heads, d_ff, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout_rate)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.BatchNorm1d(d_model, affine=True)

    
    def forward(self, x, tgt_mask):
        """ x: (batch_size, target_seq_len, d_model)
            memory: (batch_size, source_seq_len, d_model)
            tgt_mask: (batch_size, target_seq_len, target_seq_len)
        """
        for layer in self.layers:
            x = layer(x, tgt_mask)  # (batch_size, target_seq_len, d_model)
        
        x = self.layer_norm(x.reshape(-1, x.size(-1))).reshape(*x.size())  # (batch_size, target_seq_len, d_model)

        # x: (batch_size, target_seq_len, d_model)
        # attn: (batch_size, n_heads, target_seq_len, source_seq_len)
        return x


class GNN1Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        emb_size = 64
        con_nfeats = 1
        cv_edge_nfeats = 1
        ov_edge_nfeats = 1
        var_nfeats = 3
        obj_nfeats = 6

        self.emb_size = emb_size

        # CONSTRAINT EMBEDDING
        self.con_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(con_nfeats),
            torch.nn.Linear(con_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.cv_edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cv_edge_nfeats),
        )

        self.ov_edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(ov_edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # OBJECTIVE EMBEDDING
        self.obj_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(obj_nfeats),
            torch.nn.Linear(obj_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()
        self.conv_v_to_o = BipartiteGraphConvolution()
        self.conv_o_to_v = BipartiteGraphConvolution()


        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=True),
        )

        self.sig = torch.nn.Sigmoid()

    def forward(
        self, con_features, var_features, obj_features, cv_edge_indices, cv_edge_features, ov_edge_indices, ov_edge_features, num_graphs, var_batch
    ):
        #con_features[con_features>1000000] = 10000
        #con_features = con_features/10000
        #obj_features[obj_features>1000000] = 10000
        #obj_features = obj_features/10000
        #var_features[:,0] = var_features[:,0]/100
        #cv_edge_features = cv_edge_features/100
        #ov_edge_features = ov_edge_features/100

        reversed_cv_edge_indices = torch.stack([cv_edge_indices[1], cv_edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        con_features = self.con_embedding(con_features)
        #cv_edge_features = self.cv_edge_embedding(cv_edge_features)
        var_features = self.var_embedding(var_features)
        obj_features = self.obj_embedding(obj_features)


        # Two half convolutions for cv
        con_features = self.conv_v_to_c(
            var_features, reversed_cv_edge_indices, cv_edge_features, con_features
        )
        var_features = self.conv_c_to_v(
            con_features, cv_edge_indices, cv_edge_features, var_features
        )

        reversed_ov_edge_indices = torch.stack([ov_edge_indices[1], ov_edge_indices[0]], dim=0)
        # Two half convolutions for ov
        obj_features = self.conv_v_to_o(
            var_features, reversed_ov_edge_indices, ov_edge_features, obj_features
        )
        var_features = self.conv_o_to_v(
            obj_features, ov_edge_indices, ov_edge_features, var_features
        )

        # A final MLP on the variable features
        output = var_features.reshape(num_graphs, -1, self.emb_size).mean(1)
        output = self.sig(self.output_module(output))
        return output


class GNN2Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 19

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Two half convolutions
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)
        return output




class GIN1Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        emb_size = 128
        con_nfeats = 1
        cv_edge_nfeats = 1
        ov_edge_nfeats = 1
        var_nfeats = 3
        obj_nfeats = 7

        N_LAYERS = 2
        N_HEADS = 4
        DROPOUT_RATE = 0.1

        self.emb_size = emb_size

        # CONSTRAINT EMBEDDING
        self.con_embedding = torch.nn.Sequential(
            #torch.nn.LayerNorm(con_nfeats),
            torch.nn.Linear(con_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.cv_edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cv_edge_nfeats),
        )

        self.ov_edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(ov_edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            #torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # OBJECTIVE EMBEDDING
        self.obj_embedding = torch.nn.Sequential(
            #torch.nn.LayerNorm(obj_nfeats),
            torch.nn.Linear(obj_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGIN()
        self.conv_c_to_v = BipartiteGIN()
        self.conv_v_to_o = BipartiteGIN()
        self.conv_o_to_v = BipartiteGIN()

        self.lin1 = Linear(emb_size*3, emb_size)
        self.lin2 = Linear(emb_size, 1)

        #self.lin1 = torch.nn.Sequential(
        #    torch.nn.LayerNorm(emb_size*3),
        #    torch.nn.Linear(emb_size*3, emb_size),
        #)

        #self.lin2 = torch.nn.Sequential(
        #    torch.nn.LayerNorm(emb_size),
        #    torch.nn.Linear(emb_size, 1),
        #)


        self.sig = torch.nn.Sigmoid()


        # for gt1
        gate1 = torch.nn.Sequential(
            #torch.nn.LayerNorm(emb_size),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1),
            torch.nn.ReLU(),
        )
        gate2 = torch.nn.Sequential(
            #torch.nn.LayerNorm(emb_size),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1),
            torch.nn.ReLU(),
        )
        gate3 = torch.nn.Sequential(
            #torch.nn.LayerNorm(emb_size),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1),
            torch.nn.ReLU(),
        )

        self.gt1 = GlobalAttention(gate1)
        self.gt2 = GlobalAttention(gate2)
        self.gt3 = GlobalAttention(gate3)



    def get_subsequent_mask(self, x):
        """ x: (batch_size, seq_len)
        """
        seq_len = x.size(1)
        subsequent_mask = np.triu(np.ones((1, seq_len, seq_len)), k=1).astype(np.int8)  # (batch_size, seq_len, seq_len)
        subsequent_mask = (torch.from_numpy(subsequent_mask) == 0).to(x.device)  # (batch_size, seq_len, seq_len)


        # subsequent_mask: (batch_size, seq_len, seq_len)
        return subsequent_mask


    def forward(
        self, con_features, var_features, obj_features, cv_edge_indices, cv_edge_features, ov_edge_indices, ov_edge_features, num_graphs, var_batch, con_batch, obj_batch
    ):

        reversed_cv_edge_indices = torch.stack([cv_edge_indices[1], cv_edge_indices[0]], dim=0)
        reversed_ov_edge_indices = torch.stack([ov_edge_indices[1], ov_edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        con_features = self.con_embedding(con_features)
        var_features = self.var_embedding(var_features)
        obj_features = self.obj_embedding(obj_features)

        # Two half convolutions for cv
        con_features = self.conv_v_to_c(
            var_features, reversed_cv_edge_indices, cv_edge_features, con_features
        )
        var_features = self.conv_c_to_v(
            con_features, cv_edge_indices, cv_edge_features, var_features
        )


        st_bg = self.gt1(torch.cat([var_features, con_features], dim=0), batch=torch.cat([var_batch, con_batch], dim=0))


        # Two half convolutions for ov
        obj_features = self.conv_v_to_o(
            var_features, reversed_ov_edge_indices, ov_edge_features, obj_features
        )
        var_features = self.conv_o_to_v(
            obj_features, ov_edge_indices, ov_edge_features, var_features
        )

        nd_bg = self.gt2(torch.cat([var_features, obj_features], dim=0), batch=torch.cat([var_batch, obj_batch], dim=0))
        rd_bg = self.gt3(torch.cat([obj_features, con_features], dim=0), batch=torch.cat([obj_batch, con_batch], dim=0))

        #xv = global_mean_pool(var_features, var_batch)
        #xc = global_mean_pool(con_features, con_batch)
        #xo = global_mean_pool(obj_features, obj_batch)
        #x = torch.cat([xv,xc,xo],1)
        x = torch.cat([st_bg,nd_bg,rd_bg],1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.sig(self.lin2(x))
        return x

