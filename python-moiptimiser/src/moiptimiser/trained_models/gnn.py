"""
GNN models.

File adapted from https://github.com/ds4dm/ecole
by Lars Sandberg @Sandbergo
May 2021
"""

import torch

import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import global_mean_pool, MessagePassing
from torch_geometric.utils import remove_self_loops

from utilities.model import BipartiteGraphConvolution, BipartiteGIN


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
        emb_size = 64
        con_nfeats = 1
        cv_edge_nfeats = 1
        ov_edge_nfeats = 1
        var_nfeats = 3
        obj_nfeats = 6

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

        self.lin1 = Linear(emb_size, emb_size)
        self.lin2 = Linear(emb_size, 1)
        self.sig = torch.nn.Sigmoid()

    def forward(
        self, con_features, var_features, obj_features, cv_edge_indices, cv_edge_features, ov_edge_indices, ov_edge_features, num_graphs, var_batch
    ):

        reversed_cv_edge_indices = torch.stack([cv_edge_indices[1], cv_edge_indices[0]], dim=0)

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

        reversed_ov_edge_indices = torch.stack([ov_edge_indices[1], ov_edge_indices[0]], dim=0)
        # Two half convolutions for ov
        obj_features = self.conv_v_to_o(
            var_features, reversed_ov_edge_indices, ov_edge_features, obj_features
        )
        var_features = self.conv_o_to_v(
            obj_features, ov_edge_indices, ov_edge_features, var_features
        )

        x = global_mean_pool(var_features, var_batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.sig(self.lin2(x))
        return x

