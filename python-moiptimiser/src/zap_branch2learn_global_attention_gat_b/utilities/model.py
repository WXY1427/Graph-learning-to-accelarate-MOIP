"""
Model utilities.

File adapted from https://github.com/ds4dm/ecole
by Lars Sandberg @Sandbergo
May 2021
"""

import torch
import torch.nn.functional as F
import torch_geometric
import math 
from torch.nn import Parameter, Sigmoid, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import global_mean_pool, MessagePassing
from torch_geometric.utils import remove_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.dense.linear import Linear


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

class BipartiteGIN(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    def __init__(self, eps=0, train_eps=False):
        super().__init__("add")
        emb_size = 32

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )
        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))
        # output_layers
        self.nn = Sequential(
            Linear(emb_size, emb_size),
            ReLU(),
            Linear(emb_size, emb_size),
            ReLU(),
            BN(emb_size),
        )

        self.nn_n = Sequential(
            Linear(4*emb_size, emb_size),
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )


        self.initial_eps = eps
        if train_eps:
            self.eps = Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))


        self.in_channels = emb_size
        self.out_channels = emb_size
        self.heads = 4
        self.concat = True
        self.negative_slope = 0.2
        self.dropout = 0.0
        self.add_self_loops = False
        self.edge_dim = emb_size
        self.fill_value = 'mean'
        self.share_weights = False

        self.lin_l = Linear(emb_size, self.heads * self.out_channels, bias=True,
                            weight_initializer='glorot')
        self.lin_r = Linear(emb_size, self.heads * self.out_channels, bias=True,
                            weight_initializer='glorot')
        self.att = Parameter(torch.Tensor(1, self.heads, self.out_channels))
        self.lin_edge = Linear(1, self.heads * self.out_channels, bias=False,
                                   weight_initializer='glorot')
        self.bias = Parameter(torch.Tensor(self.heads * self.out_channels))
        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        self.lin_edge.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(self, left_features, edge_index, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        #H, C = self.heads, self.out_channels
        #x_l, x_r = left_features, right_features
        #assert x_l.dim() == 2
        #x_l = self.lin_l(x_l).view(-1, H, C)
        #x_r = self.lin_r(x_r).view(-1, H, C)

        #output = self.nn((1 + self.eps) * right_features + self.propagate(
        #    edge_index,
        #    size=(left_features.shape[0], right_features.shape[0]),
        #    node_features=(left_features, right_features),
        #    edge_features=edge_features,
        #))

        output = self.propagate(
            edge_index,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )

        alpha = self._alpha
        self._alpha = None
        #output = output.view(-1, self.heads * self.out_channels)
        output += self.bias

        output = self.nn((1 + self.eps) * right_features + self.nn_n(output))

        
        return output

    def message(self, node_features_i, node_features_j, edge_features, index, size_i):
        #output = self.feature_module_final(
        #    self.feature_module_left(node_features_i)
        #    + self.feature_module_edge(edge_features)
        #    + self.feature_module_right(node_features_j)
        #)
        node_features_i = self.lin_l(node_features_i).view(-1, self.heads, self.out_channels)
        node_features_j = self.lin_r(node_features_j).view(-1, self.heads, self.out_channels)
        x = node_features_i + node_features_j
        edge_attr = self.lin_edge(edge_features)
        edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
        x += edge_attr
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, None, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        output = node_features_j * alpha.unsqueeze(-1)
        return output.reshape(-1, self.heads*self.out_channels)


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    def __init__(self):
        super().__init__("add")
        emb_size = 64

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )
        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))
        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )


    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """

        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )

        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )
        return output


def process(policy, data_loader, device: str, optimizer=None):
    """
    This function will process a whole epoch of training or validation,
    depending on whether an optimizer is provided.
    """
    mean_loss = 0
    mean_acc = 0

    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in data_loader:
            batch = batch.to(device)
            # Compute the logits (i.e. pre-softmax activations)
            # according to the policy on the concatenated graphs
            logits = policy(
                batch.x_c,
                batch.x_v,
                batch.x_o,
                batch.edge_index_c,
                batch.cv_edge_features,
                batch.edge_index_o,
                batch.ov_edge_features,
                batch.num_graphs,
                batch.x_v_batch,
                batch.x_c_batch,
                batch.x_o_batch,
            )

            # logits: batch_size * 2
            #loss = F.cross_entropy(logits, batch.label)
            loss = F.binary_cross_entropy(logits, batch.label.unsqueeze(1).float())


            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates)
            #true_bestscore = true_scores.max(dim=-1, keepdims=True).values

            #predicted_bestindex = logits.max(dim=-1, keepdims=True).indices
            predicted_bestindex = (logits>0.5).long()
            accuracy = (
                (predicted_bestindex == batch.label.unsqueeze(1))
                .float()
                .mean()
                .item()
            )

            mean_loss += loss.item() * batch.num_graphs
            mean_acc += accuracy * batch.num_graphs
            n_samples_processed += batch.num_graphs

    mean_loss /= n_samples_processed
    mean_acc /= n_samples_processed
    return mean_loss, mean_acc


def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    """
    This utility function splits a tensor and pads each split to make them
    all the same size, then stacks them.
    """
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    output = torch.stack(
        [
            F.pad(slice_, (0, max_pad_size - slice_.size(0)), "constant", pad_value)
            for slice_ in output
        ],
        dim=0,
    )
    return output
