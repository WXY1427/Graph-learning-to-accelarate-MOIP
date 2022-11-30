"""
Data utilities.

File adapted from https://github.com/ds4dm/ecole
by Lars Sandberg @Sandbergo
May 2021
"""

import gzip
import pickle

import numpy as np
import torch
import torch_geometric

def preprocess_variable_features(features, interaction_augmentation, normalization):
    """
    Features preprocessing following Khalil et al. (2016) Learning to Branch in Mixed Integer Programming.
    Parameters
    ----------
    features : 2D np.ndarray
        The candidate variable features to preprocess.
    interaction_augmentation : bool
        Whether to augment features with 2-degree interactions (useful for linear models such as SVMs).
    normalization : bool
        Wether to normalize features in [0, 1] (i.e., query-based normalization).
    Returns
    -------
    variable_features : 2D np.ndarray
        The preprocessed variable features.
    """
    # 2-degree polynomial feature augmentation
    if interaction_augmentation:
        interactions = (
            np.expand_dims(features, axis=-1) * \
            np.expand_dims(features, axis=-2)
        ).reshape((features.shape[0], -1))
        features = np.concatenate([features, interactions], axis=1)

    # query-based normalization in [0, 1]
    if normalization:
        features -= features.min(axis=1, keepdims=True)
        max_val = features.max(axis=1, keepdims=True)
        max_val[max_val == 0] = 1
        features /= max_val

    return features


class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned
    by the `ecole.observation.NodeBipartite` observation function
    in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
        self,
        x_c,
        x_v,
        x_o,
        edge_index_c,
        cv_edge_features,
        edge_index_o,
        ov_edge_features,
        label,
    ):
        super().__init__()
        self.x_c = x_c
        self.x_v = x_v
        self.x_o = x_o
        self.edge_index_c = edge_index_c
        self.cv_edge_features = cv_edge_features
        self.edge_index_o = edge_index_o
        self.ov_edge_features = ov_edge_features
        self.label = label

         # add *args, **kwargs
    def __inc__(self, key, value, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices
        when concatenating graphs for those entries (edge index, candidates)
        for which this is not obvious.
        """
        if key == 'edge_index_c':
            return torch.tensor(
                [[self.x_c.size(0)], [self.x_v.size(0)]]
            )

        if key == 'edge_index_o':
            return torch.tensor(
                [[self.x_o.size(0)], [self.x_v.size(0)]]
            )

        else:
            return super().__inc__(key, value, *args, **kwargs)


class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load
    such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk
        during data collection.
        """
        with gzip.open(self.sample_files[index], "rb") as in_file:
            sample = pickle.load(in_file)

        con_mat, obj_mat, var_mat, label, kk, _ = sample 
        con_features = con_mat[:,-1:]
        var_features = var_mat
        obj_features = np.concatenate((obj_mat[:,-6:], np.eye(len(obj_mat))[:,kk:kk+1]), axis=1)  #obj_mat[:,-6:]
        cv_edge_features = con_mat[:,:-1]
        cv_edge_indices = np.vstack(np.nonzero(cv_edge_features))
        cv_edge_features = cv_edge_features.reshape(-1, 1)
        ov_edge_features = obj_mat[:,:-6]
        ov_edge_indices = np.vstack(np.nonzero(ov_edge_features))

        ov_edge_features = ov_edge_features.reshape(-1, 1)


        con_features = torch.from_numpy(con_features.astype(np.float32))
        var_features = torch.from_numpy(var_features.astype(np.float32))
        obj_features = torch.from_numpy(obj_features.astype(np.float32))
        cv_edge_features = torch.from_numpy(cv_edge_features.astype(np.float32))
        cv_edge_indices = torch.from_numpy(cv_edge_indices.astype(np.int64))
        ov_edge_features = torch.from_numpy(ov_edge_features.astype(np.float32))
        ov_edge_indices = torch.from_numpy(ov_edge_indices.astype(np.int64))

        # We note on which variables we were allowed to branch, the scores as well as the choice
        # taken by strong branching (relative to the candidates)
        #candidates = torch.LongTensor(np.array(sample_action_set, dtype=np.int32))
        #candidate_scores = torch.FloatTensor([sample_scores[j] for j in candidates])
        label = torch.LongTensor([label])

        con_features[con_features>100000] = -10000
        con_features = con_features/10000
        obj_features[obj_features>100000] = -10000
        obj_features = obj_features/10000
        var_features[:,0] = var_features[:,0]/100
        cv_edge_features = cv_edge_features/100
        ov_edge_features = ov_edge_features/100


        graph = BipartiteNodeData(
            con_features,
            var_features,
            obj_features,
            cv_edge_indices,
            cv_edge_features,
            ov_edge_indices,
            ov_edge_features,
            label,
        )

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = con_features.shape[0] + var_features.shape[0] + obj_features.shape[0] 

        return graph
