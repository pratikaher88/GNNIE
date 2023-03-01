# -*- coding: utf-8 -*-
"""deepwalk.ipynb
Author: Heidi L.

Original file is located at
    https://colab.research.google.com/drive/1GU-ZkZST63m5gPlYhN5CY5Fah3uGjIME

# Import

# Network Embeddings
Network embedding, i.e., network representation learning (NRL), is proposed to **embed network into a low dimensional space while preserving the network structure and property** so that the learned embeddings can be applied to the downstream network tasks.
* random walk
* deep neural network
* matrix factorization


All these algorithms are proposed for the **homogeneous graphs**.

### Deep Random Walk

Transfer graph into vectors. The representation vector carries structural information about nodes and its neighbor.

Steps:
1. Generate a random walk (a list of nodes walked) for each node
2. To make it a feature (for each node),
  - Use the classic walk (an 1D array)
  - Make a revision of the walk, e.g. user_only_walk, movie_only_walk (a shorter 1D array)
  - Calculate a score, e.g. cnt of different movies reached (a single value)


DONE:
1. The method is originally applied on homogeneous graphs, therefore it needs to be convert into a heterogeneous graph version.
2. Make different features: user_only_walk, movie_only_walk, scores.
3. Implement in `dgl` graphs.

Backlog:
1. Allow more walks per node (not implemented)
2. Tansform the walk into embeddings using Skip-gram model (Word2Vec) or other methods


Source:

[DeepWalk: Online Learning of Social Representations](https://arxiv.org/pdf/1403.6652.pdf)

[DeepWalk Implementation](https://towardsdatascience.com/exploring-graph-embeddings-deepwalk-and-node2vec-ee12c4c0d26d)
"""

import dgl
import torch
import random
import numpy as np
from typing import List
import copy
from gensim.models import Word2Vec

# from tqdm import tqdm


class DeepWalk:
    def __init__(self, walk_length: int, walks_per_node: int):
        """
        :param walk_length: length of the walk
        :param walks_per_node: number of walks per node
        """
        self.walk_length = walk_length
        self.walk_per_node = walks_per_node

    def random_walk(self, g: dgl.DGLGraph, node_edge_pairs: List) -> dict:
        """
        Generate a random walk for every node in the graph.
        :param g: Graph
        :param node_edge_pairs: [node type, 1st level edge type, 2nd level edge type]
        :return:
        """
        nodes_walk = {}
        for pair in node_edge_pairs:
            ntype = pair[0]
            nodes_walk[ntype] = []
            for start in g.nodes[ntype].data["node_ID"].tolist():
                walk = [start]
                for i in range(self.walk_length):
                    current = walk[i]
                    if i % 2 == 0:
                        neighbors = g.successors(
                            current, etype=pair[1]
                        ).tolist()
                    else:
                        neighbors = g.successors(
                            current, etype=pair[2]
                        ).tolist()
                    next = random.choice(
                        neighbors
                    )  # random sampling (equal probabilities)
                    walk.append(next)  # walk to the next node
                nodes_walk[ntype].append(walk)
        self.walks = nodes_walk

    def get_feature_walk(self, node_type_restriction=False):
        """
        Return the walks.
        :param node_type_restriction: Default `False`. If `True`, return walks that contains only the type of the start node.
        :return:
        """
        if not node_type_restriction:
            return self.walks
        else:
            walks = copy.deepcopy(self.walks)
            ntypes = walks.keys()
            for ntype in ntypes:
                for w in walks[ntype]:
                    # Remove nodes with odd indices. In the walk list, even indices have the same node type as start node.
                    del w[1::2]
            return walks

    def get_feature_diversity_score(self):
        """
        Return a score of (cnts of different nodes in the walk)/(walk length).
        """
        walks = self.get_feature_walk(node_type_restriction=True)
        score = {}
        ntypes = walks.keys()
        for ntype in ntypes:
            score[ntype] = []
            for w in walks[ntype]:
                node = w[0]
                distinct_node_cnt = len(set(w))
                if distinct_node_cnt > 0:
                    s = distinct_node_cnt / len(w)
                else:
                    s = 0
                score[ntype].append([node, s])
        return score


### driver ###

# initiate
deepwalk = DeepWalk(5, 1)
deepwalk.random_walk(
    movie_hetero_graph_clean,
    [['user', 'rates', 'rev-rates'], ['movie', 'rev-rates', 'rates']],
)

# get features
deep_walk = deepwalk.get_feature_walk()
same_type_walk = deepwalk.get_feature_walk(node_type_restriction=True)
diversity_score = deepwalk.get_feature_diversity_score()

print(deep_walk)
print('---------------')
print(same_type_walk)
print('---------------')
print(diversity_score)

# Add features to graph nodes
for ntype in movie_hetero_graph_clean.ntypes:
    # where should features be located at, .data['features'] or create another .data['feature_new']?
    movie_hetero_graph_clean.nodes[ntype].data['features'] = torch.stack(
        (
            movie_hetero_graph_clean.nodes[ntype].data['features'],
            torch.FloatTensor([[s[1]] for s in diversity_score[ntype]]),
        ),
        dim=1,
    ).squeeze()
    movie_hetero_graph_clean.nodes[ntype].data[
        'deepwalk_diversity_score'
    ] = torch.FloatTensor([[s[1]] for s in diversity_score[ntype]])
    movie_hetero_graph_clean.nodes[ntype].data[
        'deepwalk_sametype'
    ] = torch.Tensor([w for w in same_type_walk[ntype]])
    movie_hetero_graph_clean.nodes[ntype].data['deepwalk'] = torch.Tensor(
        [w for w in deep_walk[ntype]]
    )

# sanity check
print(movie_hetero_graph_clean.nodes['user'])
print('=================')
print(movie_hetero_graph_clean.nodes['movie'])
