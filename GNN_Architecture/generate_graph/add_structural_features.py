
import dgl, random, copy, torch, os
import numpy as np
from typing import List
from torch import nn
# from settings import BASE_DIR
# from gensim.models import Word2Vec

import argparse
parser = argparse.ArgumentParser(description='Parse arguments for adding structural features to the graph')
# parser.add_argument("--remove_duplicates", default=False, help="Remove duplicate product id and customer id combination", type=bool)
parser.add_argument('--graph_name', help="takes in the name of the graph", type=str, default="ecommerce_hetero_graph")

args = parser.parse_args()

graph_name = str(args.graph_name)


BASE_DIR = os.getcwd()

print("BASE_DIR: ", BASE_DIR)
graphs, _ = dgl.load_graphs(f"{BASE_DIR}/run_data/graph_files_subgraph/{graph_name}.dgl")
ecommerce_hetero_graph = graphs[0]


class DeepWalk:
    def __init__(self, g: dgl.DGLGraph, node_edge_pairs: List, walk_length: int, walks_per_node: int):
        """
        :param walk_length: length of the walk
        :param walks_per_node: number of walks per node
        """
        self.walk_length = walk_length
        self.walk_per_node = walks_per_node # backlog
        self.num_of_nodes = g.number_of_nodes()
        self.num_of_nodes_ntype = []
        for pair in node_edge_pairs:
            ntype = pair[0]
            self.num_of_nodes_ntype.append([ntype, len(g.nodes(ntype))])
        self.walks = self._forward(g, node_edge_pairs)

    def _forward(self, g: dgl.DGLGraph, node_edge_pairs: List) -> dict:
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
            for start in g.nodes(ntype).tolist():
                walk = [start]
                for i in range(self.walk_length):
                    current = walk[i]
                    if i%2 == 0:
                        neighbors = g.successors(current, etype=pair[1]).tolist()
                    else:
                        neighbors = g.successors(current, etype=pair[2]).tolist()
                    next = random.choice(neighbors) # random sampling (equal probabilities)
                    walk.append(next) # walk to the next node
                nodes_walk[ntype].append(walk)
        return nodes_walk

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
                    s = distinct_node_cnt/len(w)
                else:
                    s = 0                
                score[ntype].append([node, s])
        return score

    def get_embedding(self, H):
        """
        Return the embeddings.
        :param H: The output dimension of embeddings. After projection, every node_id becomes an H-dim array.
        :return: Tensors in the shape of : [num of nodes, walk length, H]0
        """
        # create input tensor
        input_walks = []
        for _, k in enumerate(self.walks):
            input_walks.extend([w for w in self.walks[k]])
        input_tensor = torch.tensor(input_walks)

        # train the embedding
        embedding_layer = nn.Embedding(num_embeddings=self.num_of_nodes, embedding_dim=H) # need unique keys for this method
        _embedding = embedding_layer(input_tensor)

        # sort into different ntypes
        embedding_walk = {}
        start = 0
        for pair in self.num_of_nodes_ntype:
            ntype = pair[0]
            n = pair[1]
            embedding_walk[ntype] = _embedding[start:(start + n)].detach()
            start += n
        return embedding_walk

def indegree_feature(graph):
    output = {}
    for s_ntype in graph.ntypes:
        for d_ntype in graph.ntypes:
            if s_ntype!=d_ntype:
                # print("Calculating indegree for source ntype: ", s_ntype, "and dest ntype: ", d_ntype)
                indegree_sum = None
                for etype in graph.etypes:
                    try:
                        indegree = graph.in_degrees(etype=(s_ntype, etype, d_ntype))
                        if indegree_sum:
                            indegree_sum += indegree
                        else:
                            indegree_sum = indegree
                    except:
                        print(f'no edge type {etype} between source {s_ntype} and dest {d_ntype}')
                        
                indegree_tensor = torch.FloatTensor([[val] for val in indegree_sum])
                output[d_ntype] = indegree_tensor
    return output

def outdegree_feature(graph):
    output = {}
    for s_ntype in graph.ntypes:
        for d_ntype in graph.ntypes:
            if s_ntype!=d_ntype:
                # print("Calculating outdegree for source ntype: ", s_ntype, "and dest ntype: ", d_ntype)
                outdegree_sum = None
                for etype in graph.etypes:
                    try:
                        outdegree = graph.out_degrees(etype=(d_ntype, etype, s_ntype))
                        if outdegree_sum:
                            outdegree_sum += outdegree
                        else:
                            outdegree_sum = outdegree
                    except:
                        print(f'no edge type {etype} between source {d_ntype} and dest {s_ntype}')
                        
                outdegree_tensor = torch.FloatTensor([[val] for val in outdegree_sum])
                output[d_ntype] = outdegree_tensor
    return output

"""
    PageRank Helper Function
"""
def pagerank_reduce_func(nodes, DAMP=.85):
    msgs = torch.sum(nodes.mailbox['pagerank_pv'], dim=1)
    N = nodes.batch_size()

    pv = (1 - DAMP) / N + DAMP * msgs
    return {'pagerank_pv' : pv}

def pagerank_message_func(edges):
    return {'pagerank_pv' : edges.src['pagerank_pv'] / edges.src['pagerank_deg']}

"""
PageRank

Implements Pagerank features in bypartite GNN

Parameters
----------
g : DGL Heterograph 
    The Graph should contain two node types only.
user_label : string, optional
    Name of the user node
product_label: string, optional
    Name of the product node
edge_label: string, optional
    Name of the user to product edge type
rev_edge_label: string, optional
    Name of the product to user edge type
DAMP: float, optional
    Damp or decay factor. This corresponds to the probability of connections sinking at any giving point (nodes with no outgoing edges). 
    It prevents the sinked nodes from "absorbing" the PageRanks of those pages connected to the sinks. 
reverse: bool, optional
    Whether or not the PageRank algorithm should run on the reverse orientation (products to users)

Returns
    
-------
DGL Heterograph 
    The Graph with pagerank features included in its nodes ("pagerank_pv").
"""
def pagerank(g, user_label = 'user', product_label = 'product', edge_label = 'purchase', rev_edge_label = 'review', DAMP = 0.85, reverse = False):
   
    N = g.number_of_nodes()
    N_user = g.num_src_nodes(user_label)
    N_product = g.num_src_nodes(product_label)
    
    g.nodes[user_label].data['pagerank_pv'] = torch.ones(N_user) / N
    g.nodes[product_label].data['pagerank_pv'] = torch.ones(N_product) / N
    g.nodes[user_label].data['pagerank_deg'] = g.out_degrees(g.nodes(user_label), etype=edge_label).float()
    g.nodes[product_label].data['pagerank_deg'] = g.out_degrees(g.nodes(product_label), etype=rev_edge_label).float()

    g.multi_update_all({edge_label: (pagerank_message_func, pagerank_reduce_func)},"sum")
    
    if(reverse):
        g.multi_update_all({rev_edge_label: (pagerank_message_func, pagerank_reduce_func)},"sum")
 
    dict1 = {}
    dict1[user_label] = torch.unsqueeze(g.nodes[user_label].data['pagerank_pv'], 1) 
    dict1[product_label] = torch.unsqueeze(g.nodes[product_label].data['pagerank_pv'], 1)
    return dict1



def concat_feature_tensors(node_types, **kwargs):
    """
    Take in multiple feature tensors, check if its a valid tensor size, and concatenate them.
    Output: Dict with different ntype as keys, tensors as value.
    """
    out_feature_tensors = {}
    for ntype in node_types:
        for key, value in kwargs.items():
            tensors = value[ntype]        
    
            # sanity check: tensor size
            if tensors.dim() > 3:
                return "Error dimension in feature:{}".format(key)
            if tensors.dim() == 3:
                value[ntype] = tensors.flatten(1, 2)
            # print(key, ntype, value[ntype])
        
        out_feature_tensors[ntype] = torch.cat(tuple([v[ntype] for k, v in kwargs.items()]), 
                                               dim=-1)
    return out_feature_tensors

def add_features_to_graph(g: dgl.DGLGraph, feature_tensor_to_add: torch.tensor):
    """
    Append feature tensors to the nodes in graph.
    """
    # if there's exsiting features in the graph
    exist_features = {}
    for ntype in g.ntypes:
        if ("features" in g.nodes[ntype].data.keys()) and (g.nodes[ntype].data["features"] is not None):
            exist_features[ntype] = g.nodes[ntype].data["features"]
    
    # TODO: removed exisiting features for now
    feature_tensor_to_add = concat_feature_tensors(node_types=g.ntypes,
                                                    exist_feature=exist_features, 
                                                    feature_tensor_to_add=feature_tensor_to_add)
    # append features to the graph
    for ntype in g.ntypes:
        g.nodes[ntype].data["features"] = feature_tensor_to_add[ntype]    
    return g
    

deepwalk = DeepWalk(g=ecommerce_hetero_graph, 
                    node_edge_pairs=[['customer', 'orders', 'rev-orders'],['product', 'rev-orders', 'orders']],
                    walk_length=3, 
                    walks_per_node=1)
deep_walk = deepwalk.get_feature_walk()
same_type_walk = deepwalk.get_feature_walk(node_type_restriction=True)
diversity_score = deepwalk.get_feature_diversity_score()
embedding_walk = deepwalk.get_embedding(H=4)

# for key in deep_walk:
#     deep_walk[key] = torch.tensor(deep_walk[key]).detach()


pagerank_op = pagerank(ecommerce_hetero_graph, 
             user_label = 'customer', 
             product_label = 'product', 
             edge_label = 'orders', 
             rev_edge_label = 'rev-orders',
             reverse = False)

indegree = indegree_feature(ecommerce_hetero_graph)
outdegree = outdegree_feature(ecommerce_hetero_graph)

structural_feature_tensors = concat_feature_tensors(node_types=["customer","product"], 
                                          in_degree=indegree,
                                          out_degree=outdegree,
                                          pagerank=pagerank_op,
                                          walk_embeddings=embedding_walk,
                                          )

ecommerce_hetero_graph_clean = add_features_to_graph(ecommerce_hetero_graph, structural_feature_tensors)


print("Saving graph! at location :", f"{BASE_DIR}/run_data/graph_files_subgraph/{graph_name}_with_sf.dgl")
dgl.save_graphs(f"{BASE_DIR}/run_data/graph_files_subgraph/{graph_name}_with_sf.dgl", [ecommerce_hetero_graph_clean])

