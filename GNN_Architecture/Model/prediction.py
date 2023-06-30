import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class CosinePrediction(nn.Module):
    """
    Scoring function that uses cosine similarity to compute similarity between user and item.
    Only used if fixed_params.pred == 'cos'.
    """

    def __init__(self):
        super().__init__()

    def forward(self, graph, h, pos_graph = False):
        # print("input graph :", graph)
        with graph.local_scope():
            for etype in graph.canonical_etypes:
                try:
                    graph.nodes[etype[0]].data['norm_h'] = F.normalize(h[etype[0]], p=2, dim=-1)
                    graph.nodes[etype[2]].data['norm_h'] = F.normalize(h[etype[2]], p=2, dim=-1)
                    graph.apply_edges(fn.u_dot_v('norm_h', 'norm_h', 'cos'), etype=etype)
                except KeyError:
                    pass  # For etypes that are not in training eids, thus have no 'h'
            ratings = graph.edata['cos']

        # print("ratings", ratings)
        return ratings

class CosinePredictionWihEdge(nn.Module):

    def __init__(self, dim_dict, embed_dim, orders_edge_dim, rev_orders_edge_dim):
        super().__init__()
        
        self.dim_dict = dim_dict
        self.orders_edge_dim = orders_edge_dim
        self.hidden_1 = nn.Linear(embed_dim * 2 + orders_edge_dim, 128)
        self.hidden_2 = nn.Linear(128, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, graph, h, pos_graph):
        # print("input graph :", graph)

        result = {}
        # print("Graph edata shape", graph.edata['features'][('customer', 'orders', 'product')].shape)

        edge_features = graph.edata['features']
        edge_features_HM = {}
        for key, value in edge_features.items():
            # print(key, value[0].shape)
            edge_features_HM[key[1]] = value

        # print('-------------')

        # print()

        with graph.local_scope():
            for etype in graph.canonical_etypes:
                # try:
                    # graph.nodes[etype[0]].data['norm_h'] = F.normalize(h[etype[0]], p=2, dim=-1)
                    # graph.nodes[etype[2]].data['norm_h'] = F.normalize(h[etype[2]], p=2, dim=-1)
                    # graph.apply_edges(fn.u_dot_v('norm_h', 'norm_h', 'cos'), etype=etype)

                utype, mtype, vtype = etype
                src_nid, dst_nid = graph.all_edges(etype=etype)
                emb_heads = h[utype][src_nid]
                emb_tails = h[vtype][dst_nid]

                # edge_emb = graph.edata['features'][('customer', 'orders', 'product')].shape
                # print(h[utype].shape)

                if not pos_graph:
                    edge_emb = torch.randn(emb_heads.shape[0], self.orders_edge_dim)
                else:
                    edge_emb = edge_features_HM[mtype]

                cat_embed = torch.cat((emb_heads, emb_tails, edge_emb), 1)

                # print("embedding shape", emb_heads.shape, pos_graph, cat_embed.shape, edge_emb.shape)
                x = self.hidden_1(cat_embed)
                x = self.relu(x)
                x = self.hidden_2(x)
                x = self.relu(x)
                x = self.output(x)
                x = self.sigmoid(x)
                
                # print(x.shape)
                result[etype] = x

                # x = self.hidden_1(cat_embed)
                # x = self.relu(x)
                # x = self.hidden_2(x)
                # x = self.relu(x)
                # x = self.output(x)
                # x = self.sigmoid(x)
                # return x

                # A =  F.normalize(h[etype[0]], p=2, dim=-1)
                # B =  F.normalize(h[etype[2]], p=2, dim=-1)
                
                # C = edge_features[etype[1]][0]

                # print("Edge values", A.shape, B.shape, C.shape)

                
                # except KeyError:
                #     pass  # For etypes that are not in training eids, thus have no 'h'
            # ratings = graph.edata['cos']
        return result

class PredictingLayer(nn.Module):
    """
    Scoring function that uses a neural network to compute similarity between user and item.
    Only used if fixed_params.pred == 'nn'.
    Given the concatenated hidden states of heads and tails vectors, passes them through neural network and
    returns sigmoid ratings.
    """

    def reset_parameters(self):
        gain_relu = nn.init.calculate_gain('relu')
        gain_sigmoid = nn.init.calculate_gain('sigmoid')
        nn.init.xavier_uniform_(self.hidden_1.weight, gain=gain_relu)
        nn.init.xavier_uniform_(self.hidden_2.weight, gain=gain_relu)
        nn.init.xavier_uniform_(self.output.weight, gain=gain_sigmoid)

    def __init__(self, embed_dim: int):
        super(PredictingLayer, self).__init__()
        self.hidden_1 = nn.Linear(embed_dim * 2, 128)
        self.hidden_2 = nn.Linear(128, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def forward(self, x):
        x = self.hidden_1(x)
        x = self.relu(x)
        x = self.hidden_2(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


class PredictingModule(nn.Module):
    """
    Predicting module that incorporate the predicting layer defined earlier.
    Only used if fixed_params.pred == 'nn'.
    Process:
        - Fetches hidden states of 'heads' and 'tails' of the edges.
        - Concatenates them, then passes them through the predicting layer.
        - Returns ratings (sigmoid function).
    """

    def __init__(self,embed_dim: int):
        super(PredictingModule, self).__init__()
        self.layer_nn = PredictingLayer(embed_dim)

    def forward(self,
                graph,
                h
                ):
        ratings_dict = {}
        for etype in graph.canonical_etypes:
            # print(etype)
            if etype[0] in ['customer', 'product'] and etype[2] in ['customer', 'product']:
                utype, _, vtype = etype
                src_nid, dst_nid = graph.all_edges(etype=etype)
                emb_heads = h[utype][src_nid]
                emb_tails = h[vtype][dst_nid]
                cat_embed = torch.cat((emb_heads, emb_tails), 1)
                ratings = self.layer_nn(cat_embed)
                ratings_dict[etype] = torch.flatten(ratings)
        ratings_dict = {key: torch.unsqueeze(ratings_dict[key], 1) for key in ratings_dict.keys()}
        return ratings_dict


class Cosine_PredictingLayer(nn.Module):
    """
    Scoring function that uses a neural network in tandem with cosine prediction
    to compute similarity between user and item.
    Only used if fixed_params.pred == 'cos_nn'.
    Given the concatenated hidden states of heads vectors, tails vectors, and the cosine similarity between  them,
    passes them through neural network and returns sigmoid ratings.
    """

    def reset_parameters(self):
        gain_relu = nn.init.calculate_gain('relu')
        gain_sigmoid = nn.init.calculate_gain('sigmoid')
        nn.init.xavier_uniform_(self.hidden_1.weight, gain=gain_relu)
        nn.init.xavier_uniform_(self.hidden_2.weight, gain=gain_relu)
        nn.init.xavier_uniform_(self.output.weight, gain=gain_sigmoid)

    def __init__(self, embed_dim: int):
        super(Cosine_PredictingLayer, self).__init__()
        self.hidden_1 = nn.Linear((embed_dim * 2)+1, 128)
        self.hidden_2 = nn.Linear(128, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def forward(self, x):
        x = self.hidden_1(x)
        x = self.relu(x)
        x = self.hidden_2(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


class Cosine_PredictingModule(nn.Module):
    """
    Predicting module that incorporate the predicting layer defined earlier.
    Only used if fixed_params.pred == 'nn'.
    Process:
        - Fetches hidden states of 'heads' and 'tails' of the edges.
        - Concatenates them, then passes them through the predicting layer.
        - Returns ratings (sigmoid function).
    """

    def __init__(self, embed_dim: int):
        super(Cosine_PredictingModule, self).__init__()
        self.layer_nn = Cosine_PredictingLayer(embed_dim)

    def forward(self,
                graph,
                h
                ):
        ratings_dict = {}
        for etype in graph.canonical_etypes:
            if etype[0] in ['customer', 'product'] and etype[2] in ['customer', 'product']:
                utype, _, vtype = etype
                src_nid, dst_nid = graph.all_edges(etype=etype)
                emb_heads = h[utype][src_nid]
                emb_tails = h[vtype][dst_nid]
                emb_heads_norm = F.normalize(emb_heads, p=2, dim=-1)
                emb_tails_norm = F.normalize(emb_tails, p=2, dim=-1)
                emb_cos_sim = nn.functional.cosine_similarity(emb_heads_norm, emb_tails_norm)
                cat_embed_1 = torch.cat((emb_heads, emb_tails), 1)
                emb_cos_sim = emb_cos_sim.unsqueeze(1)
                cat_embed = torch.cat((cat_embed_1, emb_cos_sim), 1)
                ratings = self.layer_nn(cat_embed)
                ratings_dict[etype] = torch.flatten(ratings)
        ratings_dict = {key: torch.unsqueeze(ratings_dict[key], 1) for key in ratings_dict.keys()}
        return ratings_dict