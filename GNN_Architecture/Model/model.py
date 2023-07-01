import torch.nn as nn
import dgl.nn.pytorch as dglnn
from Model.layer import ConvLayer
from Model.prediction import CosinePrediction, PredictingModule, PredictingLayer, Cosine_PredictingModule, Cosine_PredictingLayer, CosinePredictionWihEdge

class NodeEmbedding(nn.Module):
    """
    Projects the node features into embedding space.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 ):
        super().__init__()
        self.proj_feats = nn.Linear(in_feats, out_feats)

        # self.proj_feats = nn.Sequential(
        #     nn.Linear(in_feats, out_feats),
        #     nn.BatchNorm1d(out_feats),
        #     nn.Tanh())

    def forward(self,
                node_feats):
        return self.proj_feats(node_feats)

class ConvModel(nn.Module):

    def __init__(self, g, n_layers, dim_dict, norm: bool = True, dropout: float = 0.0, aggregator_type: str = 'mean', pred: str = 'cos', aggregator_hetero: str = 'mean', embedding_layer: bool = True):
        
        super(ConvModel, self).__init__()

        self.user_embed = NodeEmbedding(dim_dict['customer'], dim_dict['hidden_dim'])
        self.item_embed = NodeEmbedding(dim_dict['product'], dim_dict['hidden_dim'])

        self.layers = nn.ModuleList()

        # hidden_dim layer
        for _ in range(n_layers - 2):
            self.layers.append(
                dglnn.HeteroGraphConv(
                    {etype[1]: ConvLayer((dim_dict['hidden_dim'], dim_dict['hidden_dim']), dim_dict['hidden_dim'], dim_dict[etype[1]]['edge_dim'], nn.Sequential(
                                                nn.Linear(dim_dict[etype[1]]['edge_dim'], dim_dict['edge_hidden_dim']),
                                                nn.ReLU(),
                                                nn.Linear(dim_dict['edge_hidden_dim'], dim_dict['hidden_dim']*dim_dict['hidden_dim'])
                                            ), dropout,
                                            aggregator_type, norm)
                        for etype in g.canonical_etypes},
                    aggregate=aggregator_hetero))
        
        # output layer

        self.layers.append(
            dglnn.HeteroGraphConv(
                {etype[1]: ConvLayer((dim_dict['hidden_dim'], dim_dict['hidden_dim']), dim_dict['out_dim'], dim_dict[etype[1]]['edge_dim'], nn.Sequential(
                                                nn.Linear(dim_dict[etype[1]]['edge_dim'], dim_dict['edge_hidden_dim']),
                                                nn.ReLU(),
                                                nn.Linear(dim_dict['edge_hidden_dim'], dim_dict['hidden_dim']*dim_dict['out_dim'])
                                            ), dropout,
                                     aggregator_type, norm)
                 for etype in g.canonical_etypes},
                aggregate=aggregator_hetero))
        
        # self.pred_fn = CosinePrediction()

        if pred == 'cos':
            self.pred_fn = CosinePrediction()
        elif pred == 'nn':
            self.pred_fn = PredictingModule(dim_dict['out_dim'])
        elif pred == 'cos_nn':
            self.pred_fn = Cosine_PredictingModule(dim_dict['out_dim'])
        elif pred == 'exp_cos':
            self.pred_fn = CosinePredictionWihEdge(dim_dict, dim_dict['out_dim'], dim_dict['orders']['edge_dim'], dim_dict['rev-orders']['edge_dim'])
        else:
            print("prediction function has not been specified!")

    def get_repr(self,
                 blocks,
                 h):

        # print("Edge features", edge_features['orders'].shape)
        
        for i in range(len(blocks)):
            layer = self.layers[i]
            # print(f"layer {i} of {len(self.layers)}")

            edge_features = blocks[i].edata['features']

            # print(edge_features[('customer', 'orders', 'product')].shape, edge_features[('product', 'rev-orders', 'customer')].shape)

            # print(blocks[i])
            edge_features_HM = {}
            for key, value in edge_features.items():
                edge_features_HM[key[1]] = (value, )

            # print(HM['orders'][0].shape, HM['rev-orders'][0].shape)

            h = layer(blocks[i], h, mod_args = edge_features_HM)

        return h
    
    def forward(self,
                blocks,
                h,
                edge_features,
                pos_g,
                neg_g,
                embedding_layer: bool = True
                ):

            # print(h)

            h['customer'] = self.user_embed(h['customer'])
            h['product'] = self.item_embed(h['product'])

            # h = self.get_repr(blocks, h, edge_features)
            h = self.get_repr(blocks, h)

            # print("H-value", h['customer'].shape, h['product'].shape)

            pos_score = self.pred_fn(pos_g, h, pos_graph = True)
            neg_score = self.pred_fn(neg_g, h, pos_graph = False)

            return h, pos_score, neg_score