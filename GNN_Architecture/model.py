import torch.nn as nn
import dgl.nn.pytorch as dglnn
from layer import ConvLayer
from prediction import CosinePrediction, PredictingModule, PredictingLayer, Cosine_PredictingModule, \
    Cosine_PredictingLayer


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

    def forward(self,
                node_feats):
        x = self.proj_feats(node_feats)
        return x


class ConvModel(nn.Module):

    def __init__(self, g, n_layers, dim_dict, norm: bool = True, dropout: float = 0.0, aggregator_type: str = 'mean',
                 pred: str = 'cos', aggregator_hetero: str = 'sum', embedding_layer: bool = True):

        super(ConvModel, self).__init__()

        self.user_embed = NodeEmbedding(dim_dict['customer'], dim_dict['hidden_dim'])
        self.item_embed = NodeEmbedding(dim_dict['product'], dim_dict['hidden_dim'])

        self.layers = nn.ModuleList()

        # hidden_dim layer
        for _ in range(n_layers - 2):
            self.layers.append(
                dglnn.HeteroGraphConv(
                    {etype[1]: ConvLayer((dim_dict['hidden_dim'], dim_dict['hidden_dim']), dim_dict['hidden_dim'],
                                         dim_dict['edge_dim'], dropout,
                                         aggregator_type, norm)
                     for etype in g.canonical_etypes},
                    aggregate=aggregator_hetero))

        # output layer

        # TODO : output dimension was dim_dict['out_dim'] (instead of  dim_dict['hidden_dim']) before so I am not sure what to do 
        self.layers.append(
            dglnn.HeteroGraphConv(
                {etype[1]: ConvLayer((dim_dict['hidden_dim'], dim_dict['hidden_dim']), dim_dict['out_dim'],
                                     dim_dict['edge_dim'], dropout,
                                     aggregator_type, norm)
                 for etype in g.canonical_etypes},
                aggregate=aggregator_hetero))
        if pred == 'cos':
            self.pred_fn = CosinePrediction()
        elif pred == 'nn':
            self.pred_fn = PredictingModule(PredictingLayer, dim_dict['out_dim'])
        elif pred == 'cos_nn':
            self.pred_fn = Cosine_PredictingModule(Cosine_PredictingLayer, dim_dict['out_dim'])
        else:
            print("prediction function has not been specified!")

    def get_repr(self,
                 blocks,
                 h,
                 edge_features):

        # print("Edge features", edge_features['orders'].shape)

        for i in range(len(blocks)):
            layer = self.layers[i]
            # print(f"layer {i} of {len(self.layers)}")

            edge_features = blocks[i].edata['features']

            # print(edge_features[('customer', 'orders', 'product')].shape, edge_features[('product', 'rev-orders', 'customer')].shape)

            # print(blocks[i])
            HM = {}
            for key, value in edge_features.items():
                HM[key[1]] = (value,)

            # print(HM['orders'][0].shape, HM['rev-orders'][0].shape)

            h = layer(blocks[i], h, mod_args=HM)

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

        h = self.get_repr(blocks, h, edge_features)

        print("H-value", h['customer'].shape, h['product'].shape)

        # print("graphs", pos_g, neg_g)

        pos_score = self.pred_fn(pos_g, h)
        neg_score = self.pred_fn(neg_g, h)

        return h, pos_score, neg_score
