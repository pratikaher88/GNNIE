import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F

# TO DO : not sure how this architecture will affect things
class ConvLayer(nn.Module):

    def __init__(self, in_feats, out_feats, edge_dim, dropout, aggregator_type, norm):

        super().__init__()

        self._in_neigh_feats, self._in_self_feats = in_feats
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.dropout_fn = nn.Dropout(dropout)
        # self.fc_self = nn.Linear(self._out_feats, out_feats, bias=False)
        # self.fc_neigh = nn.Linear(self._out_feats, out_feats, bias=False)

        self.fc_self = nn.Sequential(
            nn.Linear(self._out_feats, out_feats, bias=False),
            # nn.BatchNorm1d(out_feats),
            nn.ReLU())
        
        self.fc_neigh = nn.Sequential(
          nn.Linear(self._out_feats, out_feats, bias=False),
        #   nn.BatchNorm1d(out_feats),
          nn.ReLU())

        # self.fc_preagg = nn.Linear(self._in_neigh_feats, self._out_feats, bias=False)
        self.fc_preagg = nn.Sequential(
          nn.Linear(self._in_neigh_feats, self._out_feats, bias=False),
        #   nn.BatchNorm1d(self._out_feats),
          nn.ReLU())
        

        self.edge_fc = nn.Sequential(
        nn.Linear(edge_dim, self._out_feats*self._out_feats),
        # nn.BatchNorm1d(self._out_feats*self._out_feats),
        nn.ReLU())

        # self.edge_fc = nn.Linear(edge_dim, self._out_feats*self._out_feats)
        self.edge_dim = edge_dim
        self.reset_parameters()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_self[0].weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh[0].weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_preagg[0].weight, gain=gain)
        nn.init.xavier_uniform_(self.edge_fc[0].weight, gain=gain)

    def forward(self, graph, x, edge_features):
        
        # print('------------')
        # print(graph, x[0].shape, x[1].shape)

        # print("Edge feature - pre shape", edge_features.shape)

        h_neigh, h_self = x
        h_neigh = self.dropout_fn(self.fc_preagg(h_neigh))
        h_self = self.dropout_fn(self.fc_preagg(h_self))
        # include edge weights

        edge_weights = self.edge_fc(edge_features).view(-1, self._out_feats, self._out_feats)

        # print("H neigh pre shape",h_neigh.shape)
        # print("Edge weights", h_neigh.shape, edge_weights.shape, self.edge_fc)
        graph.edata['edge_weights'] = edge_weights

        graph.srcdata['h'] = h_neigh

        graph.update_all(
            fn.u_mul_e('h', 'edge_weights', 'm'),
            fn.mean('m', 'neigh'))
        
        # graph.update_all(
        #     fn.copy_src('h', 'm'),
        #     fn.mean('m', 'neigh'))
    
        h_neigh = graph.dstdata['neigh'].sum(dim=1)

        # print("Final shape", h_neigh.shape, h_self.shape, (h_neigh+h_self).shape)

        z = self.fc_self(h_self) + self.fc_neigh(h_neigh)
        # z = h_neigh+h_self

        z = F.relu(z)

        # print("H out", z.shape)

        return z
