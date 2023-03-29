import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
import torch

# TO DO : not sure how this architecture will affect things
class ConvLayer(nn.Module):

    def __init__(self, in_feats, out_feats, edge_dim, edge_hidden_dim,dropout, aggregator_type, norm):

        super().__init__()

        self._in_neigh_feats, self._in_self_feats = in_feats
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self._edge_hidden_dim = edge_hidden_dim
        self.norm = norm
        self.dropout_fn = nn.Dropout(dropout)
        # self.fc_self = nn.Linear(self._out_feats, out_feats, bias=False)
        # self.fc_neigh = nn.Linear(self._out_feats, out_feats, bias=False)

        self.fc_self = nn.Sequential(
            nn.Linear(self._in_self_feats, out_feats, bias=False),
            nn.BatchNorm1d(out_feats),
            nn.Tanh())
        
        self.fc_neigh = nn.Sequential(
          nn.Linear(self._in_neigh_feats, out_feats, bias=False),
          nn.BatchNorm1d(out_feats),
          nn.Tanh())

        # self.fc_preagg = nn.Linear(self._in_neigh_feats, self._out_feats, bias=False)
        # self.fc_preagg = nn.Sequential(
        #   nn.Linear(self._in_neigh_feats, self._out_feats, bias=False),
        #   nn.BatchNorm1d(self._out_feats),
        #   nn.Tanh())

        # self.edge_fc = nn.Sequential(
        # nn.Linear(edge_dim, self._out_feats*self._out_feats,bias=False),
        # nn.BatchNorm1d(self._out_feats*self._out_feats),
        # nn.Tanh(),
        # )

        self.edge_fc = nn.Sequential(
            nn.Linear(edge_dim, self._edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(self._edge_hidden_dim, self._in_neigh_feats*self._out_feats)
        )

        # self.edge_fc = nn.Linear(edge_dim, self._out_feats*self._out_feats)
        self.edge_dim = edge_dim
        self.reset_parameters()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_self[0].weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh[0].weight, gain=gain)
        # nn.init.xavier_uniform_(self.fc_preagg[0].weight, gain=gain)
        nn.init.xavier_uniform_(self.edge_fc[0].weight, gain=gain)

    def forward(self, graph, x, edge_features):
        
        # print('------------')
        # print("Edge feature - pre shape", edge_features.shape)

        h_neigh, h_self = x

        # print("Shape", h_neigh.shape, h_self.shape, self._in_self_feats, self._out_feats, h_neigh.unsqueeze(-1).shape, h_self.unsqueeze(-1).shape )

        # if (h_neigh.shape != self.fc_preagg(h_neigh).shape or h_self.shape != self.fc_preagg(h_self).shape ):
        #     print(h_neigh.shape , self.fc_preagg(h_neigh).shape)

        # h_neigh = self.dropout_fn(self.fc_preagg(h_neigh))
        # h_self = self.dropout_fn(self.fc_preagg(h_self))
        # h_neigh = self.dropout_fn(h_neigh.unsqueeze(-1))
        # h_self = self.dropout_fn(h_self.unsqueeze(-1))
        # include edge weights

        graph.srcdata['h'] = self.dropout_fn(h_neigh.unsqueeze(-1))

        edge_weights = self.edge_fc(edge_features).view(-1, self._in_neigh_feats, self._out_feats)

        # print("H neigh pre shape",h_neigh.shape)
        # print("Edge weights", h_neigh.shape, edge_weights.shape, self.edge_fc(edge_features).shape)
        graph.edata['edge_weights'] = edge_weights


        if self._aggre_type == 'max':
            graph.update_all(
                fn.u_mul_e('h', 'edge_weights', 'm'),
                fn.max('m', 'neigh'))
        
        if self._aggre_type == 'sum':
            graph.update_all(
                fn.u_mul_e('h', 'edge_weights', 'm'),
                fn.sum('m', 'neigh'))

        elif self._aggre_type == 'mean':

            graph.update_all(
                fn.u_mul_e('h', 'edge_weights', 'm'),
                fn.mean('m', 'neigh'))
            
        # graph.update_all(
        #     fn.copy_src('h', 'm'),
        #     fn.mean('m', 'neigh'))
    
        h_neigh = graph.dstdata['neigh'].sum(dim=1)

        # return h_neigh

        # print("Final shape", h_neigh.shape, h_self.shape)
        # print(self.fc_neigh(h_neigh).shape)

        # Experiment : can get rid of this
        # z = self.fc_self(h_self) + self.fc_neigh(h_neigh)
        z = self.fc_self(h_self) + h_neigh
        # z = h_neigh+h_self

        z = F.relu(z)

        z_norm = z.norm(2, 1, keepdim=True)
        z_norm = torch.where(z_norm == 0,
                                torch.tensor(1.).to(z_norm),
                                z_norm)
        z = z / z_norm

        # print("H out", z.shape)
        # print('-----------------------------------------------')

        return z
