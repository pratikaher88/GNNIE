import dgl, torch
import numpy as np
from model import ConvModel

graphs, _ = dgl.load_graphs("graph_files/train_g.dgl")
train_g = graphs[0]

graphs, _ = dgl.load_graphs("graph_files/ecommerce_hetero_graph.dgl")
ecommerce_hetero_graph = graphs[0]

# print(ecommerce_hetero_graph)

# eids = np.arange(ecommerce_hetero_graph.number_of_edges(etype='orders'))
# eids = np.random.permutation(eids)

# test_size = int(len(eids) * 0.1)
# valid_size = int(len(eids) * 0.1)
# train_size = len(eids) - test_size - valid_size

# train_eids_dict = {}
# for e in ecommerce_hetero_graph.etypes:
#     train_eids_dict[e] = eids[:train_size]

dim_dict = {'customer': train_g.nodes['customer'].data['features'].shape[1],
            'product': train_g.nodes['product'].data['features'].shape[1],
            'edge_dim': train_g.edges['orders'].data['features'].shape[1],
            'hidden_dim' : 128,
            'out_dim': 16
           }

# sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)

# train_dataloader = dgl.dataloading.DataLoader(
#         train_g, train_eids_dict, 
#         sampler, 
#         # negative_sampler=dgl.dataloading.negative_sampler.Uniform(5), 
#         shuffle=True,
#         # drop_last=False,
#         batch_size = 32,
#         num_workers=0
#         )


sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

eids = np.arange(train_g.number_of_edges(etype='orders'))
eids = np.random.permutation(eids)
train_eids_dict = {}

test_size = int(len(eids) * 0.1)
valid_size = int(len(eids) * 0.1)
train_size = len(eids) - test_size - valid_size

for e in train_g.etypes:
    train_eids_dict[e] = eids[:train_size]

# dataloader = dgl.dataloading.DataLoader(
#     train_g, ids_dict, sampler,
#     batch_size=16,
#     shuffle=True,
#     drop_last=False,
#     num_workers=0)

print(train_g.etypes)

neg_sampler = dgl.dataloading.negative_sampler.Uniform(2)
node_sampler = dgl.dataloading.NeighborSampler(fanouts=[-1, -1])
# node_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
# not sure which sampler to use when
edge_sampler = dgl.dataloading.EdgePredictionSampler(
    node_sampler,
    negative_sampler=neg_sampler,
    # reverse_etypes = {'orders' : 'rev-orders'},
    exclude='self')

# sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)

# dataloader = dgl.dataloading.DataLoader(train_g,    
#                                         train_eids_dict, 
#                                         sampler,    
#                                         # negative_sampler=dgl.dataloading.negative_sampler.Uniform(5),
#                                         shuffle=True, 
#                                         batch_size=16, 
#                                         num_workers=0)

dataloader = dgl.dataloading.DataLoader(train_g, train_eids_dict, edge_sampler,  shuffle=True, batch_size=16, num_workers=0)

# input_nodes, pos_g, neg_g, blocks = next(iter(dataloader))

# print(len(next(iter(dataloader))))

# print("Input Nodes", input_nodes ) 
# print("Pos G",pos_g)
# print("Neg G",neg_g)
# print("Blocks",blocks)

# print("reverse orders",train_g.edata)

model = ConvModel(train_g, 3, dim_dict)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=0)

for _, pos_g, neg_g, blocks in dataloader:

    optimizer.zero_grad()

    input_features = blocks[0].srcdata['features']

    edge_features = blocks[0].edata['features']

    HM = {}
    for key, value in edge_features.items():
        HM[key[1]] = (value, )
    
    print("Edge Features shape : ", HM['orders'][0].shape, HM['rev-orders'][0].shape)

    # print(input_features)
    # print(len(blocks))

    # _, pos_score, neg_score = 
    model(blocks, input_features, HM, pos_g, neg_g)

    # _, pos_score, neg_score = model(blocks, input_nodes, pos_g, neg_g, input_nodes)
    # print(pos_score)

    break

