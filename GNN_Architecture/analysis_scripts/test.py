from Model.model import ConvModel
import dgl, torch
import torch.nn as nn
from evaluation import baseline_model_generator

from collections import defaultdict
import numpy as np
from settings import BASE_DIR, MODEL_DIR


graphs, _ = dgl.load_graphs(f"{BASE_DIR}/model/ecommerce_hetero_graph.dgl")
ecommerce_hetero_graph = graphs[0]

saved_model = torch.load(f"{BASE_DIR}/model/trained_model.pth")

dim_dict = {'customer': ecommerce_hetero_graph.nodes['customer'].data['features'].shape[1],
            'product': ecommerce_hetero_graph.nodes['product'].data['features'].shape[1],
            'edge_dim': ecommerce_hetero_graph.edges['orders'].data['features'].shape[1],
            'hidden_dim' : 128,
            'out_dim': 64
           }


mpnn_model = ConvModel(ecommerce_hetero_graph, 3, dim_dict)
mpnn_model.load_state_dict(saved_model['model_state_dict'])
mpnn_model.eval()



# graphs, _ = dgl.load_graphs(f"{BASE_DIR}/{MODEL_DIR}/train_g.dgl")
# train_g = graphs[0]

# graphs, _ = dgl.load_graphs(f"{BASE_DIR}/{MODEL_DIR}/test_g.dgl")
# test_g = graphs[0]

# graphs, _ = dgl.load_graphs(f"{BASE_DIR}/{MODEL_DIR}/valid_g.dgl")
# valid_g = graphs[0]

# print(ecommerce_hetero_graph)


# print( ecommerce_hetero_graph.nodes('customer'))




neg_sampler = dgl.dataloading.negative_sampler.Uniform(2)
node_sampler = dgl.dataloading.NeighborSampler(fanouts=[-1, -1])

edge_sampler = dgl.dataloading.EdgePredictionSampler(
    node_sampler,
    negative_sampler=neg_sampler,
    exclude='self')

valid_eids_dict = {}

eids = np.arange(valid_g.number_of_edges(etype='orders'))

for e in valid_g.etypes:
    valid_eids_dict[e] = eids

# valid_dataloader = dgl.dataloading.DataLoader(ecommerce_hetero_graph_subgraph, valid_eids_dict, edge_sampler,  shuffle=True, batch_size=1024, num_workers=0)

# print(valid_eids_dict)

# f = open('graph_files/valid_eids_dict.pickle', 'rb')
# valid_eids_dict = pickle.load(f)

# nodeloader_test = dgl.dataloading.DataLoader(
#         valid_g,
#         valid_eids_dict,
#         node_sampler,
#         shuffle=True,
#         batch_size=1024,
#         drop_last=False,
#         num_workers=0
#     )

# print(next(iter(nodeloader_test)))

print("Valid Graph", valid_g)

print("----------")
valid_dataloader = dgl.dataloading.DataLoader(ecommerce_hetero_graph, valid_eids_dict, edge_sampler,  shuffle=True, drop_last= False, batch_size=1024, num_workers=0)

one_iter = next(iter(valid_dataloader))

# print(one_iter[0]['customer'].shape, one_iter[0]['product'].shape)

# for value in one_iter:
#     print("Args : ", value)
#     print("-*-")

y = {ntype: torch.zeros(valid_g.num_nodes(ntype), 64)
         for ntype in valid_g.ntypes}

print(y['customer'].shape)

print(len(valid_dataloader))

count = 0

for arg0 , pos_g, neg_g, blocks in valid_dataloader:
    
    # print("Arg 0",arg0['customer'].shape, arg0['product'].shape, pos_g.ndata['_ID']['customer'].shape, pos_g.ndata['_ID']['product'].shape)
    
    output_nodes = pos_g.ndata[dgl.NID]

    input_features = blocks[0].srcdata['features']
    edge_features = blocks[0].edata['features']

    edge_features_HM = {}
    for key, value in edge_features.items():
        edge_features_HM[key[1]] = (value, )

    input_features['customer'] = mpnn_model.user_embed(input_features['customer'])
    input_features['product'] = mpnn_model.item_embed(input_features['product'])

    # print("Input features shape", input_features['customer'].shape, input_features['product'].shape)
    
    h = mpnn_model.get_repr(blocks, input_features, edge_features_HM)

    # print("Output features shape", h['customer'].shape, h['product'].shape)
    for ntype in h.keys():
        y[ntype][output_nodes[ntype]] = h[ntype]

        print(h[ntype][0])
        # break
    
    print(count)
    count += 1

    # break
 


# # random_model = baseline_model_generator.generate_random_model(ecommerce_hetero_graph, 'customer', 'product')
# baseline_model = baseline_model_generator.generate_popularity_model(ecommerce_hetero_graph, 'orders', 'customer')

# print(baseline_model)