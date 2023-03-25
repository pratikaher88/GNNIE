from model import ConvModel
import dgl, torch, pickle
from collections import defaultdict
import numpy as np
import time
from settings import BASE_DIR

start = time.time()

graphs, _ = dgl.load_graphs(f"{BASE_DIR}/graph_files/ecommerce_hetero_graph_subgraph.dgl")
ecommerce_hetero_graph_subgraph = graphs[0]

graphs, _ = dgl.load_graphs(f"{BASE_DIR}/graph_files/train_g.dgl")
train_g = graphs[0]

graphs, _ = dgl.load_graphs(f"{BASE_DIR}/graph_files/test_g.dgl")
test_g = graphs[0]

graphs, _ = dgl.load_graphs(f"{BASE_DIR}/graph_files/valid_g.dgl")
valid_g = graphs[0]

dim_dict = {'customer': ecommerce_hetero_graph_subgraph.nodes['customer'].data['features'].shape[1],
            'product': ecommerce_hetero_graph_subgraph.nodes['product'].data['features'].shape[1],
            'edge_dim': ecommerce_hetero_graph_subgraph.edges['orders'].data['features'].shape[1],
            'hidden_dim' : 128,
            'out_dim': 64
           }

saved_model = torch.load(f"{BASE_DIR}/graph_files/trained_model.pth")

mpnn_model = ConvModel(ecommerce_hetero_graph_subgraph, 3, dim_dict)
mpnn_model.load_state_dict(saved_model['model_state_dict'])
mpnn_model.eval()

from collections import defaultdict

# for test, get ground truth entites (entites that users will request in the future)
def get_test_recs(g):

    customers_test, products_test  = (g.edges(etype='orders'))
    already_rated_arr = np.stack((np.asarray(customers_test), np.asarray(products_test)), axis=1)
    test_rated_dict = defaultdict(list)
    
    for key, val in already_rated_arr:
        test_rated_dict[key].append(val)
        
    return test_rated_dict

def create_already_rated(g):
    
    customers_train, product_train  = (g.edges(etype='orders'))
    already_rated_arr = np.stack((np.asarray(customers_train), np.asarray(product_train)), axis=1)
    already_rated_dict = defaultdict(list)
    
    for key, val in already_rated_arr:
        already_rated_dict[key].append(val)
        
    return already_rated_dict

already_rated_dict = create_already_rated(train_g)
recommendations_from_valid_graph = get_test_recs(valid_g)

# print(train_g, test_g)

# for key, value in already_rated_dict.items():
#     print(already_rated_dict[key], recommendations_from_valid_graph[key])


# edge_features = valid_g.edata['features']

# edge_features_HM = {}
# for key, value in edge_features.items():
#     edge_features_HM[key[1]] = value.to(torch.float)

# h = mpnn_model.get_repr(valid_g, valid_g.ndata['features'], edge_features_HM)

# print("h features", h['customer'].shape)

# validation recs get embeddings

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

# f = open('f"{BASE_DIR}/graph_files/valid_eids_dict.pickle', 'rb')
# valid_eids_dict = pickle.load(f)

# nodeloader_test = dgl.dataloading.DataLoader(
#         valid_g,
#         valid_eids_dict,
#         node_sampler,
#         shuffle=True,
#         batch_size=100,
#         drop_last=False,
#         num_workers=0
#     )

# print(next(iter(nodeloader_test))[0]['customer'].shape)
# print(next(iter(nodeloader_test)))
# print(len(nodeloader_test))
# print("Valid eids", valid_g , valid_g.edges(etype='orders'))

# print('--------------------------------------')

# exit()

print(valid_g)

valid_dataloader = dgl.dataloading.DataLoader(ecommerce_hetero_graph_subgraph, valid_eids_dict, edge_sampler,  shuffle=True, drop_last= False, batch_size=1024, num_workers=0)

y = {ntype: torch.zeros(valid_g.num_nodes(ntype), 64)
         for ntype in valid_g.ntypes}

for arg0 , pos_g, neg_g, blocks in valid_dataloader:

    output_nodes = pos_g.ndata[dgl.NID]

    input_features = blocks[0].srcdata['features']
    edge_features = blocks[0].edata['features']

    edge_features_HM = {}
    for key, value in edge_features.items():
        edge_features_HM[key[1]] = (value, )
    
    input_features['customer'] = mpnn_model.user_embed(input_features['customer'])
    input_features['product'] = mpnn_model.item_embed(input_features['product'])

    print("Input features shape", input_features['customer'].shape, input_features['product'].shape)
    
    h = mpnn_model.get_repr(blocks, input_features, edge_features_HM)

    print("Output features shape", h['customer'].shape, h['product'].shape)
    for ntype in h.keys():
        y[ntype][output_nodes[ntype]] = h[ntype]

    # for ntype in h.keys():
    #     # print(ntype, output_nodes)
    #     y[ntype][output_nodes['orders']] = h[ntype]


print(y['customer'].shape, y['product'].shape)

# y = defaultdict(dict)

# # TODO : check what is wrong with the output feature stuff
# for _, pos_g, neg_g, blocks in valid_dataloader:
#   input_features = blocks[0].srcdata['features']
#   edge_features = blocks[0].edata['features']

#   edge_features_HM = {}
#   for key, value in edge_features.items():
#       edge_features_HM[key[1]] = (value, )

#   input_features['customer'] = mpnn_model.user_embed(input_features['customer'])
#   input_features['product'] = mpnn_model.item_embed(input_features['product'])

#   print(input_features['customer'].shape, input_features['product'].shape)

#   h = mpnn_model.get_repr(blocks, input_features, edge_features_HM)

#   print("H-value", h)
#   print(h['customer'].shape, h['product'].shape)

#   for ntype in h.keys():
# #     print(h[ntype].shape)
#     y[ntype][output_nodes[ntype]] = h[ntype]    
#     # print(ip_nodes[ntype].shape)

print(time.time() - start)
