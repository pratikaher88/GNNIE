
from model import ConvModel
import dgl, torch
from collections import defaultdict
import numpy as np
import torch.nn as nn
import time
from settings import BASE_DIR

print(time.time())


graphs, _ = dgl.load_graphs(f"{BASE_DIR}/graph_file_vmcloud/ecommerce_hetero_graph_subgraph.dgl")
ecommerce_hetero_graph_subgraph = graphs[0]

graphs, _ = dgl.load_graphs(f"{BASE_DIR}/graph_file_vmcloud/ecommerce_hetero_graph.dgl")
ecommerce_hetero_graph = graphs[0]

graphs, _ = dgl.load_graphs(f"{BASE_DIR}/graph_file_vmcloud/train_g.dgl")
train_g = graphs[0]

graphs, _ = dgl.load_graphs(f"{BASE_DIR}/graph_file_vmcloud/test_g.dgl")
test_g = graphs[0]

graphs, _ = dgl.load_graphs(f"{BASE_DIR}/graph_file_vmcloud/valid_g.dgl")
valid_g = graphs[0]

dim_dict = {'customer': ecommerce_hetero_graph_subgraph.nodes['customer'].data['features'].shape[1],
            'product': ecommerce_hetero_graph_subgraph.nodes['product'].data['features'].shape[1],
            'edge_dim': ecommerce_hetero_graph_subgraph.edges['orders'].data['features'].shape[1],
            'hidden_dim' : 128,
            'out_dim': 64
           }

saved_model = torch.load(f'{BASE_DIR}/graph_files_full/trained_model.pth')

mpnn_model = ConvModel(ecommerce_hetero_graph_subgraph, 3, dim_dict)
mpnn_model.load_state_dict(saved_model['model_state_dict'])
mpnn_model.eval()


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

import pickle
with open( f'{BASE_DIR}/graph_files_full/trained_embeddings.pickle', 'wb') as f:
    pickle.dump(y, f, pickle.HIGHEST_PROTOCOL)

# print(y['product'])
## check what they do on valid graph

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

user_ids = valid_g.num_nodes('customer')
    
recs = {}

for user in range(user_ids):

    already_rated = already_rated_dict[user]

    user_emb = y['customer'][user]
    # user_emb_rpt = torch.cat(valid_g.num_nodes('product')*[user_emb]).reshape(-1, dim_dict['out_dim'])
    user_emb_rpt = user_emb.repeat(valid_g.num_nodes('product'), 1)

    # print("User embedding shape",y['product'].shape, user_emb_rpt.shape)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    ratings = cos(user_emb_rpt, y['product'])
    
    ratings_formatted = ratings.detach().numpy().reshape(valid_g.num_nodes('product'),)
    order = np.argsort(-ratings_formatted)
    
    order = [item for item in order if item not in already_rated]
    
    rec = order[:50]
    recs[user] = rec

print(recs)

def compare_rec(valid_recs, model_recs):
  
  total = 0
  correct = 0 

  for key, value in model_recs.items():

    model_recs_list = model_recs[key]
    test_recs_list = valid_recs[key]

    recommended_movies_correct = set(list(set(model_recs_list) & set(test_recs_list)))

    if len(set(test_recs_list)) > 0:

        print("User ID", key, "Correctly predicted movies", recommended_movies_correct)
        print("Total test values", len(recommended_movies_correct), "out of", len(set(test_recs_list)))
        
        correct += len(recommended_movies_correct)
        total += len(set(test_recs_list))
    # print("Ratings", [ ratings_HM[movie_id] for movie_id in recommended_movies_correct ])

  return correct, total


print(compare_rec(recommendations_from_valid_graph, recs))

print(time.time())
