from Model.model import ConvModel
import dgl, torch
import torch.nn as nn
from evaluation import baseline_model_generator

from collections import defaultdict
import numpy as np
from settings import BASE_DIR, MODEL_DIR

graphs, _ = dgl.load_graphs(f"{BASE_DIR}/{MODEL_DIR}/ecommerce_hetero_graph.dgl")
ecommerce_hetero_graph_subgraph = graphs[0]

graphs, _ = dgl.load_graphs(f"{BASE_DIR}/{MODEL_DIR}/train_g.dgl")
train_g = graphs[0]

graphs, _ = dgl.load_graphs(f"{BASE_DIR}/{MODEL_DIR}/test_g.dgl")
test_g = graphs[0]

graphs, _ = dgl.load_graphs(f"{BASE_DIR}/{MODEL_DIR}/valid_g.dgl")
valid_g = graphs[0]

dim_dict = {'customer': ecommerce_hetero_graph_subgraph.nodes['customer'].data['features'].shape[1],
            'product': ecommerce_hetero_graph_subgraph.nodes['product'].data['features'].shape[1],
            'edge_dim': ecommerce_hetero_graph_subgraph.edges['orders'].data['features'].shape[1],
            'hidden_dim' : 128,
            'out_dim': 64
           }

saved_model = torch.load(f"{BASE_DIR}/{MODEL_DIR}/trained_model.pth")

mpnn_model = ConvModel(ecommerce_hetero_graph_subgraph, 3, dim_dict)

print(mpnn_model.layers)

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
    
    final_history = defaultdict(list)
    for key, value in test_rated_dict.items():
        final_history[key] = list(set(value))
        
    return final_history
    # return test_rated_dict

def create_already_rated(g):
    
    customers_train, product_train  = (g.edges(etype='orders'))
    already_rated_arr = np.stack((np.asarray(customers_train), np.asarray(product_train)), axis=1)
    already_rated_dict = defaultdict(list)
    
    for key, val in already_rated_arr:
        already_rated_dict[key].append(val)
        
    return already_rated_dict

already_rated_dict = create_already_rated(train_g)
recommendations_from_valid_graph = get_test_recs(valid_g)

neg_sampler = dgl.dataloading.negative_sampler.Uniform(2)
node_sampler = dgl.dataloading.NeighborSampler(fanouts=[-1, -1])

edge_sampler = dgl.dataloading.EdgePredictionSampler(
    node_sampler,
    negative_sampler=neg_sampler,
    exclude='self')

valid_eids_dict = {}
for e in valid_g.edata[dgl.EID].keys():
    valid_eids_dict[e[1]] = valid_g.edata[dgl.EID][e]

print(valid_g)

valid_dataloader = dgl.dataloading.DataLoader(ecommerce_hetero_graph_subgraph, valid_eids_dict, edge_sampler,  shuffle=True, drop_last= False, batch_size=1024, num_workers=0)

train_embeddings = {ntype: torch.zeros(valid_g.num_nodes(ntype), dim_dict['out_dim'])
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
        train_embeddings[ntype][output_nodes[ntype]] = h[ntype]

print(train_embeddings['customer'][1], train_embeddings['customer'].shape, train_embeddings['product'].shape)

print('zeros count : ', (train_embeddings['product'][5].shape[0] - torch.count_nonzero(train_embeddings['product'][5])).item(), "out of",train_embeddings['product'][5].shape[0])

def get_model_recs():

    user_ids = valid_g.num_nodes('customer')
        
    recs = {}

    for user in range(user_ids):

        already_rated = already_rated_dict[user]

        user_emb = train_embeddings['customer'][user]
        # user_emb_rpt = torch.cat(valid_g.num_nodes('product')*[user_emb]).reshape(-1, dim_dict['out_dim'])
        user_emb_rpt = user_emb.repeat(valid_g.num_nodes('product'), 1)

        # print("User embedding shape",y['product'].shape, user_emb_rpt.shape)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        ratings = cos(user_emb_rpt, train_embeddings['product'])
        
        ratings_formatted = ratings.detach().numpy().reshape(valid_g.num_nodes('product'),)
        order = np.argsort(-ratings_formatted)
        
        # order = [item for item in order if item not in already_rated]
        
        recs[user] = order
    
    return recs

# print(recs)

model_recommendations = get_model_recs()


def compare_rec(ground_truth_recs, model_recs, threshold = 10):
  
  total = 0
  correct = 0 

  for key, value in model_recs.items():

    model_recs_list = model_recs[key]
    ground_truth_recs_list = ground_truth_recs[key][:10]

    recommended_movies_correct = list(set(model_recs_list) & set(ground_truth_recs_list))

    if len(set(ground_truth_recs_list)) > 0:

        # print("User ID", key, "Correctly predicted movies", recommended_movies_correct)
        # print("Total test values", len(recommended_movies_correct), "out of", len(set(test_recs_list)))
        
        correct += len(recommended_movies_correct)
        total += len(set(ground_truth_recs_list))
    # print("Ratings", [ ratings_HM[movie_id] for movie_id in recommended_movies_correct ])

  return correct, total


print(compare_rec(recommendations_from_valid_graph, model_recommendations))
random_model = baseline_model_generator.generate_random_model(ecommerce_hetero_graph_subgraph, 'customer', 'product')
baseline_model = baseline_model_generator.generate_popularity_model(ecommerce_hetero_graph_subgraph, 'orders', 'customer')

HM = defaultdict(list)
for key, value in recommendations_from_valid_graph.items():
    HM[key] = list(set(value))

print(HM[446])
print(model_recommendations[446])
print(model_recommendations[0])

from evaluation.evaluation_metrics import mmr,hit_rate_accuracy

# print("MMR GNN Model: ", hit_rate_accuracy(HM, model_recommendations, 10))
print("MMR Random Model: ", hit_rate_accuracy(HM, random_model, 10))
print("MMR Popularity Model: ", hit_rate_accuracy(HM, baseline_model, 10))

