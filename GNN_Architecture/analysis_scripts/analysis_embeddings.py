from Model.model import ConvModel
import dgl, torch, os, yaml
import torch.nn as nn
from evaluation import baseline_model_generator
from collections import defaultdict
import numpy as np
import pickle
from settings import BASE_DIR, CONFIG_PATH
from evaluation.evaluation_metrics import mmr,hit_rate_precision, hit_rate_recall

# Function to load yaml configuration file
print("2",BASE_DIR)
print("3",CONFIG_PATH)

def load_config(config_name):
    print(os.path.join(f"{CONFIG_PATH}", config_name))
    with open(os.path.join(f"{CONFIG_PATH}", config_name)) as file:
        config = yaml.safe_load(file)
    return config

model_config = load_config("../model_config.yml")

graphs, _ = dgl.load_graphs(f"../run_data/graph_files_subgraph/ecommerce_hetero_graph_subgraph.dgl")
ecommerce_hetero_graph_subgraph = graphs[0]

graphs, _ = dgl.load_graphs(f"../run_data/graph_files_subgraph/train_g.dgl")
train_g = graphs[0]

graphs, _ = dgl.load_graphs(f"../run_data/graph_files_subgraph/test_g.dgl")
test_g = graphs[0]

graphs, _ = dgl.load_graphs(f"../run_data/graph_files_subgraph/valid_g.dgl")
valid_g = graphs[0]

dim_dict = {'customer': ecommerce_hetero_graph_subgraph.nodes['customer'].data['features'].shape[1],
            'product': ecommerce_hetero_graph_subgraph.nodes['product'].data['features'].shape[1],
            'edge_dim': ecommerce_hetero_graph_subgraph.edges['orders'].data['features'].shape[1],
            'edge_hidden_dim': model_config['edge_hidden_dim'],
            'hidden_dim' : model_config['hidden_dim'],
            'out_dim': model_config['output_dim']}

saved_model = torch.load(f"../run_data/graph_files_subgraph/trained_model.pth")

def get_ground_truth_recs(g):

    customers_test, products_test  = (g.edges(etype='orders'))
    test_rated_arr = np.stack((np.asarray(customers_test), np.asarray(products_test)), axis=1)
    test_rated_dict = defaultdict(list)
    
    for key, val in test_rated_arr:
        test_rated_dict[key].append(val)
    
    return {customer: list(set(products)) for customer, products in test_rated_dict.items()}

recommendations_from_valid_graph = get_ground_truth_recs(valid_g)

with open(f"../run_data/graph_files_subgraph/trained_embeddings.pickle", 'rb') as pickle_file:
    train_embeddings = pickle.load(pickle_file)

print(train_embeddings['customer'][1].shape, train_embeddings['customer'].shape, train_embeddings['product'].shape)

print('zeros count : ', (train_embeddings['product'][5].shape[0] - torch.count_nonzero(train_embeddings['product'][5])).item(), "out of",train_embeddings['product'][5].shape[0])

def get_model_recs():

    user_ids = valid_g.num_nodes('customer')
        
    recs = {}

    for user in range(user_ids):

        user_emb = train_embeddings['customer'][user]
        # user_emb_rpt = torch.cat(valid_g.num_nodes('product')*[user_emb]).reshape(-1, dim_dict['out_dim'])
        user_emb_rpt = user_emb.repeat(valid_g.num_nodes('product'), 1)

        # print("User embedding shape",y['product'].shape, user_emb_rpt.shape)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        ratings = cos(user_emb_rpt, train_embeddings['product'])
        
        ratings_formatted = ratings.detach().numpy().reshape(valid_g.num_nodes('product'),)
        order = np.argsort(-ratings_formatted)
        
        # already_rated = already_rated_dict[user]
        # order = [item for item in order if item not in already_rated]
        
        recs[user] = order
    
    return recs

# print(recs)

model_recommendations = get_model_recs()

print("Model recs length",len(model_recommendations))
print("Valid graph length",len(recommendations_from_valid_graph))

# use thif function to see actual recommendations
# print(compare_rec(recommendations_from_valid_graph, model_recommendations))

random_model = baseline_model_generator.generate_random_model(ecommerce_hetero_graph_subgraph, 'customer', 'product')
baseline_model = baseline_model_generator.generate_popularity_model(ecommerce_hetero_graph_subgraph, 'orders', 'customer')

# print(model_recommendations[446][:20], len(model_recommendations[446]))
# print(model_recommendations)


print("MMR Baseline: ", mmr(recommendations_from_valid_graph, baseline_model, 1))
print("MMR Random: ", mmr(recommendations_from_valid_graph, random_model, 1))
print("MMR GNN Model: ", mmr(recommendations_from_valid_graph, model_recommendations, 1))

thresholds = [5,15,30]

for t in thresholds:
    print('K=',t)
    print('HR-precision popularity: ', round(hit_rate_precision(recommendations_from_valid_graph, baseline_model, t),5))
    print('HR-precision random: ', round(hit_rate_precision(recommendations_from_valid_graph, random_model, t),5))
    print('HR-precision GNNIE: ', round(hit_rate_precision(recommendations_from_valid_graph, model_recommendations, t),5))
    print('HR-recall popularity: ', round(hit_rate_recall(recommendations_from_valid_graph, baseline_model, t),5))
    print('HR-recall random: ', round(hit_rate_recall(recommendations_from_valid_graph, random_model, t),5))
    print('HR-recall GNNIE: ', round(hit_rate_recall(recommendations_from_valid_graph, model_recommendations, t),5))
