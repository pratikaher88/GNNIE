import pickle, dgl, torch, numpy as np
import torch.nn as nn
from settings import BASE_DIR
from evaluation import baseline_model_generator

with open(f'{BASE_DIR}/graph_files_subgraph/trained_embeddings.pickle', 'rb') as pickle_file:
    train_embeddings = pickle.load(pickle_file)

print(train_embeddings['customer'].shape, train_embeddings['product'].shape)

graphs, _ = dgl.load_graphs(f"{BASE_DIR}/graph_files_subgraph/valid_g.dgl")
valid_g = graphs[0]

graphs, _ = dgl.load_graphs(f"{BASE_DIR}/graph_files_subgraph/ecommerce_hetero_graph_withsf.dgl")
ecommerce_hetero_graph_subgraph = graphs[0]

print(ecommerce_hetero_graph_subgraph)

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

        if user % 10000 == 0:
            print(user)
        
        recs[user] = order
    
    return recs

print("Getting model recs start")

model_recommendations = get_model_recs()

print("Getting model recs end")

with open( f'{BASE_DIR}/graph_files_subgraph/model_recommendations.pickle', 'wb') as f:
    pickle.dump(model_recommendations, f, pickle.HIGHEST_PROTOCOL)


# random_model = baseline_model_generator.generate_random_model(ecommerce_hetero_graph_subgraph, 'customer', 'product')
baseline_model = baseline_model_generator.generate_popularity_model(ecommerce_hetero_graph_subgraph, 'orders', 'customer')

print("Getting populaity model recs start")

with open( f'{BASE_DIR}/graph_files_subgraph/popularity_baseline.pickle', 'wb') as f:
    pickle.dump(baseline_model, f, pickle.HIGHEST_PROTOCOL)

print("Getting populaity model recs end")
