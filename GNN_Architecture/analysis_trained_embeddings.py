
from collections import defaultdict
import pickle, dgl
import numpy as np
import torch.nn as nn
from settings import BASE_DIR

f = open(f"{BASE_DIR}graph_files_latest_run/trained_embeddings.pickle", 'rb')
y = pickle.load(f)

print(y['customer'][0], y['product'][1])

graphs, _ = dgl.load_graphs(f"{BASE_DIR}graph_files_latest_run/valid_g.dgl")
valid_g = graphs[0]

graphs, _ = dgl.load_graphs(f"{BASE_DIR}graph_files_latest_run/train_g.dgl")
train_g = graphs[0]

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

print("creating already related dict")
already_rated_dict = create_already_rated(train_g)

print("building valid_g recs list")
recommendations_from_valid_graph = get_test_recs(valid_g)

user_ids = valid_g.num_nodes('customer')
    
recs = {}

print("building actual recs")
for user in range(100):

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

    if user % 100 == 0:
        print(user)

print(recs[0], recs[1])

def compare_rec(valid_recs, model_recs):
  
  total = 0
  correct = 0 

  for key, value in model_recs.items():

    model_recs_list = model_recs[key]
    test_recs_list = valid_recs[key]

    recommended_movies_correct = set(list(set(model_recs_list) & set(test_recs_list)))

    if len(set(test_recs_list)) > 0:

        # print("User ID", key, "Correctly predicted movies", recommended_movies_correct)
        # print("Total test values", len(recommended_movies_correct), "out of", len(set(test_recs_list)))
        
        correct += len(recommended_movies_correct)
        total += len(set(test_recs_list))
    # print("Ratings", [ ratings_HM[movie_id] for movie_id in recommended_movies_correct ])

  return correct, total

print("comparing recs")
print(compare_rec(recommendations_from_valid_graph, recs))

