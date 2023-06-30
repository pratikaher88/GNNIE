from Model.model import ConvModel
import dgl, torch, os, yaml
from datetime import datetime
import torch.nn as nn
from evaluation import baseline_model_generator
from collections import defaultdict
import numpy as np
from settings import BASE_DIR, CONFIG_PATH, MODEL_DIR
from evaluation.evaluation_metrics import mmr,hit_rate_precision, hit_rate_recall, rbo
import matplotlib.pyplot as plt
import random, pickle

# with open( f'{BASE_DIR}/graph_files_subgraph/trained_embeddings.pickle', 'wb') as pickle_file:
#     train_embeddings = pickle.load(pickle_file)

def load_config(config_name):
    with open(os.path.join(f"{CONFIG_PATH}", config_name)) as file:
        config = yaml.safe_load(file)
    return config

print("Loading model config")
model_config = load_config("model_config.yml")
graph_details = model_config['graph_details']
# op_file = open(f"{BASE_DIR}/{MODEL_DIR}/output.txt", "a")

current_directory = os.path.join(os.getcwd(), BASE_DIR)

# Create a new directory
new_directory = os.path.join(current_directory, datetime.now().strftime('logfile_%H_%M_%d_%m_%Y'))
os.mkdir(new_directory)

new_file_path = os.path.join(new_directory, 'logfile.log')

op_file = open(new_file_path, 'w')

print('-----------------------------------------------',  file=op_file)
print(file=op_file)
print(model_config, file=op_file)
print(file=op_file)


graph_name = model_config['input_graph_name']

with open(f'{BASE_DIR}/graph_files_subgraph/trained_embeddings.pickle', 'rb') as pickle_file:
    train_embeddings = pickle.load(pickle_file)

graphs, _ = dgl.load_graphs(f"{BASE_DIR}/graph_files_subgraph/valid_g.dgl")
valid_g = graphs[0]

graphs, _ = dgl.load_graphs(f"{BASE_DIR}/graph_files_subgraph/{graph_name}")
ecommerce_hetero_graph_subgraph = graphs[0]

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
print("Sample model rec", model_recommendations)

# print("Valid graph length",len(recommendations_from_valid_graph))

# with open( f'{BASE_DIR}/graph_files_subgraph/model_recommendations.pickle', 'wb') as f:
#     pickle.dump(model_recommendations, f, pickle.HIGHEST_PROTOCOL)

# # with open( f'{BASE_DIR}/graph_files_subgraph/recommendations_from_valid_graph.pickle', 'wb') as f:
# #     pickle.dump(recommendations_from_valid_graph, f, pickle.HIGHEST_PROTOCOL)

# def compare_rec(ground_truth_recs, model_recs, threshold = 10):
  
#   total = 0
#   correct = 0 

#   for key, value in ground_truth_recs.items():

#     model_recs_list = model_recs[key][:10]
#     ground_truth_recs_list = ground_truth_recs[key]

#     recommended_movies_correct = list(set(model_recs_list) & set(ground_truth_recs_list))

#     if len(set(ground_truth_recs_list)) > 0:

#         if len(recommended_movies_correct) >= 2:
#             print("User ID", key, "Correctly predicted movies", recommended_movies_correct)
#             print("Total test values", len(recommended_movies_correct), "out of", len(set(ground_truth_recs_list)))
        
#         correct += len(recommended_movies_correct)
#         total += len(set(ground_truth_recs_list))

#   return correct, total


# # use thif function to see actual recommendations
# print(compare_rec(recommendations_from_valid_graph, model_recommendations))

def get_ground_truth_recs(g):

    customers_test, products_test  = (g.edges(etype='orders'))
    test_rated_arr = np.stack((np.asarray(customers_test), np.asarray(products_test)), axis=1)
    test_rated_dict = defaultdict(list)
    
    for key, val in test_rated_arr:
        test_rated_dict[key].append(val)
    
    return {customer: list(set(products)) for customer, products in test_rated_dict.items()}


random_model = baseline_model_generator.generate_random_model(ecommerce_hetero_graph_subgraph, 'customer', 'product')
baseline_model = baseline_model_generator.generate_popularity_model(ecommerce_hetero_graph_subgraph, 'orders', 'customer')

recommendations_from_valid_graph = get_ground_truth_recs(valid_g)


# # print(model_recommendations[446][:20], len(model_recommendations[446]))
# # print(model_recommendations)
# # print(random_model)
# # print(baseline_model)

# with open( f'{BASE_DIR}/graph_files_subgraph/popularity_baseline.pickle', 'wb') as f:
#     pickle.dump(baseline_model, f, pickle.HIGHEST_PROTOCOL)


# ## EVALUATION METRICS

# # random_model = baseline_model_generator.generate_random_model(ecommerce_hetero_graph_subgraph, 'customer', 'product')
# # baseline_model = baseline_model_generator.generate_popularity_model(ecommerce_hetero_graph_subgraph, 'orders', 'customer')

# print("graph_details: ", graph_details, file=op_file)

# MRR
print("MRR Popular: ", mmr(recommendations_from_valid_graph, baseline_model, 1), file=op_file)
print("MRR Random: ", mmr(recommendations_from_valid_graph, random_model, 1), file=op_file)
print("MRR GNN Model: ", mmr(recommendations_from_valid_graph, model_recommendations, 1), file=op_file)

# #HIT RATES
thresholds = [5,10,15,20,25,30,35,40,45,50]

hit_rates_prec_baseline = []
hit_rates_prec_random = []
hit_rates_prec_model = []

hit_rates_recall_baseline = []
hit_rates_recall_random = []
hit_rates_recall_model = []

for t in thresholds:
    hit_rates_prec_baseline.append(hit_rate_precision(recommendations_from_valid_graph, baseline_model, t))
    hit_rates_prec_random.append(hit_rate_precision(recommendations_from_valid_graph, random_model, t))
    hit_rates_prec_model.append(hit_rate_precision(recommendations_from_valid_graph, model_recommendations, t))
    hit_rates_recall_baseline.append(hit_rate_recall(recommendations_from_valid_graph, baseline_model, t))
    hit_rates_recall_random.append(hit_rate_recall(recommendations_from_valid_graph, random_model, t))
    hit_rates_recall_model.append(hit_rate_recall(recommendations_from_valid_graph, model_recommendations, t))
    
fig = plt.figure()
plt.plot(thresholds,hit_rates_prec_baseline, label = "Popularity Model")
plt.plot(thresholds,hit_rates_prec_random, label = "Random Model")
plt.plot(thresholds,hit_rates_prec_model, label = "GNNIE Model")
plt.legend()
plt.xlabel("# Recs per Customer")
plt.ylabel("Hit Rate")
plt.title(f"Hit Rate Precision Performance: {graph_details}")
# fig.savefig(f'{BASE_DIR}/{MODEL_DIR}/hit_rate_precision.png', dpi=fig.dpi)
fig.savefig(os.path.join(new_directory, 'hit_rate_precision.png'), dpi=fig.dpi)

fig = plt.figure()
plt.plot(thresholds,hit_rates_recall_baseline, label = "Popularity Model")
plt.plot(thresholds,hit_rates_recall_random, label = "Random Model")
plt.plot(thresholds,hit_rates_recall_model, label = "GNNIE Model")
plt.legend()
plt.xlabel("# Recs per Customer")
plt.ylabel("Hit Rate")
plt.title(f"Hit Rate Recall Performance : {graph_details}")
# fig.savefig(f'{BASE_DIR}/{MODEL_DIR}/hit_rate_recall.png', dpi=fig.dpi)
fig.savefig(os.path.join(new_directory, 'hit_rate_recall.png'), dpi=fig.dpi)

print("Hit Rate precision", file=op_file)
print("pop =",hit_rates_prec_baseline, file=op_file)
print("random =",hit_rates_prec_random, file=op_file)
print("model =",hit_rates_prec_model, file=op_file)

print("Hit Rate recall", file=op_file)
print("pop =",hit_rates_recall_baseline, file=op_file)
print("random =",hit_rates_recall_random, file=op_file)
print("model =",hit_rates_recall_model, file=op_file)

# #RBO

# # Popular vs. Random
rbo_scores_5 = []
rbo_scores_15 = []
rbo_scores_30 = []

recs_sample = random.sample(list(model_recommendations.items()), 2000)

for i in range(0,len(recs_sample)):
    rbo_scores_5.append(rbo(baseline_model[0][0:100],random.sample(list(random_model[0]), 100),0.76))
    rbo_scores_15.append(rbo(baseline_model[0][0:100],random.sample(list(random_model[0]), 100),0.9165))
    rbo_scores_30.append(rbo(baseline_model[0][0:100],random.sample(list(random_model[0]), 100),0.9578))
    
print("Random-Popularity RBO giving 90% weight to first 5 recs: ", sum(rbo_scores_5) / len(rbo_scores_5), file=op_file)
print("Random-Popularity RBO giving 90% weight to first 15 recs: ", sum(rbo_scores_15) / len(rbo_scores_15), file=op_file)
print("Random-Popularity RBO giving 90% weight to first 30 recs: ", sum(rbo_scores_30) / len(rbo_scores_30), file=op_file)


# # Popular vs. GNNIE
rbo_scores_5 = []
rbo_scores_15 = []
rbo_scores_30 = []

recs_sample = random.sample(list(model_recommendations.items()), 2000)

for i in range(0,len(recs_sample)):
    rbo_scores_5.append(rbo(baseline_model[0][0:100],recs_sample[i][1][0:100],0.76))
    rbo_scores_15.append(rbo(baseline_model[0][0:100],recs_sample[i][1][0:100],0.9165))
    rbo_scores_30.append(rbo(baseline_model[0][0:100],recs_sample[i][1][0:100],0.9578))

print("GNNIE-Popularity RBO giving 90% weight to first 5 recs: ", sum(rbo_scores_5) / len(rbo_scores_5), file=op_file)
print("GNNIE-Popularity RBO giving 90% weight to first 15 recs: ", sum(rbo_scores_15) / len(rbo_scores_15), file=op_file)
print("GNNIE-Popularity RBO giving 90% weight to first 30 recs: ", sum(rbo_scores_30) / len(rbo_scores_30), file=op_file)


op_file.close()

print("Finished run!")
