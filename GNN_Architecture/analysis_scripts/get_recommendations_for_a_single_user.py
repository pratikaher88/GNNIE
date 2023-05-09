import pickle
from evaluation import recommendations_combined_with_popularity

popularity_baseline = pickle.load(open("run_data/graph_files_subgraph/popularity_baseline.pickle", "rb"))
# print(len(popularity_baseline[1]))
gnn_recs = pickle.load(open("run_data/graph_files_subgraph/model_recommendations.pickle", "rb"))
# print(len(gnn_recs[1]))


uuid_customer_mapping = pickle.load(open("run_data/graph_files_subgraph/customer_uuid_to_int.pickle", "rb"))
uuid_product_mapping = pickle.load(open("run_data/graph_files_subgraph/product_uuid_to_int.pickle", "rb"))

# for key, value in uuid_customer_mapping.items():
#     print(value, key)
#     if value > 10000:
#         break

def recommend(user_id, delta, epislon, num_recs = 100):

    # # get products with delta and epislon
    # combined_recs = recommendations_combined_with_popularity(gnn_recs, popularity_baseline, delta, epislon)
    # combined_recs = combined_recs[:100]

    # print(type(gnn_recs))
    customer = uuid_customer_mapping[user_id]
    print(len(gnn_recs))

    gnn_recs_list = gnn_recs[customer][:num_recs]
    popularity_baseline_recs = popularity_baseline[customer][:num_recs]
    combined_recs = recommendations_combined_with_popularity.get_recommendations_combined_with_popularity(gnn_recs_list, popularity_baseline_recs, delta, epislon)
    # combined_recs = combined_recs[:num_recs]

    # customer = uuid_customer_mapping[user_id]
    product_list = []

    for rec in combined_recs:
        product_list.append(uuid_product_mapping[rec])
        # product_list.append(rec)

    print(customer, product_list)
    # print(customer)

sample_user_id = "1b7c152aba1c7478309329cfec0bb5d1"
recommend(sample_user_id, 1, 10)
    
    


