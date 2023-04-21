import dgl
import torch
import pandas as pd
import numpy as np
# from settings import BASE_DIR
# from validate_graph import validate_customer_features, validate_product_features, validate_edges
import uuid


BASE_DIR = 'generate_graph/graph_files' # set this up before running this file

# import the whole set of data features
df_final = pd.read_csv(BASE_DIR+"/features.csv")


# create list of features that belongs to different type of nodes
list_of_features = list(df_final.columns)

list_of_user_features = [f for f in list_of_features if "id" not in f and "user" in f]
len(list_of_user_features), list_of_user_features

list_of_product_features = [f for f in list_of_features if "id" not in f and "product" in f]
len(list_of_product_features), list_of_product_features

list_of_order_features = [f for f in list_of_features if "id" not in f and "_order" in f]
len(list_of_order_features), list_of_order_features

list_of_revorder_features = [f for f in list_of_features if "id" not in f and "_revorder" in f]
len(list_of_revorder_features), list_of_revorder_features
# sanity check
# set(list_of_features) - set(list_of_user_features) - set(list_of_product_features) - set(list_of_order_features) - set(list_of_revorder_features)


### generate integer node ids ###

# Create a dictionary that maps UUIDs to integer IDs
uuid_to_int = {}
for i, u in enumerate(df_final['customer_id'].unique()):
    uuid_to_int[u] = i

# Map the UUIDs to integer IDs using the dictionary
df_final['customer_id_int'] = df_final['customer_id'].map(uuid_to_int)

# repaet the process for product id
uuid_to_int = {}
for i, u in enumerate(df_final['product_id'].unique()):
    uuid_to_int[u] = i

df_final['product_id_int'] = df_final['product_id'].map(uuid_to_int)



### generate integer edge ids ###

# in the graph, order ids should be the indicator of "pairs between user and product id"
df_final['new_order_id'] = df_final.customer_id.str.cat(df_final.product_id)
# concatenate user and prod node ids so that df_final.new_order_id.nunique() matches ecommerce_hetero_graph.num_edges
# df_final.new_order_id.nunique()

# repaet the process for node id
uuid_to_int = {}
for i, u in enumerate(df_final['new_order_id'].unique()):
    uuid_to_int[u] = i

df_final['order_id_int'] = df_final['new_order_id'].map(uuid_to_int)


### create graph ###

graph_data = {
        ('customer','orders','product') : (df_final['customer_id_int'].to_numpy(), df_final['product_id_int'].to_numpy()),
        ('product','rev-orders','customer') : (df_final['product_id_int'].to_numpy(), df_final['customer_id_int'].to_numpy())
    }
ecommerce_hetero_graph = dgl.heterograph(graph_data)


### node feature assignment ###

def _process_customer_features(df_final):
    HM = {}
    for _, row in df_final.iterrows():
        HM[row['customer_id_int']] = torch.tensor([row[list_of_user_features]]).float()
    return HM

custid_to_feat = _process_customer_features(df_final)
customer_features = [value[1].squeeze() for value in list(custid_to_feat.items())]


ecommerce_hetero_graph.nodes['customer'].data['features'] = torch.stack(customer_features, 0)

def _process_product_features(df_final):
    HM = {}
    for _, row in df_final.iterrows():
        HM[row['product_id_int']] = torch.tensor([row[list_of_product_features]])
    return HM
prodid_to_feat = _process_product_features(df_final)
product_features = [value[1].squeeze() for value in list(prodid_to_feat.items())]
ecommerce_hetero_graph.nodes['product'].data['features'] = torch.stack(product_features)


### edge feature assignment ###

def _process_order_features(df_final):
    HM = {}
    for _, row in df_final.iterrows():
        HM[row['order_id_int']] = torch.tensor([row[list_of_order_features]])
    return HM

orderid_to_feat = _process_order_features(df_final)
order_features = [value[1].squeeze() for value in list(orderid_to_feat.items())]
ecommerce_hetero_graph.edges['orders'].data['features'] = torch.stack(order_features, axis=0).unsqueeze(-1)

# print(ecommerce_hetero_graph.edges['orders'].data['features'].shape)

def _process_revorder_features(df_final):
    HM = {}
    for _, row in df_final.iterrows():
        HM[row['order_id_int']] = torch.tensor([row[list_of_revorder_features]])
    return HM

orderid_to_feat = _process_revorder_features(df_final)
revorder_features = [value[1] for value in list(orderid_to_feat.items())]
ecommerce_hetero_graph.edges['rev-orders'].data['features'] = torch.stack(revorder_features, axis=0).unsqueeze(-1)

### graph validation is skipped for now because it's not generalized yet ###

# # run validation scripts
# # TODO: error handling in case validation scripts return errors
# print(validate_customer_features(ecommerce_hetero_graph, df_final))
# print(validate_product_features(ecommerce_hetero_graph, df_final))
# print(validate_edges(ecommerce_hetero_graph, df_final, 'orders', 'customer_id_int', 'product_id_int'))
# print(validate_edges(ecommerce_hetero_graph, df_final, 'rev-orders', 'product_id_int', 'customer_id_int'))

print(ecommerce_hetero_graph.nodes['product'].data['features'].shape)
print(ecommerce_hetero_graph.edges['orders'].data['features'].shape)

SAVE_DIR = 'run_data/graph_files_subgraph/'
print("SAVE GRAPH !!")
dgl.save_graphs(f"{SAVE_DIR}/ecommerce_hetero_graph_with_features.dgl", [ecommerce_hetero_graph])
