import dgl
import torch, os
import pandas as pd
import numpy as np
from settings import BASE_DIR, MODEL_DIR
from validate_graph import validate_customer_features, validate_product_features, validate_edges,validate_edge_features

import argparse
parser = argparse.ArgumentParser(description='Parse arguments for graph creation')
# parser.add_argument("--remove_duplicates", default=False, help="Remove duplicate product id and customer id combination", type=bool)
parser.add_argument('--remove_duplicates', action='store_true', help="Remove duplicate product id and customer id combination")

args = parser.parse_args()

remove_duplicates = args.remove_duplicates

# BASE_DIR_ARCHIVE = 'DatasetEDA/archive'

# Get the current directory
full_path = os.path.realpath(__file__)
current_dir, filename = os.path.split(full_path)
UNIFIED_CSV_PATH = 'graph_files/unified.csv'
BASE_DIR_ARCHIVE_PATH = 'graph_files/archive'

BASE_DIR_ARCHIVE = os.path.join(current_dir, BASE_DIR_ARCHIVE_PATH)

customer_data = pd.read_csv(f'{BASE_DIR_ARCHIVE}/olist_customers_dataset.csv')
geolocation_data = pd.read_csv(f'{BASE_DIR_ARCHIVE}/olist_geolocation_dataset.csv')
order_items_dataset = pd.read_csv(f'{BASE_DIR_ARCHIVE}/olist_order_items_dataset.csv')
order_payments_dataset = pd.read_csv(f'{BASE_DIR_ARCHIVE}/olist_order_payments_dataset.csv')
order_reviews_dataset = pd.read_csv(f'{BASE_DIR_ARCHIVE}/olist_order_reviews_dataset.csv')
order_dataset = pd.read_csv(f'{BASE_DIR_ARCHIVE}/olist_orders_dataset.csv')
order_products_dataset = pd.read_csv(f'{BASE_DIR_ARCHIVE}/olist_products_dataset.csv')
order_sellers_dataset = pd.read_csv(f'{BASE_DIR_ARCHIVE}/olist_sellers_dataset.csv')
product_translation_dataset = pd.read_csv(f'{BASE_DIR_ARCHIVE}/product_category_name_translation.csv')

order_reviews_dataset = order_reviews_dataset[['order_id','review_score', 'review_comment_message']]
order_review_data = order_reviews_dataset.merge(order_dataset,on='order_id')
order_products_dataset_english = pd.merge(order_products_dataset,product_translation_dataset,on='product_category_name', how='left')
order_products_dataset_english = order_products_dataset_english.drop(labels='product_category_name',axis=1)
order_product_item_dataset = pd.merge(order_items_dataset,order_products_dataset_english,on='product_id')
ordered_product_reviews = pd.merge(order_product_item_dataset,order_review_data,on='order_id')
ordered_product_reviews_payments = pd.merge(ordered_product_reviews,order_payments_dataset,on='order_id')
df_final = pd.merge(ordered_product_reviews_payments,customer_data,on='customer_id')

df_final = pd.read_csv(os.path.join(current_dir, UNIFIED_CSV_PATH))


# Handling missing values
df_final['product_name_lenght'].fillna(df_final['product_name_lenght'].median(),inplace=True)
df_final['product_description_lenght'].fillna(df_final['product_description_lenght'].median(),inplace=True)
df_final['product_photos_qty'].fillna(df_final['product_photos_qty'].median(),inplace=True)
df_final['product_weight_g'].fillna(df_final['product_weight_g'].median(),inplace=True)
df_final['product_length_cm'].fillna(df_final['product_length_cm'].median(),inplace=True)
df_final['product_height_cm'].fillna(df_final['product_height_cm'].median(),inplace=True)
df_final['product_width_cm'].fillna(df_final['product_width_cm'].median(),inplace=True)


#Handling missing values
ids = (df_final[df_final['order_delivered_customer_date'].isnull() == True].index.values)
vals = df_final.iloc[ids]['order_estimated_delivery_date'].values
df_final.loc[ids,'order_delivered_customer_date'] = vals

ids = (df_final[df_final['order_approved_at'].isnull() == True].index.values)
df_final.loc[ids,'order_approved_at'] = df_final.iloc[ids]['order_purchase_timestamp'].values

df_final.drop(labels='order_delivered_carrier_date',axis=1,inplace=True)




product_id = order_product_item_dataset.groupby('product_id').count()['seller_id'].index
seller_count = order_product_item_dataset.groupby('product_id').count()['seller_id'].values
product_seller_count = pd.DataFrame({'product_id':product_id,'sellers_count':seller_count})

order_id = order_product_item_dataset.groupby('order_id').count()['product_id'].index
pd_count = order_product_item_dataset.groupby('order_id').count()['product_id'].values
order_items_count = pd.DataFrame({'order_id':order_id,'products_count':pd_count})


# Adding the seller count and products count feature to the final data set
df_final = pd.merge(df_final,product_seller_count,on='product_id')
df_final = pd.merge(df_final,order_items_count,on='order_id')

# converting date to datetime and extracting dates from the datetime columns in the data set
datetime_cols = ['order_purchase_timestamp','order_approved_at','order_delivered_customer_date','order_estimated_delivery_date']
for col in datetime_cols:
    df_final[col] = pd.to_datetime(df_final[col]).dt.date


# calculating estimated delivery time
df_final['estimated_delivery_time'] = (df_final['order_estimated_delivery_date'] - df_final['order_approved_at']).dt.days

# calculating actual delivery time
df_final['actual_delivery_time'] = (df_final['order_delivered_customer_date'] - df_final['order_approved_at']).dt.days

# calculating diff_in_delivery_time
df_final['diff_in_delivery_time'] = df_final['estimated_delivery_time'] - df_final['actual_delivery_time']

# finding if delivery was late
df_final['on_time_delivery'] = df_final['order_delivered_customer_date'] < df_final['order_estimated_delivery_date']
df_final['on_time_delivery'] = df_final['on_time_delivery'].astype('int')

# calculating mean product value
df_final['avg_product_value'] = df_final['price']/df_final['products_count']

# finding total order cost
df_final['total_order_cost'] = df_final['price'] + df_final['freight_value']

# calculating order freight ratio
df_final['order_freight_ratio'] = df_final['freight_value']/df_final['price']

# finding the day of week on which order was made
df_final['purchase_dayofweek'] = pd.to_datetime(df_final['order_purchase_timestamp']).dt.dayofweek

# finding the day of month on which order was made
df_final['purchase_dayofmonth'] = pd.to_datetime(df_final['order_purchase_timestamp']).dt.day

# finding the week of month on which order was made
df_final['purchase_weekofmonth'] = df_final['purchase_dayofmonth']//7

# finding the week year on which order was made
df_final['purchase_weekofyear'] = pd.to_datetime(df_final['order_purchase_timestamp']).dt.isocalendar().week

# adding is_reviewed where 1 is if review comment is given otherwise 0.
df_final['is_reviewed'] = (df_final['review_comment_message'] != 'no_review').astype('int')

### modified from GNN_Architecture/generate_graph/create_graph_using_features.py ###
# drop duplicates

if remove_duplicates:

    print("Removing duplicates of customer and product id combinations...")

    df_final = df_final.sample(frac=1, random_state=42)
    df_final.reset_index(drop=True, inplace=True)

    df_final.drop_duplicates(subset=['product_id', 'customer_id'], keep='first', inplace=True)
    # reset the index
    df_final.reset_index(drop=True, inplace=True)

### 

import pandas as pd
import uuid

# Create a dictionary that maps UUIDs to integer IDs
uuid_to_int = {}
for i, u in enumerate(df_final['customer_id'].unique()):
    uuid_to_int[u] = i

# Map the UUIDs to integer IDs using the dictionary
df_final['customer_id_int'] = df_final['customer_id'].map(uuid_to_int)

import pandas as pd
import uuid

# Create a dictionary that maps UUIDs to integer IDs
uuid_to_int = {}
for i, u in enumerate(df_final['product_id'].unique()):
    uuid_to_int[u] = i

# Map the UUIDs to integer IDs using the dictionary
df_final['product_id_int'] = df_final['product_id'].map(uuid_to_int)


graph_data = {
        ('customer','orders','product') : (df_final['customer_id_int'].to_numpy(), df_final['product_id_int'].to_numpy()),
        ('product','rev-orders','customer') : (df_final['product_id_int'].to_numpy(), df_final['customer_id_int'].to_numpy())
    }


ecommerce_hetero_graph = dgl.heterograph(graph_data)

def _process_customer_features(df_final):
    
    HM = {}
    for _, row in df_final.iterrows():   
        HM[row['customer_id_int']] = torch.tensor([row['customer_zip_code_prefix']]).float()
    
    return HM

custid_to_feat = _process_customer_features(df_final)
customer_features = [value[1] for value in list(custid_to_feat.items())]

ecommerce_hetero_graph.nodes['customer'].data['features'] = torch.stack(customer_features, axis=0)

def _process_product_features(df_final):


    HM = {}
    for _, row in df_final.iterrows():
        # HM[row['product_id_int']] = torch.tensor([round(row['price'], 4), row['purchase_weekofyear']])
        HM[row['product_id_int']] = torch.tensor([row['product_weight_g'], row['product_photos_qty']])
    return HM

# print(df_final.columns)

prodid_to_feat = _process_product_features(df_final)

product_features = [value[1] for value in list(prodid_to_feat.items())]
ecommerce_hetero_graph.nodes['product'].data['features'] = torch.stack(product_features, axis=0)    

# edge feature assignment
edge_features = [[float(x), float(y)] for x, y in zip(df_final['price'], df_final['purchase_weekofyear'])]
print(edge_features[:5])
ecommerce_hetero_graph.edges['orders'].data['features']= torch.tensor(edge_features)
ecommerce_hetero_graph.edges['rev-orders'].data['features']= torch.tensor(edge_features)
# REMOVED .unsqueeze(-1) from torch.tensor(edge_features).unsqueeze(-1) as it was giving error

# run validation scripts
# TODO: error handling in case validation scripts return errors
# print(validate_customer_features(ecommerce_hetero_graph, df_final, ['customer_zip_code_prefix']))
# print(validate_product_features(ecommerce_hetero_graph, df_final, ['price', 'purchase_weekofyear']))
# print(validate_edges(ecommerce_hetero_graph, df_final, 'orders', 'customer_id_int', 'product_id_int'))
# print(validate_edges(ecommerce_hetero_graph, df_final, 'rev-orders', 'product_id_int', 'customer_id_int'))
# print(validate_edge_features(ecommerce_hetero_graph, df_final, 'orders', ['is_reviewed']))

print(ecommerce_hetero_graph.nodes['product'].data['features'].shape, ecommerce_hetero_graph.nodes['customer'].data['features'].shape)
print(ecommerce_hetero_graph.edges['orders'].data['features'].shape)

print(f"SAVED GRAPH AT LOCATION : {BASE_DIR}/{MODEL_DIR}/ecommerce_hetero_graph.dgl")
dgl.save_graphs(f"{BASE_DIR}/{MODEL_DIR}/ecommerce_hetero_graph.dgl", [ecommerce_hetero_graph])











