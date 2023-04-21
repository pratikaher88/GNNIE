#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sys
import os
import pandas as pd
import numpy as np
import uuid
from datetime import timedelta


# In[13]:


# CONFIGS
UNIFIED_CSV_PATH = '/Users/pratikaher/SPRING23/Capstone/DatasetEDA/archive/unified.csv'
BASE_DIR = '/Users/pratikaher/SPRING23/Capstone/go_prod'

USER_NODE = "customer_id"
PRODUCT_NODE = "product_id"
USER_FEATURES = [
    "gnn_user_customer_state",
    "gnn_user_reviews_average",
    "gnn_user_reviews_last",
    "gnn_user_reviews_last3_average",
    #"gnn_user_zip_code_5",
    #"gnn_user_zip_code_4",
    #"gnn_user_zip_code_3",
    #"gnn_user_zip_code_2",
    "gnn_user_zip_code_1",
]
PRODUCT_FEATURES = [    
    "gnn_product_reviews_average",
    "gnn_product_reviews_last",
    "gnn_product_reviews_last3_average",
    "gnn_product_name_lenght",
    "gnn_product_description_lenght",
    "gnn_product_category_code",
    "gnn_product_photos_qty",
    "gnn_product_weight_g",
    "gnn_product_volume_cm3",
    "gnn_product_seller_count",
    "gnn_product_avg_price",
    "gnn_product_quantity",
]
ORDER_EDGE = "order_id"
ORDER_FEATURES = [ 
    "gnn_order_items_count",
    "gnn_order_time_purchased_to_approved",
    "gnn_order_time_approved_to_carrier",
    "gnn_order_time_carrier_to_customer",
    "gnn_order_time_customer_to_eta",
    # => shipping_limit_date missing
    "gnn_order_is_on_time_delivery",
    "gnn_order_total_amount",
    "gnn_order_item_count",
    "gnn_order_freight_value",
    "gnn_order_freight_ratio",
    "gnn_order_purchase_dayofweek",
    "gnn_order_purchase_dayofmonth",
    "gnn_order_purchase_weekofmonth",
    "gnn_order_purchase_weekofyear"
    "gnn_order_is_holiday", #at time of purchase
    "gnn_order_days_to_next_holidays",
    "gnn_order_delivery_state_code",
    "gnn_order_delivery_zip_code_5",
    "gnn_order_delivery_zip_code_4",
    "gnn_order_delivery_zip_code_3",
    "gnn_order_delivery_zip_code_2",
    "gnn_order_delivery_zip_code_1",
    "gnn_order_weight_g",
    #"gnn_order_payment_method_code",
    #"gnn_order_has_installments",
    #"gnn_order_installments",
]
REV_ORDER_FEATURES = [
    "gnn_revorder_is_reviewed",
    "gnn_revorder_has_review_comment",
    "gnn_revorder_review_score",
    "gnn_revorder_is_returned",
    "gnn_revorder_return_reason_code",
]

RAW_COLUMNS_REQUIREMENTS = [
    ORDER_EDGE, 
    USER_NODE, 
    PRODUCT_NODE,
    'order_item_id',  #Remove when derived df function is partially moved to merge_data.py
    'seller_id',
    'shipping_limit_date', 
    'price', 
    'freight_value', 
    'product_name_lenght',
    'product_description_lenght', 
    'product_photos_qty', 
    'product_weight_g',
    'product_length_cm', 
    'product_height_cm', 
    'product_width_cm',
    'product_category_name_english', 
    'review_score',
    'review_comment_message', 
    'customer_id', 
    'order_status',
    'order_purchase_timestamp', 
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date', 
    'customer_zip_code_prefix', 
    'customer_city', 
    'customer_state',
]


# In[19]:


def check_columns(df):
    """
    Check if all the specified columns exist in the data. Return dataframe that contains only the selected columns.
    :return: boolean
    """
    
    print("Checking Columns...")
    exist_columns = df.columns.tolist()

    for feature in RAW_COLUMNS_REQUIREMENTS:
        if not feature in exist_columns:
            raise AttributeError(
                'Specified feature `{}` is not in the uploaded data'.format(
                    feature
                )
            )

    return df[set(RAW_COLUMNS_REQUIREMENTS)]


def remove_duplicate(df):
    """
    Remove deplicate records. Report duplicate percentage when it's higher than 20%.
    :return: deduped dataframe
    """
    
    print("Removing Duplicates...")

    N = df.shape[0]
    df.drop_duplicates(keep='last', inplace=True)
    N_deduped = df.shape[0]
    dup_pct = (((N - N_deduped) / N) * 100) // 1

    if not dup_pct <= 20:
        print(
            'Warning: Duplicate records take up more than 20% of the uploaded data. Duplicate percentage: {}'.format(
                dup_pct
            )
        )

    return df


def handle_nulls(df):
    """
    Stategies:
    * user_node and product_node: remove all null records.
    * Numerical features: (1) drop the column if null_pct >=10%. (2) impute using median if 3%< null_pct <10%. (3) remove the record if null_pct <=3%.
    * Categorical features: (1) drop the column if null_pct >=5%. (2) remove the record if null_pct <5%.
    :return: dataframe
    """
    
    print("Handling Nulls...")

    df.dropna(subset=[USER_NODE, PRODUCT_NODE], inplace=True)

    remove_columns = []
    impute_median_columns = []

    categorical_list = list(
        set(df.columns) - set(df.select_dtypes(include=np.number).columns)
    )
    categorical_null_pct = df[categorical_list].isna().sum() / df.shape[0]
    remove_columns.extend(
        categorical_null_pct[categorical_null_pct >= 0.05].index.tolist()
    )

    numerical_null_pct = (
        df.select_dtypes(include=np.number).isna().sum() / df.shape[0]
    )
    remove_columns.extend(
        numerical_null_pct[numerical_null_pct >= 0.1].index.tolist()
    )
    impute_median_columns.extend(
        numerical_null_pct[
            (numerical_null_pct < 0.1) & (numerical_null_pct >= 0.03)
        ].index.tolist()
    )

    for feature in impute_median_columns:  # impute columns
        df[feature].fillna(df[feature].median(), inplace=True)
    df.drop(columns=remove_columns, inplace=True)  # drop columns
    df.dropna(
        inplace=True
    )  # drop all the records that still contains null value

    return df


# In[7]:


def assign_node_id(df, user_node, product_node):
    """
    Assign node id as a continuous integer series starting from 0.
    """
    print("Assigning Node IDs...")

    # Create a dictionary that maps UUIDs to integer IDs
    uuid_to_int = {}
    for i, u in enumerate(df[user_node].unique()):
        uuid_to_int[u] = i
    # Map the UUIDs to integer IDs using the dictionary
    df['node_id_user'] = df[user_node].map(uuid_to_int)

    uuid_to_int = {}
    for i, u in enumerate(df[product_node].unique()):
        uuid_to_int[u] = i
    df['node_id_product'] = df[product_node].map(uuid_to_int)

    return df


# In[8]:


def one_hot_encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res) 


# In[9]:


def derive_dataframes(df):
    print("Deriving Orders Dataframe...")

    datetime_cols = ['order_purchase_timestamp','order_delivered_carrier_date','order_approved_at','order_delivered_customer_date','order_estimated_delivery_date','shipping_limit_date']
    df[datetime_cols] = df[datetime_cols].astype(str)
    df[datetime_cols] = df[datetime_cols].apply(pd.to_datetime)
    
    df = df.sort_values(by=['order_purchase_timestamp'], ignore_index = True)
    
    df_order_items = df.groupby(df.columns.difference(['order_item_id','review_score']).tolist(),as_index=False).agg({
                         'order_item_id': 'count',
                         'review_score': 'mean'}).rename(columns={'order_item_id': 'quantity'})
    
    assert len(df_order_items.groupby(['order_id','product_id']).filter(lambda x: len(x) > 1)) == 0 , "Dataset does not contain one row per order-item"
    
    df_orders = df_order_items.groupby(df_order_items.columns.difference(['price','freight_value','product_weight_g','review_score','quantity','product_category_name_english','product_description_lenght','product_name_lenght','product_height_cm','product_length_cm','product_width_cm','seller_id','shipping_limit_date','customer_unique_id','product_id','product_photos_qty']).tolist(),as_index=False).agg({
                         'price': 'sum', 
                         'freight_value':'sum', 
                         'product_weight_g':'sum', 
                         'review_score':'mean',
                         'quantity': 'sum'})
    
    assert len(df_orders.groupby("order_id").filter(lambda x: len(x) > 1)) == 0 , "Derived order dataset does not contain one row per order"
    
    return df_order_items, df_orders
    


# In[22]:


def feature_engineering(df):
    """
    Extract features.
    :return: dataframe
    """
    print("Extracting Features...")

    df_order_items, df_orders = derive_dataframes(df)
    
##### USER_FEATURES
    print("Extracting USER Features...")

    if "gnn_user_customer_state" in USER_FEATURES:
        df_orders.rename(columns={'customer_state': 'gnn_user_customer_state'},inplace=True)
        df_orders = one_hot_encode_and_bind(df_orders,"gnn_user_customer_state")
      
    if "gnn_user_reviews_average" in USER_FEATURES:
        df_orders['gnn_user_reviews_average'] = df_orders.groupby('customer_id')['review_score'].transform('mean')
     
    if "gnn_user_reviews_last" in USER_FEATURES:
        df_orders['gnn_user_reviews_last'] = df_orders.groupby('customer_id')['review_score'].transform('last')
    
    if "gnn_user_reviews_last3_average" in USER_FEATURES:
        df_orders['gnn_user_reviews_last3_average'] = df_orders.groupby('customer_id')['review_score'].transform(lambda x: x.rolling(3, 1).mean())

    if "gnn_user_zip_code_5" in USER_FEATURES:
        df_orders['gnn_user_zip_code_5'] = df_orders['customer_zip_code_prefix']
        df_orders = one_hot_encode_and_bind(df_orders,"gnn_user_zip_code_5")
   
    if "gnn_user_zip_code_4" in USER_FEATURES:
        df_orders['gnn_user_zip_code_4'] = df_orders['customer_zip_code_prefix'].map(lambda x: str(x)[:-1])
        df_orders = one_hot_encode_and_bind(df_orders,"gnn_user_zip_code_4")

    if "gnn_user_zip_code_3" in USER_FEATURES:
        df_orders['gnn_user_zip_code_3'] = df_orders['customer_zip_code_prefix'].map(lambda x: str(x)[:-2])
        df_orders = one_hot_encode_and_bind(df_orders,"gnn_user_zip_code_3")
    
    if "gnn_user_zip_code_2" in USER_FEATURES:
        df_orders['gnn_user_zip_code_2'] = df_orders['customer_zip_code_prefix'].map(lambda x: str(x)[:-3])
        df_orders = one_hot_encode_and_bind(df_orders,"gnn_user_zip_code_2")

    if "gnn_user_zip_code_1" in USER_FEATURES:
        df_orders['gnn_user_zip_code_1'] = df_orders['customer_zip_code_prefix'].map(lambda x: str(x)[:-4])
        df_orders = one_hot_encode_and_bind(df_orders,"gnn_user_zip_code_1")

        
##### PRODUCT_FEATURES
    print("Extracting PRODUCT Features...")

    if "gnn_product_reviews_average" in PRODUCT_FEATURES:
        df_order_items['gnn_product_reviews_last'] = df_order_items.groupby('product_id')['review_score'].transform('mean')
    
    if "gnn_product_reviews_last" in PRODUCT_FEATURES:
        df_order_items['gnn_product_reviews_last'] = df_order_items.groupby('product_id')['review_score'].transform('last')
     
    if "gnn_product_reviews_last3_average" in PRODUCT_FEATURES:
        df_order_items['gnn_product_reviews_last3_average'] = df_order_items.groupby('product_id')['review_score'].transform(lambda x: x.rolling(3, 1).mean())
        
    if "gnn_product_seller_count" in PRODUCT_FEATURES:
        df_order_items['gnn_product_seller_count'] = df_order_items.groupby('product_id')['seller_id'].transform('nunique')
       
    if "gnn_product_avg_price" in PRODUCT_FEATURES:
        df_order_items['gnn_product_avg_price'] = df_order_items.groupby('product_id')['price'].transform('mean')
    
    if "gnn_product_volume_cm3" in PRODUCT_FEATURES:
        df_order_items['gnn_product_volume_cm3'] = df_order_items['product_height_cm']*df_order_items['product_length_cm']*df_order_items['product_width_cm']

    if "gnn_product_name_lenght" in PRODUCT_FEATURES:
        df_order_items.rename(columns={'product_name_lenght': 'gnn_product_name_lenght'},inplace=True)

    if "gnn_product_description_lenght" in PRODUCT_FEATURES:
        df_order_items.rename(columns={'product_description_lenght': 'gnn_product_description_lenght'},inplace=True)
    
    if "gnn_product_category_code" in PRODUCT_FEATURES:
        df_order_items.rename(columns={'product_category_name_english': 'gnn_product_category_code'},inplace=True)
        df_order_items = one_hot_encode_and_bind(df_order_items,"gnn_product_category_code")

    if "gnn_product_photos_qty" in PRODUCT_FEATURES:
        df_order_items.rename(columns={'product_photos_qty': 'gnn_product_photos_qty'},inplace=True)

    if "gnn_product_weight_g" in PRODUCT_FEATURES:
        df_order_items.rename(columns={'product_weight_g': 'gnn_product_weight_g'},inplace=True)
 
    if "gnn_product_quantity" in PRODUCT_FEATURES:
            df_order_items.rename(columns={'quantity': 'gnn_product_quantity'},inplace=True)


##### ORDER_FEATURES
    print("Extracting ORDER Features...")

    if "gnn_order_items_count" in ORDER_FEATURES:
        df_orders.rename(columns={"quantity": "gnn_order_items_count"},inplace=True)
    
    if "gnn_order_time_purchased_to_approved" in ORDER_FEATURES:
        df_orders['gnn_order_time_purchased_to_approved'] = df_orders['order_approved_at'] - df_orders['order_purchase_timestamp']
        df_orders['gnn_order_time_purchased_to_approved'] = df_orders['gnn_order_time_purchased_to_approved'].dt.total_seconds()/60
    
    if "gnn_order_time_approved_to_carrier" in ORDER_FEATURES:
        df_orders['gnn_order_time_approved_to_carrier'] = df_orders['order_delivered_carrier_date'] - df_orders['order_approved_at']
        df_orders['gnn_order_time_approved_to_carrier'] = df_orders['gnn_order_time_approved_to_carrier'].dt.total_seconds()/60
    
    if "gnn_order_time_carrier_to_customer" in ORDER_FEATURES:
        df_orders['gnn_order_time_carrier_to_customer'] = df_orders['order_delivered_customer_date'] - df_orders['order_delivered_carrier_date']
        df_orders['gnn_order_time_carrier_to_customer'] = df_orders['gnn_order_time_carrier_to_customer'].dt.total_seconds()/60
          
    if "gnn_order_time_customer_to_eta" in ORDER_FEATURES:
        df_orders['gnn_order_time_customer_to_eta'] = df_orders['order_estimated_delivery_date'] + timedelta(days=1) - df_orders['order_delivered_customer_date']
        df_orders['gnn_order_time_customer_to_eta'] = df_orders['gnn_order_time_customer_to_eta'].dt.total_seconds()/60
        
    if "gnn_order_is_on_time_delivery" in ORDER_FEATURES:
        df_orders['gnn_order_is_on_time_delivery'] = np.where(df_orders['gnn_order_time_customer_to_eta'] >= 0, 1, 0)

    if "gnn_order_total_amount" in ORDER_FEATURES:
        df_orders.rename(columns={'price': 'gnn_order_total_amount'},inplace=True)

    if "gnn_order_freight_value" in ORDER_FEATURES:
        df_orders.rename(columns={'freight_value': 'gnn_order_freight_value'},inplace=True)
    
    if "gnn_order_weight_g" in ORDER_FEATURES:
        df_orders.rename(columns={'product_weight_g': 'gnn_order_weight_g'},inplace=True)
    
    if "gnn_order_item_count" in ORDER_FEATURES:
        df_orders.rename(columns={'quantity': 'gnn_order_item_count'},inplace=True)

    
##### REV_ORDER_FEATURES
    print("Extracting REV-ORDER Features...")
   
    if "gnn_revorder_review_score" in REV_ORDER_FEATURES:
        df_orders.rename(columns={'review_score': 'gnn_revorder_review_score'},inplace=True)
        
##### MERGE & CLEAN DATAFRAMES
    print("Cleaning & Merging Features...")

    #Rename ID columns so featured df is characterized by "gnn_" column names
    #df_order_items.rename(columns={USER_NODE: 'gnn_'+USER_NODE, PRODUCT_NODE: 'gnn_'+PRODUCT_NODE, ORDER_EDGE: 'gnn_'+ORDER_EDGE},inplace=True)
    #df_orders.rename(columns={USER_NODE: 'gnn_'+USER_NODE, PRODUCT_NODE: 'gnn_'+PRODUCT_NODE, ORDER_EDGE: 'gnn_'+ORDER_EDGE},inplace=True)

    #Drop all columns that are not features
    df_order_items = df_order_items.loc[:, df_order_items.columns.str.startswith(("gnn", USER_NODE, PRODUCT_NODE, ORDER_EDGE))]
    df_orders = df_orders.loc[:, df_orders.columns.str.startswith(("gnn", USER_NODE, PRODUCT_NODE, ORDER_EDGE))]

    #Merge dataframes
    df_features = pd.merge(df_order_items, df_orders, on=[USER_NODE, ORDER_EDGE])
    
    
    
    #Validate featured df size before returning
    assert len(df_features) == len(df_order_items), f"Features dataframe row size ({len(features_df)}) does not match original row size ({len(df_order_items)})"

    return df_features


# In[23]:


if __name__ == '__main__':
    df = pd.read_csv(UNIFIED_CSV_PATH)
    df = check_columns(df)
    df = remove_duplicate(df)
    df = handle_nulls(df)
    df = feature_engineering(df)
    df = assign_node_id(df, USER_NODE, PRODUCT_NODE)
    df.to_csv(f"{BASE_DIR}/features.csv", index=False)
    print(f"Features Saved at {BASE_DIR}/features.csv")

