#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np

BASE_DIR_ARCHIVE = '/Users/benjaminfell/Documents/GitHub/GNNIE/DatasetEDA/archive'

customer_data = pd.read_csv(f'{BASE_DIR_ARCHIVE}/olist_customers_dataset.csv')
geolocation_data = pd.read_csv(f'{BASE_DIR_ARCHIVE}/olist_geolocation_dataset.csv')
order_items_dataset = pd.read_csv(f'{BASE_DIR_ARCHIVE}/olist_order_items_dataset.csv')
#order_payments_dataset = pd.read_csv(f'{BASE_DIR_ARCHIVE}/olist_order_payments_dataset.csv')
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
#ordered_product_reviews_payments = pd.merge(ordered_product_reviews,order_payments_dataset,on='order_id')
df_unified = pd.merge(ordered_product_reviews,customer_data,on='customer_id')

df_unified.to_csv(BASE_DIR_ARCHIVE + '/unified.csv', index=False)