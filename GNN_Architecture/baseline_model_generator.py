#!/usr/bin/env python
# coding: utf-8

# In[5]:


from model import ConvModel
import dgl, torch
import torch.nn as nn
import numpy as np
import random


# In[2]:

"""
generate_popularity_model

Generates a recommendation list based on how popular products/items are. 
First, number of in-degree edges are calculated for distinct items.
Second, we generate an ordered list from most popular (higher count of in-degree edges) to least popular (lower count of in-degree edges) distinct items.
Lastly, we append this same list to every user as a recommendation.

Parameters
----------
graph : DGL Heterograph
    Graph that will be used to extract the most popular items.
edge_label : string
    Name of the edge labels to be considered as "popular" (orders, purchases, etc.) 
user_node_label: string
    Name of the node labels for which the recommendations will be provided (users, customers, etc.) 

Returns: dictionary
    Recommendations dictionary: (key,value) = (user,recommendations), where recommendations is a numpy array
-------
DGL Heterograph"""
def generate_popularity_model(graph, edge_label, user_node_label):

    product_orders = graph.edges(etype=edge_label)[1].numpy()
    baseline_recs = {}

    for product in product_orders:
        if baseline_recs.get(product) == None:
            baseline_recs[product] = 1
        else:
            baseline_recs[product] += 1
    baseline_recs = np.array(list(dict(sorted(baseline_recs.items(), key=lambda item: item[1],reverse=True)).keys()))
    customers_list = graph.nodes[user_node_label].data['_ID'].tolist()
    baseline_model = {}
    for customer in customers_list:
        baseline_model[customer] = baseline_recs

    return baseline_model


# In[7]:

"""
generate_random_model

Generates a random recommendation list. 
First, we extract distinct items from the graph and append them into a list.
Then, for each user, we shuffle the list and append it as a recommendation.

Parameters
----------
graph : DGL Heterograph
    Graph that will be used to extract the items and users.
user_node_label: string
    Name of the node labels in "graph" for which the recommendations will be provided (users, customers, etc.) 
product_node_label : string
    Name of the node labels in "graph" that will be recommended (orders, purchases, etc.) 

Returns: dictionary
    Recommendations dictionary: (key,value) = (user,recommendations), where recommendations is a numpy array
-------
DGL Heterograph"""
def generate_random_model(graph, user_node_label, product_node_label):

    orders = np.array(graph.nodes[product_node_label].data['_ID'].tolist())
    customers_list = graph.nodes[user_node_label].data['_ID'].tolist()
    random_model = {}

    for customer in customers_list:
        np.random.shuffle(orders)
        random_model[customer] = np.array(orders)

    return random_model

