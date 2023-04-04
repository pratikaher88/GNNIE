#!/usr/bin/env python
# coding: utf-8

import numpy as np
from collections import defaultdict


"""
Hit Rate

Calculates Hit Rate score from a set of recommendations.  
For each positive edge (purchase, review, etc.) in the test set, we generate a list of recommendations. Hit-rate is 
defined as the fraction of edges in the test set that is reflected in the first K recommendations for that user. 
This metric directly measures the probability that recommendations
made by the algorithm contain the items related to the user u.

Parameters
----------
test_recs : dictionary 
    Dictionary of test recommendations, in which keys corresponds to user indexes, and values corresponds to product indexes.
    This dictionary should be ordered by best to worst recommendations for each user key.
recommendations : dictionary
    Dictionary of recommendations, in which keys corresponds to user indexes, and values corresponds to product indexes.
    This dictionary should be ordered by best to worst recommendations for each user key. 
K: int
    Number of recommendations that will be considered for the evaluation.  

Returns: float
    Hit Rate Score
-------
DGL Heterograph"""
    
def hit_rate_accuracy(test_recs, recommendations, num_recs):
    
    if(num_recs==0): 
        return 0
    hits, total = 0, 0
    for k, v in test_recs.items():
        hits += sum(edge in v for edge in recommendations.get(k)[:num_recs])
        total = total + num_recs
    
    return hits/total

def hit_rate_recall(test_recs, recommendations, num_recs):
    
    if(num_recs==0): 
        return 0
    hits, total = 0, 0
    for k, v in test_recs.items():
        hits += sum(edge in v for edge in recommendations.get(k)[:num_recs])
        total += len(v)

    return hits/total

def hr_auc_rr(test_recs, recommendations, thresholds):
    
    auc = 0
    for t in thresholds:
        hit_rate_precision
        auc += hit_rate_precision(test_recs, recommendations, t)*hit_rate_recall(test_recs, recommendations, t)
    return auc


"""
Mean Reciprocal Rank

Score that takes into account of the rank of each item among user's recommendations. 

Parameters
----------
test_recs : dictionary 
    Dictionary of test recommendations, in which keys corresponds to user indexes, and values corresponds to product indexes.
    This dictionary should be ordered by best to worst recommendations for each user key.
recommendations : dictionary
    Dictionary of recommendations, in which keys corresponds to user indexes, and values corresponds to product indexes.
    This dictionary should be ordered by best to worst recommendations for each user key. 
scaling_factor: int, optional
    For bigger datasets, the scaling factor ensures that, for example, the difference between rank 
    at 1,000 and rank at 2,000 is still noticeable, instead of being very close to 0.

Returns: float
    MMR Score
-------
DGL Heterograph"""
    
def mmr(test_recs, recommendations, scaling_factor = 1):
    
    agg_rec_rank, total = 0, 0
    for user, product_list in test_recs.items():
        for product in product_list:
            rec_rank = 0
            if product in recommendations.get(user):
                rec_rank = 1/((recommendations.get(user).tolist().index(product)+1)/scaling_factor)
            else:
                rec_rank = 1/(recommendations.get(user).size+1)
            agg_rec_rank += rec_rank
            total += 1

    return agg_rec_rank/total
    

