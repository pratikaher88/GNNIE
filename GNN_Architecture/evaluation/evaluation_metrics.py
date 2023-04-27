#!/usr/bin/env python
# coding: utf-8

import numpy as np
from collections import defaultdict
import math


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
    
def hit_rate_precision(test_recs, recommendations, num_recs):
    
    if(num_recs==0): 
        return 0
    hits, total = 0, 0
    count_none = 0
    for k, v in test_recs.items():
        if recommendations.get(k) is None:
            count_none += 1
            continue
        hits += sum(edge in v for edge in recommendations.get(k)[:num_recs])
        total = total + num_recs
    
    if count_none > 0 :
        print("Non-existent users in Precision :", count_none)
    
    return hits/total

def hit_rate_recall(test_recs, recommendations, num_recs):
    
    if(num_recs==0): 
        return 0
    hits, total = 0, 0
    count_none = 0
    for k, v in test_recs.items():
        if recommendations.get(k) is None:
            count_none += 1
            continue
        hits += sum(edge in v for edge in recommendations.get(k)[:num_recs])
        total += len(v)

    if count_none:
        print("Non-existent users in Recall :", count_none)

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
    count_none = 0
    for user, product_list in test_recs.items():
        for product in product_list:
            
            if recommendations.get(user) is None:
                count_none += 1
                continue

            rec_rank = 0
            if product in recommendations.get(user):
                rec_rank = 1/((recommendations.get(user).tolist().index(product)+1)/scaling_factor)
            else:
                rec_rank = 1/(recommendations.get(user).size+1)
            agg_rec_rank += rec_rank
            total += 1

    if count_none > 0:
        print("None existent users in MMR: ", count_none)

    return agg_rec_rank/total




""" 
Takes values of p and d
----------
p : Weight parameter, giving the influence of the first d
    elements on the final score. p<0<1.
d : depth at which the weight has to be calculated
    
Returns
-------
Float of Weightage Wrbo at depth d
"""
def weightage_calculator(p,d):


    summation_term = 0

    for i in range (1, d): # taking d here will loop upto the value d-1 
        summation_term = summation_term + math.pow(p,i)/i


    Wrbo_1_d = 1 - math.pow(p, d-1) + (((1-p)/p) * d *(np.log(1/(1-p)) - summation_term))

    return Wrbo_1_d


""" Takes two lists S and T of any lengths and gives out the RBO Score
Parameters
----------
S, T : Lists (str, integers)
p : Weight parameter, giving the influence of the first d
    elements on the final score. p<0<1. Default 0.9 give the top 10 
    elements 86% of the contribution in the final score.
    
Returns
-------
Float of RBO score
"""
    
def rbo(S,T, p= 0.9):

    # Fixed Terms
    k = max(len(S), len(T))
    x_k = len(set(S).intersection(set(T)))
    
    summation_term = 0

    # Loop for summation
    # k+1 for the loop to reach the last element (at k) in the bigger list    
    for d in range (1, k+1): 
        # Create sets from the lists
        set1 = set(S[:d]) if d < len(S) else set(S)
        set2 = set(T[:d]) if d < len(T) else set(T)
            
        # Intersection at depth d
        x_d = len(set1.intersection(set2))

        # Agreement at depth d
        a_d = x_d/d   
            
        # Summation
        summation_term = summation_term + math.pow(p, d) * a_d

    # Rank Biased Overlap - extrapolated
    rbo_ext = (x_k/k) * math.pow(p, k) + ((1-p)/p * summation_term)

    return rbo_ext

