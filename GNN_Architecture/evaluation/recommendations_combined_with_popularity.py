import numpy as np

def get_recommendations_combined_with_popularity(rec_list, pop_list, epsilon = 0, sample_values = False):
    '''
    The input is the list of all products along with thier scores not in a sorted order. 
    e.g [100, 20, 120, 200] : this means that product 0 has score 0, product 1 has score 20, product 3 has score 120 and so on.
    '''
    rec_scores = rec_scores / np.max(rec_list)
    pop_scores = pop_scores / np.max(pop_list)

    combined_scores = epsilon*pop_scores + (1-epsilon)*rec_scores

    if sample_values:
        number_of_recommendations = len(rec_list)
        cdf_products = np.cumsum(combined_scores)
        rec_products = np.zeros(number_of_recommendations, dtype=int)
        for i in range(number_of_recommendations):
            found_new_product = False
            while not found_new_product:
                rand_p = np.random.rand()
                candidate_rec_product = np.argmax(rand_p <= cdf_products)
                found_new_product = all(candidate_rec_product != rec_products)
                rec_products[i] = candidate_rec_product
        
        return rec_products
                
    else:
        return combined_scores.argsort()



