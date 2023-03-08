import torch
import torch.nn as nn

def max_margin_loss(pos_score, neg_score, delta=0.5):

    all_scores = torch.empty(0)
    
    for etype in pos_score.keys():
        neg_score_tensor = neg_score[etype]
        pos_score_tensor = pos_score[etype]

        neg_score_tensor = neg_score_tensor.reshape(pos_score_tensor.shape[0], -1)
        scores = (neg_score_tensor - pos_score_tensor + delta).clamp(min=0)

        relu = nn.ReLU()
        scores = relu(scores)
        all_scores = torch.cat((all_scores, scores), 0)

    return torch.mean(all_scores)