import torch
import torch.nn as nn
import torch.nn.functional as F


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


def binary_cross_entropy_loss(pos_score, neg_score):
    all_scores = torch.empty(0)

    for etype in pos_score.keys():
        neg_score_tensor = neg_score[etype]
        pos_score_tensor = pos_score[etype]

        pred = torch.cat((pos_score_tensor, neg_score_tensor), dim=0)

        ground_truth = torch.cat([torch.ones(pos_score_tensor.size(0)), torch.zeros(neg_score_tensor.size(0))], dim=0)
        ground_truth = ground_truth.unsqueeze(1)

        scores = F.binary_cross_entropy_with_logits(pred, ground_truth)
        relu = nn.ReLU()
        scores = relu(scores)
        scores = scores.reshape(1)

        all_scores = torch.cat((all_scores, scores), 0)

    return torch.mean(all_scores)