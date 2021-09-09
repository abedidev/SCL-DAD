import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):

    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, n_vec, a_vec, tau):

        n_scores = torch.mm(n_vec, n_vec.t())
        n_scores = n_scores[~torch.eye(n_scores.shape[0], dtype=bool)].reshape(n_vec.shape[0], -1).div_(tau).exp_().view(-1, 1)  # n_scores is numerator
        n_a_scores = torch.mm(n_vec, a_vec.t()).div_(tau).exp()
        sum_n_a = torch.sum(n_a_scores, dim=1, keepdim=True)
        sum_n_a = sum_n_a.repeat(1, (n_vec.shape[0]-1)).view(-1, 1)
        denominator = n_scores + sum_n_a / a_vec.shape[0]
        p = torch.log(torch.div(n_scores, denominator))
        loss = -torch.sum(p)
        return loss


