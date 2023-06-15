import torch
from torch import nn


def pos_neg_mask_xy(labels_col, labels_row, batch_size):

    n_row = labels_row.shape[0] // batch_size
    n_col = labels_col.shape[0] // batch_size

    pos_mask = (labels_row.unsqueeze(0) == labels_col.unsqueeze(1)) 
    # reomve the anchor itself
    pos_mask = pos_mask  * (~torch.eye(batch_size, dtype=torch.bool, device=labels_col.device).repeat(n_col, n_row))

    neg_mask = (labels_row.unsqueeze(0) != labels_col.unsqueeze(1)) 

    return pos_mask, neg_mask


class MultiSimilarityLoss(nn.Module):
    def __init__(self, batch_size=128, all_anchor=False):
        super().__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = 2.0
        self.scale_neg = 40.0  
        self.epsilon = 1e-5

        self.batch_size = batch_size
        self.all_anchor = all_anchor

    def forward(self, inputs_col, targets_col):

        # common
        if self.batch_size == inputs_col.shape[0]:
            inputs_row = inputs_col
            targets_row = targets_col
        # with aug
        else:
            inputs_row = inputs_col
            targets_row = targets_col
            if not self.all_anchor:
                inputs_col = inputs_col[:self.batch_size]
                targets_col = targets_col[:self.batch_size]

        sim_mat = torch.matmul(inputs_col, inputs_row.t())

        loss = []
        pos_mask, neg_mask = pos_neg_mask_xy(targets_col, targets_row, batch_size=self.batch_size)

        for i in range(self.batch_size):

            pos_pair_ = torch.masked_select(sim_mat[i], pos_mask[i])

            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1 - self.epsilon)

            neg_pair_ = torch.masked_select(sim_mat[i], neg_mask[i])

            if len(neg_pair_) < 1 or len(pos_pair_) < 1:
                continue

            neg_pair = neg_pair_[neg_pair_ + self.margin > torch.min(pos_pair_)]  # hard neg_pair select
            pos_pair = pos_pair_[pos_pair_ - self.margin < torch.max(neg_pair_)]  # hard pos_pair select

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True, device=inputs_col.device)
        else:
            loss = sum(loss) / self.batch_size
 
            return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, batch_size=128, pos_margin=0.8, neg_margin=0.5, all_anchor=False):
        super().__init__()

        self.margin = margin
        self.neg_margin = neg_margin
        self.pos_margin = pos_margin
        self.batch_size = batch_size

        self.all_anchor = all_anchor

    def forward(self, inputs_col, targets_col):

        # common
        if self.batch_size == inputs_col.shape[0]:
            inputs_row = inputs_col
            targets_row = targets_col
        # with aug
        else:
            inputs_row = inputs_col
            targets_row = targets_col
            if not self.all_anchor:
                inputs_col = inputs_col[:self.batch_size]
                targets_col = targets_col[:self.batch_size]

        sim_mat = torch.matmul(inputs_col, inputs_row.t())

        pos_mask, neg_mask = pos_neg_mask_xy(targets_col, targets_row, batch_size=self.batch_size)

        pos_mat = torch.masked_select(sim_mat, pos_mask)
        pos_mat = torch.masked_select(pos_mat, pos_mat < self.pos_margin)

        neg_mat = torch.masked_select(sim_mat, neg_mask)
        neg_mat = torch.masked_select(neg_mat, neg_mat > self.neg_margin)

        loss = torch.sum(self.pos_margin - pos_mat) + torch.sum(neg_mat - self.neg_margin)
        loss = loss / self.batch_size
        
        return loss
