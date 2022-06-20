# -*- encoding: utf-8 -*-
#Time        :2020/12/19 21:17:50
#Author      :Chen
#FileName    :loss.py
#Version     :1.0


import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

"""BCE loss"""

class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss(weight=weight, size_average=size_average)

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        loss = self.bceloss(pred_flat, target_flat)

        return loss


"""Dice loss"""


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target, weight=None):
        smooth = 1

        size = pred.size(0)

        dice = 0.0
        for i in range(pred.shape[1]):
            p = pred[:,i,:,:]
            lab = (target==i)

            pred_flat = p.view(size, -1)
            target_flat = lab.view(size, -1)

            intersection = pred_flat * target_flat
            dice_score = (2 * intersection.sum(1) + smooth)/(pred_flat.sum(1) + target_flat.sum(1) + smooth)
            if weight is not None:
                #print(weight)
                dice_score = weight * dice_score
                dice_loss = weight.sum() /size - dice_score.sum()/size
            else:
                dice_loss = 1 - dice_score.sum()/size
            dice += dice_loss

        return dice


"""BCE + DICE Loss"""


class BceDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss(weight, size_average)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = diceloss + bceloss

        return loss


""" Entropy Minimization"""
class softCrossEntropy(nn.Module):
    def __init__(self, ignore_index= -1):
        super(softCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions (N, C, H, W)
        :param target: target distribution (N, C, H, W)
        :return: loss
        """
        assert inputs.size() == target.size()
        mask = (target != self.ignore_index)

        log_likelihood = F.log_softmax(inputs, dim=1)
        loss = torch.mean(torch.mul(-log_likelihood, target)[mask])

        return loss


"""Maxsquare Loss"""
class MaxSquareloss(nn.Module):
    def __init__(self, ignore_index= -1):
        super().__init__()
        self.ignore_index = ignore_index
        #self.num_class = num_class
    
    def forward(self, prob):
        """
        :param pred: predictions (N, C, H, W)
        :param prob: probability of pred (N, C, H, W)
        :return: maximum squares loss
        """
        # prob -= 0.5
        #mask = (prob != self.ignore_index)    
        loss = -torch.mean(torch.pow(prob, 2) + torch.pow(1-prob, 2)) / 2
        return loss


class Entropy(nn.Module):
    def __init__(self, ignore_index= -1):
        super().__init__()
        self.ignore_index = ignore_index
        #self.num_class = num_class
    
    def forward(self, prob):
        """
        :param pred: predictions (N, C, H, W)
        :param prob: probability of pred (N, C, H, W)
        :return: entropy loss
        """
        # prob -= 0.5
        #mask = (prob != self.ignore_index)    
        loss = -torch.mean(-prob * torch.log(prob + 1e-5))
        return loss


class Diverse(nn.Module):
    def __init__(self, ignore_index= -1):
        super().__init__()
        self.ignore_index = ignore_index
        #self.num_class = num_class
    
    def forward(self, prob):
        """
        :param pred: predictions (N, C, H, W)
        :param prob: probability of pred (N, C, H, W)
        :return: diverse loss
        """
        # prob -= 0.5
        #mask = (prob != self.ignore_index)   
        prob = prob.mean(dim=0)
        loss = torch.mean(-prob * torch.log(prob + 1e-5))
        return loss