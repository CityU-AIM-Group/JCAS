import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255, is_softmax=True):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label
        self.is_softmax = is_softmax

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        if self.is_softmax:
            loss = F.cross_entropy(predict, target, weight=weight, reduction='mean')
        else:
            log_out = torch.log(predict.clamp(min=1e-10, max=1.0))
            loss = F.nll_loss(log_out, target, weight=weight, reduction='mean')
        return loss

class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(1)
        return b.mean()

class Focal_CrossEntropy2d(nn.Module):
    def __init__(self, ignore_label=255, gamma=0, is_softmax=True):
        super(Focal_CrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.gamma = gamma
        self.is_softmax = is_softmax

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        #print(predict.shape, target.shape)
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, reduction='none')
        
        # focal loss
        if self.is_softmax:
            logpt = F.log_softmax(predict, dim=1)
        else:
            logpt = torch.log(predict)
        target = target.view(-1, 1)
        logpt = logpt.gather(1, target)
        pt = Variable(logpt.data.exp()).view(-1)
        loss = (1 - pt)**self.gamma * loss
        return loss.mean()

class BinaryCE(nn.Module):
    def __init__(self):
        super(BinaryCE, self).__init__()
    
    def forward(self, predict, label):
        """calculate bce
        Args:
            predict: A tensor of shape (n, c, h, w)
            label: one digital representing domain label
        Returns:
            loss
        """
        label = torch.FloatTensor(predict.data.size()).fill_(label).cuda(predict.get_device())
        loss = F.binary_cross_entropy_with_logits(predict, label)
        return loss

class WeightBinaryCE(nn.Module):
    def __init__(self):
        super(WeightBinaryCE, self).__init__()
    
    def forward(self, predict, label, weight):
        """calculate bce
        Args:
            predict: A tensor of shape (n, c, h, w)
            label: one digital representing domain label
        Returns:
            loss
        """
        label = torch.FloatTensor(predict.data.size()).fill_(label).cuda(predict.get_device())
        loss = F.binary_cross_entropy_with_logits(predict, label, reduction='none')
        b, c, w, h = loss.shape
        weight = F.interpolate(weight.unsqueeze(1), (w, h))
        loss = loss * weight
        loss = loss.mean()
        return loss

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape).cuda()
    result = result.scatter_(1, input, 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        #num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        #den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth
        #print(num, den)
        num = 2 * (predict * target).sum()
        den = predict.sum() + target.sum() + 1e-5
        loss = 1 - num / den
        return loss


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of weight
        ignore_index: a list of class index to ignore
        class_num: class number
        predict: [B, C, W, H]
        target: [B, *]
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, class_num=None, ignore_label=255, is_softmax=True, **kwargs):
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.class_num = class_num
        self.ignore_label = ignore_label
        self.dice = BinaryDiceLoss(**kwargs)
        self.is_softmax = is_softmax

    def forward(self, predict, target):
        total_loss = 0
        
        if self.is_softmax:
            predict = F.softmax(predict, dim=1)
        b, c, w, h = predict.shape
        predict, target = torch.reshape(predict, [b, c, w * h]), torch.reshape(target, [b, w * h])
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask].unsqueeze(0).unsqueeze(0)
        predict = predict.transpose(1, 2).contiguous()
        predict = predict[target_mask.view(b, w*h, 1).repeat(1, 1, c)].view(-1, c).transpose(1, 0).unsqueeze(0)
        target = make_one_hot(target, self.class_num)
        for i in range(1, target.shape[1]):
            if i not in self.ignore_index:# and i != 0:
                dice_loss = self.dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert len(self.weight) == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], len(self.weight))
                    dice_loss *= self.weight[i]
                total_loss += dice_loss

        return total_loss/(target.shape[1]-1)