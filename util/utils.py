import numpy as np
from PIL import Image
from thop import profile
from thop import clever_format
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as initer
import cv2, os
import random


def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def step_learning_rate(base_lr, epoch, step_epoch, multiplier=0.1):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = base_lr * (multiplier ** (epoch // step_epoch))
    return lr


def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def init_weights(model, conv='kaiming', batchnorm='normal', linear='kaiming', lstm='kaiming'):
    """
    :param model: Pytorch Model which is nn.Module
    :param conv:  'kaiming' or 'xavier'
    :param batchnorm: 'normal' or 'constant'
    :param linear: 'kaiming' or 'xavier'
    :param lstm: 'kaiming' or 'xavier'
    """
    for m in model.modules():
        if isinstance(m, (nn.modules.conv._ConvNd)):
            if conv == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif conv == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of conv error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, (nn.modules.batchnorm._BatchNorm)):
            if batchnorm == 'normal':
                initer.normal_(m.weight, 1.0, 0.02)
            elif batchnorm == 'constant':
                initer.constant_(m.weight, 1.0)
            else:
                raise ValueError("init type of batchnorm error.\n")
            initer.constant_(m.bias, 0.0)

        elif isinstance(m, nn.Linear):
            if linear == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif linear == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of linear error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    if lstm == 'kaiming':
                        initer.kaiming_normal_(param)
                    elif lstm == 'xavier':
                        initer.xavier_normal_(param)
                    else:
                        raise ValueError("init type of lstm error.\n")
                elif 'bias' in name:
                    initer.constant_(param, 0)


def group_weight(weight_group, module, lr):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group


def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


def overlay(img, mask, color=(1.0, 0, 0), alpha=0.4, resize=None):
    """Combines image and its segmentation mask into a single image.

    Params:
        image: Training image.
        mask: Segmentation mask.
        color: Color for segmentation mask rendering.
        alpha: Segmentation mask's transparency.
        resize: If provided, both image and its mask are resized before blending them together.

    Returns:
        image_combined: The combined image.

    """
    color = np.asarray(color).reshape(3, 1, 1)
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    masked = np.ma.MaskedArray(img, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()
    img = img.transpose(1, 2, 0)
    image_overlay = image_overlay.transpose(1, 2, 0)

    if resize is not None:
        img = cv2.resize(img, resize)
        image_overlay = cv2.resize(image_overlay, resize)

    image_combined = cv2.addWeighted(img, 1 - alpha, image_overlay, alpha, 0)

    return image_combined


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, p=1, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def mse_loss(input, target):
    """Takes sigmoid on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input.size() == target.size()

    # input_sigmoid = F.sigmoid(input)
    # target_sigmoid = F.sigmoid(target)
    mse_loss = (input - target) ** 2
    c = 1
    for s in input.size():
        c *= s
    mse_loss = mse_loss.sum() / c
    return mse_loss

class DiceLoss(nn.Module):

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        if predict.shape[1] == 1:
            predict = predict.sigmoid()
        else:
            predict = F.softmax(predict, dim=1)
        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]


class BinaryIoU():
    def __init__(self, epsilon=1e-6, reduction='mean'):
        super(BinaryIoU, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def __call__(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        inter = torch.sum(predict * target, dim=1)
        union = torch.sum(predict + target, dim=1) - inter

        iou = inter / (union + self.epsilon)

        if self.reduction == 'mean':
            return iou.mean()
        elif self.reduction == 'sum':
            return iou.sum()
        elif self.reduction == 'none':
            return iou
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class mIoU():
    def __init__(self, num_classes, weight=None, ignore_index=None, **kwargs):
        super(mIoU, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def __call__(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        assert predict.shape[1] == self.num_classes
        binaryIoU = BinaryIoU(**self.kwargs)
        total_iou = 0
        if self.num_classes > 1:
            predict = torch.argmax(predict, dim=1)
            predict = F.one_hot(predict, self.num_classes).permute(0, 3, 1, 2)
        else:
            predict = predict.sigmoid() > 0.15
            predict = predict.to(torch.float32)
        for i in range(self.num_classes):
            if i != self.ignore_index:
                iou = binaryIoU(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    iou *= self.weights[i]
                total_iou += iou
        return total_iou/self.num_classes


def check(feat):
    if feat.shape[0] == 0 or len(feat.shape) != 2:
        return None
    return feat

def randomSample(feat:torch.Tensor, pred:torch.Tensor, gt:torch.Tensor, training=False, sample_num=100):    #HXWXC, HXW, HXW
    pred = pred.sigmoid() > 0.5
    pred = pred.view(-1).long()
    gt = gt.view(-1).long()
    TP = gt * pred
    FP = pred * (1 - gt)
    TN = (1 - pred) * (1 - gt)
    FN = (1 - pred) * gt
    tp_feat = feat[torch.nonzero(TP)].squeeze()
    fp_feat = feat[torch.nonzero(FP)].squeeze()
    tn_feat = feat[torch.nonzero(TN)].squeeze()
    fn_feat = feat[torch.nonzero(FN)].squeeze()

    tp_n, fp_n, tn_n, fn_n = tp_feat.shape[0], fp_feat.shape[0], tn_feat.shape[0], fn_feat.shape[0]
    a_n = tp_n + fp_n + fn_n
    if a_n:
        # tp_n, fp_n, fn_n = int(tp_n*sample_num/(2*a_n)), int(fp_n*sample_num/(2*a_n)), int(fn_n*sample_num/(2*a_n))
        tp_n, fp_n, fn_n = 100, 100, 100
    else:
        tp_n = fp_n = fp_n = 0
    s1 = torch.Tensor(random.sample(range(tp_feat.shape[0]), min(tp_n, tp_feat.shape[0]))).to(feat.device)
    s2 = torch.Tensor(random.sample(range(fp_feat.shape[0]), min(fp_n, fp_feat.shape[0]))).to(feat.device)
    s3 = torch.Tensor(random.sample(range(tn_feat.shape[0]), min(sample_num//2, tn_n))).to(feat.device)
    s4 = torch.Tensor(random.sample(range(fn_feat.shape[0]), min(fn_n, fn_feat.shape[0]))).to(feat.device)

    if training:
        s1 = s1.requires_grad_()
        s2 = s2.requires_grad_()
        s3 = s3.requires_grad_()
        s4 = s4.requires_grad_()
    
    tp_feat = check(torch.index_select(tp_feat, 0, s1.long()))
    fp_feat = check(torch.index_select(fp_feat, 0, s2.long()))
    tn_feat = check(torch.index_select(tn_feat, 0, s3.long()))
    fn_feat = check(torch.index_select(fn_feat, 0, s4.long()))
    return tp_feat, fp_feat, tn_feat, fn_feat

def avgFeat(pred:torch.Tensor, gt:torch.Tensor, feat:torch.Tensor):

    tp_feat, fp_feat, tn_feat, fn_feat = randomSample(feat, pred, gt, sample_num=50)

    feat_list_ = [tp_feat, fp_feat, tn_feat, fn_feat]
    feat_list, label_list = [], []

    for i in range(4):
        if feat_list_[i] is not None:
            feat_list.append(feat_list_[i])
            label_list.append(torch.ones(feat_list_[i].shape[0]) * (i + 1))


    return feat_list, label_list
    
def sigmoid_rampup(current_epoch, rampup_length):
    current = np.clip(current_epoch, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return np.exp(-5.0 * phase * phase).astype(np.float32)

def get_current_consistency_weight(epoch, consistency_rampup, weight = 0.1):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return weight * sigmoid_rampup(epoch, consistency_rampup)

def sigmoid_mse_loss(input_logits, target_logits):
    """Takes sigmoid on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_sigmoid = torch.sigmoid(input_logits)
    target_sigmoid = torch.sigmoid(target_logits)

    mse_loss = (input_sigmoid - target_sigmoid)**2
    return mse_loss

def mean_metric(preds: torch.Tensor, target: torch.Tensor):

    assert preds.shape == target.shape

    preds = preds.reshape(preds.shape[0], -1)
    target = target.reshape(target.shape[0], -1)

    preds = (preds > 0.5).float()
    target = (target > 0).float()

    TP = (preds * target).sum(1)
    FN = ((1 - preds) * target).sum(1)
    TN = ((1 - preds) * (1 - target)).sum(1)
    FP = (preds * (1 - target)).sum(1)

    acc = (TP + TN) / (TP + TN + FP + FN + 1e-4)
    iou = TP / (TP + FP + FN + 1e-4)
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-4)
    spe = TP / (TP + FP + 1e-4)
    sen = TP / (TP + FN + 1e-4)
    
    return acc.mean(), iou.mean(), dice.mean(), spe.mean(), sen.mean()
