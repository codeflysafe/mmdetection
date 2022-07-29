# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES
from .utils import weight_reduce_loss


def co_dice_loss(pred,
              label_matrixs,
              weight=None,
              eps=1e-3,
              reduction='mean',
              naive_dice=False,
              avg_factor=None):
    """Calculate co dice loss, there are two forms of dice loss is supported:

        - the one proposed in `V-Net: Fully Convolutional Neural
            Networks for Volumetric Medical Image Segmentation
            <https://arxiv.org/abs/1606.04797>`_.
        - the dice loss in which the power of the number in the
            denominator is the first power instead of the second
            power.

    Args:
        pred (torch.Tensor): The prediction, has a shape (n, *)
        target (torch.Tensor): The learning label of the prediction,
            shape (n, *), same shape of pred.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-3.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power.Defaults to False.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """

    batch, num_grid, h, w = pred.shape
    input = pred.reshape(batch, num_grid, -1)
    a = torch.matmul(input, input.permute(0,2,1))
    if naive_dice:
        b = torch.sum(input, 1).view(1, -1)
        d = (2 * a + eps) / (b + b.t() + eps)
    else:
        b = torch.matmul(input , torch.ones(h, w)) + eps
        c = b[:,:, None] + b[:,None,:]
        d = 2 * a/ c
    # 只取上三角矩阵
    d = torch.triu(d, diagonal=1)
    # ground truth
    gt = label_matrixs.reshape(batch, -1)
    pred_d = d.reshape(batch, -1)
    # 是一个分类问题
    loss = F.binary_cross_entropy(pred_d, gt)
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss



@LOSSES.register_module()
class CoMaskDiceLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 activate=True,
                 reduction='mean',
                 naive_dice=False,
                 loss_weight=1.0,
                 eps=1e-3):
        """Compute dice loss.

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            activate (bool): Whether to activate the predictions inside,
                this will disable the inside sigmoid operation.
                Defaults to True.
            reduction (str, optional): The method used
                to reduce the loss. Options are "none",
                "mean" and "sum". Defaults to 'mean'.
            naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power. Defaults to False.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            eps (float): Avoid dividing by zero. Defaults to 1e-3.
        """

        super(CoMaskDiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.loss_weight = loss_weight
        self.eps = eps
        self.activate = activate

    def forward(self,
                pred,
                label_matrixs,
                weight=None,
                reduction_override=None,
                avg_factor=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction, has a shape (n, *).
            target (torch.Tensor): The label of the prediction,
                shape (n, *), same shape of pred.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction, has a shape (n,). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            else:
                raise NotImplementedError
        loss = self.loss_weight * co_dice_loss(
            pred,
            label_matrixs,
            weight,
            eps=self.eps,
            reduction=reduction,
            naive_dice=self.naive_dice,
            avg_factor=avg_factor)

        return loss