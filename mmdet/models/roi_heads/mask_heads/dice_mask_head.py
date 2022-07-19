'''
Author: sjhuang
Date: 2022-07-19 22:24:13
LastEditTime: 2022-07-19 22:37:02
FilePath: /mmdetection/mmdet/models/roi_heads/mask_heads/dice_mask_head.py
'''
import torch
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, ModuleList, auto_fp16, force_fp32
from mmdet.core import mask
from mmdet.models.builder import HEADS, build_loss
from .fcn_mask_head import FCNMaskHead


@HEADS.register_module()
class DiceMaskHead(FCNMaskHead):

    def __init__(self, co_mask_loss = dict(
                     type='DiceLoss', use_mask=True, loss_weight=1.0),
                     *args, **kwargs):
        super(DiceMaskHead, self).__init__(*args, **kwargs)
        self.co_mask_loss = build_loss(co_mask_loss)
        

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, mask_targets, labels):
        """
        Example:
            >>> from mmdet.models.roi_heads.mask_heads.fcn_mask_head import *  # NOQA
            >>> N = 7  # N = number of extracted ROIs
            >>> C, H, W = 11, 32, 32
            >>> # Create example instance of FCN Mask Head.
            >>> # There are lots of variations depending on the configuration
            >>> self = FCNMaskHead(num_classes=C, num_convs=1)
            >>> inputs = torch.rand(N, self.in_channels, H, W)
            >>> mask_pred = self.forward(inputs)
            >>> sf = self.scale_factor
            >>> labels = torch.randint(0, C, size=(N,))
            >>> # With the default properties the mask targets should indicate
            >>> # a (potentially soft) single-class label
            >>> mask_targets = torch.rand(N, H * sf, W * sf)
            >>> loss = self.loss(mask_pred, mask_targets, labels)
            >>> print('loss = {!r}'.format(loss))
        """
        loss = dict()
        if mask_pred.size(0) == 0:
            loss_mask = mask_pred.sum()
        else:
            if self.class_agnostic:
                loss_mask = self.loss_mask(mask_pred, mask_targets,
                                           torch.zeros_like(labels))
            else:
                loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask

        if mask_pred.size(0) == 0:
            return loss
        else:
            co_loss_mask = self.co_mask_loss(mask_pred, mask_pred, None)
        loss['loss_mask'] = loss_mask + co_loss_mask
        return loss
