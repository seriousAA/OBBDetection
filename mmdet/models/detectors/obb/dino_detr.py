# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from mmdet.models.builder import DETECTORS
from .obb_single_stage import OBBSingleStageDetector



@DETECTORS.register_module()
class DinoDetrOBB(OBBSingleStageDetector):
    """Implementation of rotated version 
    `DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection
    <https://arxiv.org/abs/2203.03605>`_ for OBB detection."""
    
    def __init__(self,
                 backbone,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(DinoDetrOBB, self).__init__(backbone, None, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)
    

    # over-write `forward_dummy` because:
    # the forward of bbox_head requires img_metas
    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        warnings.warn('Warning! MultiheadAttention in DETR does not '
                      'support flops computation! Do not use the '
                      'results in your papers!')

        batch_size, _, height, width = img.shape
        dummy_img_metas = [
            dict(
                batch_input_shape=(height, width),
                img_shape=(height, width, 3)) for _ in range(batch_size)
        ]
        x = self.extract_feat(img)
        outs = self.bbox_head(x, dummy_img_metas)
        return outs

