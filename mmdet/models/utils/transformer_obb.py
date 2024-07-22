# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import math
import warnings
from typing import Sequence
from typing import Optional
import copy
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmdet.models.utils.builder import TRANSFORMER

from mmdet.ops.ms_deform_attn import MSDeformAttn
DEBUG = 'DEBUG' in os.environ
from .transformer import MLP, gen_sineembed_for_position, inverse_sigmoid

def _get_clones(module, N, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation, d_model=256, batch_dim=0):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu

    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



def gen_encoder_output_proposals(memory:Tensor, memory_padding_mask:Tensor, spatial_shapes:Tensor, learnedwh=None):
    """
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
        - learnedwh: 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 5
    """
    N_, S_, C_ = memory.shape
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

        # import ipdb; ipdb.set_trace()

        grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                        torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1) # H_, W_, 2

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale

        if learnedwh is not None:
            # import ipdb; ipdb.set_trace()
            wh = torch.ones_like(grid) * learnedwh.sigmoid() * (2.0 ** lvl)
        else:
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)

     
        proposal = torch.cat((grid, wh, torch.zeros((*grid.shape[:-1], 1))
                                                .to(grid.device)), -1).view(N_, -1, 5)
        proposals.append(proposal)
        _cur += (H_ * W_)
    # import ipdb; ipdb.set_trace()
    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals)) # unsigmoid
    output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
    output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

    output_memory = memory
    output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

    return output_memory, output_proposals


@TRANSFORMER_LAYER.register_module()
class OBBDinoTransformerEncoderLayer(nn.Module):
    """Implements encoder layer in DINO.

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self, d_model=256, d_ffn=1024,
                       dropout=0.1, activation='relu',
                       n_levels=4, n_heads=8, n_points=5,
                       **kwargs):
        super(OBBDinoTransformerEncoderLayer, self).__init__()
        
        # self-attention, in DINO, the encoder's self-attention is
        # implemented with deformable attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)


    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None):
        # breakpoint()
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class OBBDinoTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, 
                       norm=None, d_model=256, two_stage_type='standard',
                       num_queries=900, deformable_encoder=True, 
                       enc_layer_share=False, enc_layer_dropout_prob=None,
                       **kwargs):
        super().__init__()
        # prepare layer
        if num_layers > 0:
            self.layers = _get_clones(encoder_layer, num_layers, layer_share=enc_layer_share)
        else:
            self.layers = []
            del encoder_layer

        self.query_scale = None
        self.two_stage_type = two_stage_type
        self.enc_layer_share = enc_layer_share
        self.num_queries = num_queries
        self.deformable_encoder = deformable_encoder
        self.num_layers = num_layers
        self.norm = norm
        self.d_model = d_model

        if enc_layer_dropout_prob is not None:
            assert isinstance(enc_layer_dropout_prob, list)
            assert len(enc_layer_dropout_prob) == num_layers
            for i in enc_layer_dropout_prob:
                assert 0.0 <= i <= 1.0


    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points in the first stage to generate the 
        initial object query.
        """
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
    
    def forward(self, src: Tensor, 
                      pos: Tensor, 
                      spatial_shapes: Tensor, 
                      level_start_index: Tensor, 
                      valid_ratios: Tensor, 
                      key_padding_mask: Tensor,
                      ref_token_index: Optional[Tensor]=None,
                      ref_token_coord: Optional[Tensor]=None):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - key_padding_mask: [bs, sum(hi*wi)]

            - ref_token_index: bs, nq
            - ref_token_coord: bs, nq, 5
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        Outpus: 
            - output: [bs, sum(hi*wi), 256]
        """
        # breakpoint()
        assert self.two_stage_type == 'standard'
        assert self.deformable_encoder is True
        assert ref_token_index is None
        assert ref_token_coord is None
        
        output = src

        # preparation and reshape
        if self.num_layers > 0:
            if self.deformable_encoder:
                reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
                # import ipdb; ipdb.set_trace()

        # intermediate_coord = []
        # main process
        for layer_id, layer in enumerate(self.layers):
            # main process output: [bs, HW, 256]
            output = layer(src=output, pos=pos, \
                           reference_points=reference_points, spatial_shapes=spatial_shapes, \
                           level_start_index=level_start_index, key_padding_mask=key_padding_mask)  
           
        if self.norm is not None:
            output = self.norm(output)

        intermediate_output = intermediate_ref = None

        return output, intermediate_output, intermediate_ref

@TRANSFORMER_LAYER.register_module()
class OBBDinoTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                       dropout=0.1, activation="relu",
                       n_levels=4, n_heads=8, n_points=5,
                       decoder_sa_type='ca',
                       module_seq=['sa', 'ca', 'ffn'],
                       **kwargs):
        super().__init__()

        self.module_seq = module_seq
        assert sorted(module_seq) == ['ca', 'ffn', 'sa']

        # cross attention        
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_type = None
        self.key_aware_proj = None
        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type == 'sa'

    
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_sa(self, tgt: Optional[Tensor],  # nq, bs, d_model
                         tgt_query_pos: Optional[Tensor] = None, # pos for query. MLP(Sine(pos))
                         tgt_query_sine_embed: Optional[Tensor] = None, # pos for query. Sine(pos)
                         tgt_key_padding_mask: Optional[Tensor] = None,
                         tgt_reference_points: Optional[Tensor] = None, # nq, bs, 5
                         memory: Optional[Tensor] = None, # hw, bs, d_model
                         memory_key_padding_mask: Optional[Tensor] = None,
                         memory_level_start_index: Optional[Tensor] = None, # num_levels
                         memory_spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                         memory_pos: Optional[Tensor] = None, # pos for memory
                         self_attn_mask: Optional[Tensor] = None, # mask used for self-attention
                         cross_attn_mask: Optional[Tensor] = None, # mask used for cross-attention
            ):
        # self attention
        if self.self_attn is not None:
            if self.decoder_sa_type == 'sa':
                q = k = self.with_pos_embed(tgt, tgt_query_pos)
                tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            else:
                raise NotImplementedError("Unknown decoder_sa_type {}".format(self.decoder_sa_type))

        return tgt

    def forward_ca(self, tgt: Optional[Tensor],  # nq, bs, d_model
                         tgt_query_pos: Optional[Tensor] = None, # pos for query. MLP(Sine(pos))
                         tgt_query_sine_embed: Optional[Tensor] = None, # pos for query. Sine(pos)
                         tgt_key_padding_mask: Optional[Tensor] = None,
                         tgt_reference_points: Optional[Tensor] = None, # nq, bs, 5
                         memory: Optional[Tensor] = None, # hw, bs, d_model
                         memory_key_padding_mask: Optional[Tensor] = None,
                         memory_level_start_index: Optional[Tensor] = None, # num_levels
                         memory_spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                         memory_pos: Optional[Tensor] = None, # pos for memory
                         self_attn_mask: Optional[Tensor] = None, # mask used for self-attention
                         cross_attn_mask: Optional[Tensor] = None, # mask used for cross-attention
            ):
        # cross attention
        if self.key_aware_type is not None:
            raise NotImplementedError("Unknown key_aware_type: {}".format(self.key_aware_type))
        
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                               tgt_reference_points.transpose(0, 1).contiguous(),
                               memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index, memory_key_padding_mask).transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        return tgt

    def forward(self, tgt: Optional[Tensor],  # nq, bs, d_model
                      tgt_query_pos: Optional[Tensor] = None, # pos for query. MLP(Sine(pos))
                      tgt_query_sine_embed: Optional[Tensor] = None, # pos for query. Sine(pos)
                      tgt_key_padding_mask: Optional[Tensor] = None,
                      tgt_reference_points: Optional[Tensor] = None, # nq, bs, 5
                      memory: Optional[Tensor] = None, # hw, bs, d_model
                      memory_key_padding_mask: Optional[Tensor] = None,
                      memory_level_start_index: Optional[Tensor] = None, # num_levels
                      memory_spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                      memory_pos: Optional[Tensor] = None, # pos for memory
                      self_attn_mask: Optional[Tensor] = None, # mask used for self-attention
                      cross_attn_mask: Optional[Tensor] = None, # mask used for cross-attention
            ):
        # breakpoint()
        for funcname in self.module_seq:
            if funcname == 'ffn':
                tgt = self.forward_ffn(tgt)
            elif funcname == 'ca':
                tgt = self.forward_ca(tgt, tgt_query_pos, tgt_query_sine_embed, \
                    tgt_key_padding_mask, tgt_reference_points, \
                        memory, memory_key_padding_mask, memory_level_start_index, \
                            memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask)
            elif funcname == 'sa':
                tgt = self.forward_sa(tgt, tgt_query_pos, tgt_query_sine_embed, \
                    tgt_key_padding_mask, tgt_reference_points, \
                        memory, memory_key_padding_mask, memory_level_start_index, \
                            memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask)
            else:
                raise ValueError('unknown funcname {}'.format(funcname))

        return tgt

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class OBBDinoTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, 
                    return_intermediate=False, 
                    d_model=256, query_dim=4, 
                    modulate_hw_attn=False,
                    num_feature_levels=1,
                    deformable_decoder=False,
                    decoder_query_perturber=None,
                    dec_layer_number=None, # number of queries each layer in decoder
                    rm_dec_query_scale=False,
                    dec_layer_share=False,
                    dec_layer_dropout_prob=None,
                    use_detached_boxes_dec_out=False,
                    **kwargs,
                    ):
        super().__init__()
        # prepare layers
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers, layer_share=dec_layer_share)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4, 5], "query_dim should be 2/4/5 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out

        
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(d_model, d_model, d_model, 2)
        else:
            self.query_pos_sine_scale = None

        if rm_dec_query_scale:
            self.query_scale = None
        else:
            raise NotImplementedError
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        # NOTE: in mmdetection, we can't use after class initial to 
        # change the attribute
        # self.bbox_embed = None
        # self.class_embed = None

        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.deformable_decoder = deformable_decoder

        if not deformable_decoder and modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
        else:
            self.ref_anchor_head = None

        self.decoder_query_perturber = decoder_query_perturber
        self.box_pred_damping = None

        self.dec_layer_number = dec_layer_number
  
            
        self.dec_layer_dropout_prob = dec_layer_dropout_prob
        if dec_layer_dropout_prob is not None:
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == num_layers
            for i in dec_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self.rm_detach = None

    def forward(self, tgt, memory,
                      tgt_mask: Optional[Tensor] = None,
                      memory_mask: Optional[Tensor] = None,
                      tgt_key_padding_mask: Optional[Tensor] = None,
                      memory_key_padding_mask: Optional[Tensor] = None,
                      pos: Optional[Tensor] = None,
                      refpoints_unsigmoid: Optional[Tensor] = None, # num_queries, bs, 2
                      # for memory
                      level_start_index: Optional[Tensor] = None, # num_levels
                      spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                      valid_ratios: Optional[Tensor] = None,
                      fc_reg = None,
                      fc_cls = None):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4/5
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        # breakpoint()
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]  
        # breakpoint()
        for layer_id, layer in enumerate(self.layers):
            # preprocess ref points
    
            if self.deformable_decoder:
                if reference_points.shape[-1] == 5:
                    reference_points_input = reference_points[:, :, None] \
                                            * torch.cat([valid_ratios, valid_ratios, 
                                                         torch.ones((*valid_ratios.shape[:-1], 1))
                                                         .to(valid_ratios.device)], -1)[None, :] # nq, bs, nlevel, 5
                else:
                    assert reference_points.shape[-1] == 2
                    reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]
                query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :]) # nq, bs, 256*2 
            else:
                query_sine_embed = gen_sineembed_for_position(reference_points) # nq, bs, 256*2
                reference_points_input = None

            # conditional query
            # import ipdb; ipdb.set_trace()
            raw_query_pos = self.ref_point_head(query_sine_embed) # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos
            if not self.deformable_decoder:
                query_sine_embed = query_sine_embed[..., :self.d_model] * self.query_pos_sine_scale(output)

            # modulated HW attentions
            if not self.deformable_decoder and self.modulate_hw_attn:
                refHW_cond = self.ref_anchor_head(output).sigmoid() # nq, bs, 2
                query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / reference_points[..., 2]).unsqueeze(-1)
                query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / reference_points[..., 3]).unsqueeze(-1)

            # main process
            # import ipdb; ipdb.set_trace()
           
           
            # breakpoint()
            output = layer(
                tgt = output,
                tgt_query_pos = query_pos,
                tgt_query_sine_embed = query_sine_embed,
                tgt_key_padding_mask = tgt_key_padding_mask,
                tgt_reference_points = reference_points_input,

                memory = memory,
                memory_key_padding_mask = memory_key_padding_mask,
                memory_level_start_index = level_start_index,
                memory_spatial_shapes = spatial_shapes,
                memory_pos = pos,

                self_attn_mask = tgt_mask,
                cross_attn_mask = memory_mask)

            # iter update
            if fc_reg is not None:
                # breakpoint()
                
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = fc_reg[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

    
                reference_points = new_reference_points.detach()
                ref_points.append(new_reference_points)


            intermediate.append(self.norm(output))
          

        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points]
        ]

@TRANSFORMER.register_module()
class OBBDinoTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, 
                       num_queries=1000, 
                       num_encoder_layers=6,
                       num_decoder_layers=6, 
                       dim_feedforward=2048, dropout=0.0,
                       activation="relu", normalize_before=False,
                       return_intermediate_dec=True, query_dim=4,
                       num_patterns=0,
                       modulate_hw_attn=True,
                       # for deformable encoder
                       deformable_encoder=True,
                       deformable_decoder=True,
                       num_feature_levels=4,
                       enc_n_points=5,
                       dec_n_points=5,
                       # init query
                       learnable_tgt_init=True,
                       decoder_query_perturber=None,
                       add_channel_attention=False,
                       add_pos_value=False,
                       random_refpoints_xy=False,
                       # two stage
                       # ['no', 'standard', 'early', 'combine', 'enceachlayer', 'enclayer1']
                       two_stage_type='standard', 
                       two_stage_pat_embed=0,
                       two_stage_add_query_num=0,
                       two_stage_learn_wh=False,
                       two_stage_keep_all_tokens=False,
                       # evo of #anchors
                       dec_layer_number=None,
                       rm_enc_query_scale=True,
                       rm_dec_query_scale=True,
                       rm_self_attn_layers=None,
                       key_aware_type=None,
                       # layer share
                       layer_share_type=None,
                       # for detach
                       rm_detach=None,
                       decoder_sa_type='sa', 
                       module_seq=['sa', 'ca', 'ffn'],
                       # for dn
                       embed_init_tgt=True,
                       use_detached_boxes_dec_out=False,
            ):
        """
        Args:
            d_model (int): The dimension of the model. Used in both encoder and decoder layers.
            nhead (int): The number of attention heads. Determines the number of parallel attention layers.
            num_queries (int): The number of queries. Used in both the initialization and forward 
            function to handle the number of queries.
            num_encoder_layers (int): The number of encoder layers. Specifies the depth of the encoder.
            num_decoder_layers (int): The number of decoder layers. Specifies the depth of the decoder.
            dim_feedforward (int): The dimension of the feedforward network. Applied in the feedforward 
            layers of both encoder and decoder.
            dropout (float): The dropout rate. Used to prevent overfitting in the encoder and decoder layers.
            activation (str): The activation function. Used in the feedforward layers of both encoder 
            and decoder.
            normalize_before (bool): Whether to normalize before the attention layer. Applied in encoder 
            and decoder normalization.
            return_intermediate_dec (bool): Whether to return intermediate decoder layers. Used in the 
            decoder for returning intermediate outputs.
            query_dim (int): The dimension of the query. Applied in the decoder for handling query 
            dimensions.
            num_patterns (int): The number of patterns. Used in two-stage process to handle pattern 
            embeddings.
            modulate_hw_attn (bool): Whether to modulate height and width attention. Applied in the 
            decoder to adjust attention.
            deformable_encoder (bool): Whether to use a deformable encoder. Specifies the type of encoder 
            layer used.
            deformable_decoder (bool): Whether to use a deformable decoder. Specifies the type of decoder 
            layer used.
            num_feature_levels (int): The number of feature levels. Used in the encoder and decoder 
            to handle multi-scale features.
            enc_n_points (int): The number of sampling points for the encoder. Determines sampling 
            points in deformable encoder.
            dec_n_points (int): The number of sampling points for the decoder. Determines sampling 
            points in deformable decoder.
            learnable_tgt_init (bool): Whether to learn the initial target embedding. Applied in the 
            initialization of target embeddings.
            decoder_query_perturber (object): Perturber for the decoder queries. Used to add noise or 
            perturbations to the decoder queries.
            add_channel_attention (bool): Whether to add channel attention. Applied in the encoder for 
            channel-wise attention.
            add_pos_value (bool): Whether to add positional value. Applied in positional embedding layers.
            random_refpoints_xy (bool): Whether to randomly initialize reference points in xy. Used in 
            initialization of reference points.
            two_stage_type (str): The type of two-stage process. Determines the behavior of the 
            two-stage training.
            two_stage_pat_embed (int): The number of pattern embeddings for the two-stage process. Used 
            in two-stage process to add pattern embeddings.
            two_stage_add_query_num (int): The number of additional queries for the two-stage process. 
            Used to extend the number of queries in two-stage.
            two_stage_learn_wh (bool): Whether to learn width and height for the two-stage process. 
            Applied in two-stage to learn bounding box dimensions.
            two_stage_keep_all_tokens (bool): Whether to keep all tokens in the two-stage process. 
            Used to retain tokens through two-stage process.
            dec_layer_number (int): The layer number for the decoder. Used to configure specific layers 
            in the decoder.
            rm_enc_query_scale (bool): Whether to remove encoder query scaling. Applied in encoder 
            configuration to handle scaling.
            rm_dec_query_scale (bool): Whether to remove decoder query scaling. Applied in decoder 
            configuration to handle scaling.
            rm_self_attn_layers (list): The layers to remove self-attention. Specifies which layers to 
            skip self-attention in.
            key_aware_type (str): The type of key-aware attention. Used to configure key-aware 
            mechanisms in the decoder.
            layer_share_type (str): The type of layer sharing. Determines sharing strategy for layers 
            in encoder and decoder.
            rm_detach (bool): Whether to remove detach. Applied to detach gradients in specific parts 
            of the network.
            decoder_sa_type (str): The type of self-attention for the decoder. Specifies the 
            self-attention mechanism used in the decoder.
            module_seq (list): The sequence of modules. Defines the order of operations in the decoder.
            embed_init_tgt (bool): Whether to embed the initial target. Used in the initialization of 
            target embeddings.
            use_detached_boxes_dec_out (bool): Whether to use detached boxes in decoder output. 
            Applied in the post-processing of decoder outputs.
        """

        super().__init__()
        # breakpoint()
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.deformable_encoder = deformable_encoder
        self.deformable_decoder = deformable_decoder
        self.two_stage_keep_all_tokens = two_stage_keep_all_tokens
        self.num_queries = num_queries
        self.random_refpoints_xy = random_refpoints_xy
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out
        assert query_dim == 4

        if num_feature_levels > 1:
            assert deformable_encoder, "only support deformable_encoder for num_feature_levels > 1"
     

        assert layer_share_type in [None, 'encoder', 'decoder', 'both']
        if layer_share_type in ['encoder', 'both']:
            enc_layer_share = True
        else:
            enc_layer_share = False
        if layer_share_type in ['decoder', 'both']:
            dec_layer_share = True
        else:
            dec_layer_share = False
        assert layer_share_type is None

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']

        # choose encoder layer type
        if deformable_encoder:
            encoder_layer = OBBDinoTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points, 
                                                          add_channel_attention=add_channel_attention)
        else:
            raise NotImplementedError
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = OBBDinoTransformerEncoder(
            encoder_layer, num_encoder_layers, 
            encoder_norm, d_model=d_model, 
            num_queries=num_queries,
            deformable_encoder=deformable_encoder, 
            enc_layer_share=enc_layer_share, 
            two_stage_type=two_stage_type
        )

        # choose decoder layer type
        if deformable_decoder:
            decoder_layer = OBBDinoTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points, 
                                                          key_aware_type=key_aware_type,
                                                          decoder_sa_type=decoder_sa_type,
                                                          module_seq=module_seq)

        else:
            raise NotImplementedError

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = OBBDinoTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                        return_intermediate=return_intermediate_dec,
                                        d_model=d_model, query_dim=query_dim, 
                                        modulate_hw_attn=modulate_hw_attn,
                                        num_feature_levels=num_feature_levels,
                                        deformable_decoder=deformable_decoder,
                                        decoder_query_perturber=decoder_query_perturber, 
                                        dec_layer_number=dec_layer_number, rm_dec_query_scale=rm_dec_query_scale,
                                        dec_layer_share=dec_layer_share,
                                        use_detached_boxes_dec_out=use_detached_boxes_dec_out
                                        )

        self.d_model = d_model
        self.embed_dims = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries # useful for single stage model only
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0

        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
            else:
                self.level_embed = None
        
        self.learnable_tgt_init = learnable_tgt_init
        assert learnable_tgt_init, "why not learnable_tgt_init"
        self.embed_init_tgt = embed_init_tgt
        if (two_stage_type == 'standard' and embed_init_tgt) or (two_stage_type == 'no'):
            self.tgt_embed = nn.Embedding(self.num_queries, d_model)
            nn.init.normal_(self.tgt_embed.weight.data)
        else:
            NotImplementedError
            
        # for two stage
        self.two_stage_type = two_stage_type
        self.two_stage_pat_embed = two_stage_pat_embed
        self.two_stage_add_query_num = two_stage_add_query_num
        self.two_stage_learn_wh = two_stage_learn_wh
        assert two_stage_type == 'standard', "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type =='standard':
            # anchor selection at the output of encoder
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)      
            
            if two_stage_pat_embed > 0:
                self.pat_embed_for_2stage = nn.Parameter(torch.Tensor(two_stage_pat_embed, d_model))
                nn.init.normal_(self.pat_embed_for_2stage)

            if two_stage_add_query_num > 0:
                self.tgt_embed = nn.Embedding(self.two_stage_add_query_num, d_model)

            if two_stage_learn_wh:
                # import ipdb; ipdb.set_trace()
                self.two_stage_wh_embedding = nn.Embedding(1, 2)
            else:
                self.two_stage_wh_embedding = None
      

        self._reset_parameters()

        self.rm_self_attn_layers = rm_self_attn_layers
        self.rm_detach = rm_detach
       
        self.decoder.rm_detach = rm_detach

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

        if self.two_stage_learn_wh:
            nn.init.constant_(self.two_stage_wh_embedding.weight, math.log(0.05 / (1 - 0.05)))


    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, 5)
        
        if self.random_refpoints_xy:
            # import ipdb; ipdb.set_trace()
            self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False
    
    def forward(self, srcs, masks, refpoint_embed, pos_embeds, tgt, attn_mask=None, fc_reg=None, fc_cls=None, fc_enc_reg=None, fc_enc_cls=None, vis_metas=None):                                                                 
        """decoder forward in DINO, "refpoint_embed" and "tgt" is the dn component, attn_mask is also provided accorddingly.
        Input:
            - srcs: List of multi features [bs, ci, hi, wi]
            - masks: List of multi masks [bs, hi, wi]
            - refpoint_embed: [bs, num_dn, 5]. None in infer
            - pos_embeds: List of multi pos embeds [bs, ci, hi, wi]
            - tgt: [bs, num_dn, d_model]. None in infer
            
        """
        # breakpoint()
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)                # bs, hw, c
            mask = mask.flatten(1)                              # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)    # bs, hw, c
            # 多个feature level的feature需要加入level embedding
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)    # bs, \sum{hxw}, c 
        mask_flatten = torch.cat(mask_flatten, 1)   # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) # bs, \sum{hxw}, c 
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # two stage
        enc_topk_proposals = enc_refpoint_embed = None

        #########################################################
        # Begin Encoder
        #########################################################
        # memory: [bs, hw, c]
        # enc_intermediate_output: [n_enc, bs, nq, c]
        # enc_intermediate_refpoints: [n_enc, bs, nq, c]
        memory, enc_intermediate_output, enc_intermediate_refpoints = self.encoder(
                src_flatten, 
                pos=lvl_pos_embed_flatten, 
                level_start_index=level_start_index, 
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios,
                key_padding_mask=mask_flatten,
                ref_token_index=enc_topk_proposals, # bs, nq 
                ref_token_coord=enc_refpoint_embed, # bs, nq, 5
                )
    

        if self.two_stage_type =='standard':
            """DINO take the standard two-stage manner to generate the initial object queries"""
            input_hw = None
            output_memory, output_proposals = gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes, input_hw)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
            
           
            enc_outputs_class_unselected = fc_enc_cls(output_memory)
            enc_outputs_coord_unselected = fc_enc_reg(output_memory) + output_proposals # [bs, \sum{hw}, 5] unsigmoid, output_proposlas maybe have inf value
            topk = self.num_queries
            topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1] # [bs, topk] is the index value
            

            # gather boxes
            refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 5)) # unsigmoid [bs, topk, 5]
            refpoint_embed_ = refpoint_embed_undetach.detach()
            init_box_proposal = torch.gather(output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 5)).sigmoid() # sigmoid [bs, topk, 5]

            # gather tgt
            tgt_undetach = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))
            if self.embed_init_tgt:
                tgt_ = self.tgt_embed.weight[:self.num_queries, None, :].repeat(1, bs, 1).transpose(0, 1) # [bs, topk, d_model]
            else:
                NotImplementedError
                
            if refpoint_embed is not None:
                refpoint_embed=torch.cat([refpoint_embed,refpoint_embed_],dim=1)    
                tgt = torch.cat([tgt, tgt_],dim=1)
            else:
                # tgt: [bs, num_query, d_model]
                # refpoint_embed: [bs, num_query, d_model]
                refpoint_embed, tgt = refpoint_embed_, tgt_

        elif self.two_stage_type == 'no':
            tgt_ = self.tgt_embed.weight[:self.num_queries, None, :].repeat(1, bs, 1).transpose(0, 1)                 # nq, bs, d_model
            refpoint_embed_ = self.refpoint_embed.weight[:self.num_queries, None, :].repeat(1, bs, 1).transpose(0, 1) # nq, bs, 5

            if refpoint_embed is not None:
                refpoint_embed = torch.cat([refpoint_embed,refpoint_embed_],dim=1)
                tgt = torch.cat([tgt, tgt_],dim=1)
            else:
                # tgt: [bs, num_query, d_model]
                # refpoint_embed: [bs, num_query, d_model]
                refpoint_embed, tgt = refpoint_embed_, tgt_

            if self.num_patterns > 0:
                tgt_embed = tgt.repeat(1, self.num_patterns, 1)
                refpoint_embed = refpoint_embed.repeat(1, self.num_patterns, 1)
                tgt_pat = self.patterns.weight[None, :, :].repeat_interleave(self.num_queries, 1) # 1, n_q*n_pat, d_model
                tgt = tgt_embed + tgt_pat

            init_box_proposal = refpoint_embed_.sigmoid()

        else:
            raise NotImplementedError("unknown two_stage_type {}".format(self.two_stage_type))
      

        #########################################################
        # Begin Decoder
        #########################################################
        # hs: [n_dec, bs, nq, d_model], 这里的hs和references已经包含了dn_part和two stage产生的hs和references
        # references: [n_dec+1, bs, nq, query_dim], 注意这里的references比decoder layer多了一个
        hs, references = self.decoder(
                                tgt=tgt.transpose(0, 1), 
                                memory=memory.transpose(0, 1), 
                                memory_key_padding_mask=mask_flatten, 
                                pos=lvl_pos_embed_flatten.transpose(0, 1),
                                refpoints_unsigmoid=refpoint_embed.transpose(0, 1), 
                                level_start_index=level_start_index, 
                                spatial_shapes=spatial_shapes,
                                valid_ratios=valid_ratios,tgt_mask=attn_mask,
                                fc_reg=fc_reg,
                                fc_cls=fc_cls)
      


        #########################################################
        # Begin postprocess
        #########################################################     
        if self.two_stage_type == 'standard':
            # hs_enc: [n_enc+1, bs, nq, d_model] or [1, bs, nq, d_model] or [n_enc, bs, nq, d_model] or None
            # ref_enc: [n_enc+1, bs, nq, query_dim] or [1, bs, nq, query_dim] or [n_enc, bs, nq, d_model] or None
            hs_enc = tgt_undetach.unsqueeze(0)  # [1, 2, 900, 256], 这里只有two stage产生的topk的encoder embedding
            ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0) # [1, 2, 900, 5]
        else:
            hs_enc = ref_enc = None 

        # hs: (n_dec, bs, nq, d_model)
        # references: sigmoid coordinates. (n_dec+1, bs, bq, 5)
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or None
        # ref_enc: sigmoid coordinates. (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or None
        return hs, references, hs_enc, ref_enc, init_box_proposal   # init_box_proposal: [2, 900, 5]


    def forward_with_query(self, srcs, masks, refpoint_embed_, pos_embeds, tgt_, attn_mask=None, fc_reg=None, fc_cls=None, fc_enc_reg=None, fc_enc_cls=None):                                                                
        """decoder forward with provided "refpoint_embed" and "tgt".
        Input:
            - srcs: List of multi features [bs, ci, hi, wi]
            - masks: List of multi masks [bs, hi, wi]
            - refpoint_embed: [num_consistency_query, 5]. 
            - pos_embeds: List of multi pos embeds [bs, ci, hi, wi]
            - tgt: [num_consistency_query, d_model].
        """
        # breakpoint()
        # import ipdb;ipdb.set_trace()
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)                # bs, hw, c
            mask = mask.flatten(1)                              # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)    # bs, hw, c
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)    # bs, \sum{hxw}, c 
        mask_flatten = torch.cat(mask_flatten, 1)   # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) # bs, \sum{hxw}, c 
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # two stage
        enc_topk_proposals = enc_refpoint_embed = None

    
        # memory: [bs, hw, c]
        # enc_intermediate_output: [n_enc, bs, nq, c]
        # enc_intermediate_refpoints: [n_enc, bs, nq, c]
        memory, enc_intermediate_output, enc_intermediate_refpoints = self.encoder(
                src_flatten, 
                pos=lvl_pos_embed_flatten, 
                level_start_index=level_start_index, 
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios,
                key_padding_mask=mask_flatten,
                ref_token_index=enc_topk_proposals, # bs, nq 
                ref_token_coord=enc_refpoint_embed,)
    

        tgt = tgt_[:, None, :].repeat(1, bs, 1).transpose(0, 1)                         # (num_consistency_query, bs, d_model)
        refpoint_embed = refpoint_embed_[:, None, :].repeat(1, bs, 1).transpose(0, 1)   # (num_consistency_query, bs, 5)

        # hs: [n_dec, bs, nq, d_model]
        # references: [n_dec+1, bs, nq, 5], sigmoid normalized (cx, cy, w, h) format
        hs, references = self.decoder(
                                tgt=tgt.transpose(0, 1), 
                                memory=memory.transpose(0, 1), 
                                memory_key_padding_mask=mask_flatten, 
                                pos=lvl_pos_embed_flatten.transpose(0, 1),
                                refpoints_unsigmoid=refpoint_embed.transpose(0, 1), 
                                level_start_index=level_start_index, 
                                spatial_shapes=spatial_shapes,
                                valid_ratios=valid_ratios, tgt_mask=attn_mask,
                                fc_reg=fc_reg,
                                fc_cls=fc_cls)
        return hs, references
 