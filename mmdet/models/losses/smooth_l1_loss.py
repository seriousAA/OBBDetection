import torch
import torch.nn as nn
import math

from ..builder import LOSSES
from .utils import weighted_loss


@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


@weighted_loss
def l1_loss(pred, target):
    """L1 loss

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    return loss


@LOSSES.register_module()
class SmoothL1Loss(nn.Module):
    """Smooth L1 loss

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox


@LOSSES.register_module()
class L1Loss(nn.Module):
    """L1 loss

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(L1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox


@LOSSES.register_module()
class NLLoss(nn.Module):
    def __init__(self, beta: float, reduction: str = 'none'):
        """
        Args:
            beta (float): L1 to L2 change point.
                          For beta values < 1e-5, L1 loss is computed.
            reduction (str): Specifies the reduction to apply to the output.
                             'none' | 'mean' | 'sum'
                             'none': No reduction will be applied to the output.
                             'mean': The output will be averaged.
                             'sum': The output will be summed.
        """
        super(NLLoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum'], "Reduction must be one of 'none', 'mean', or 'sum'."
        self.beta = beta
        self.reduction = reduction

    def forward(
        self, 
        input: torch.Tensor, 
        input_std: torch.Tensor, 
        target: torch.Tensor, 
        iou_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): Input tensor of any shape.
            input_std (torch.Tensor): Standard deviation tensor with the same shape as input.
            target (torch.Tensor): Target tensor with the same shape as input.
            iou_weight (torch.Tensor): IoU weight tensor to weigh the loss per instance.
        
        Returns:
            torch.Tensor: The calculated loss.
        """
        mean = input
        sigma = input_std.sigmoid()
        sigma_sq = torch.square(sigma)

        # First term of the loss: (target - mean)^2 / (2 * sigma^2)
        first_term = torch.square(target - mean) / (2 * sigma_sq)

        # Second term of the loss: 0.5 * log(sigma^2)
        second_term = 0.5 * torch.log(sigma_sq)

        # Combine terms and add normalization constant for Gaussian distribution
        sum_before_iou = (first_term + second_term).sum(dim=1) + 2 * torch.log(
            2 * torch.tensor([math.pi]).cuda()
        )

        # Multiply by IoU weight
        loss_m = sum_before_iou * iou_weight

        # Apply reduction method
        if self.reduction == "mean":
            loss = loss_m.mean()
        elif self.reduction == "sum":
            loss = loss_m.sum()
        else:
            loss = loss_m  # No reduction

        return loss