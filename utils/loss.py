import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.student.layer import SoftMaskedConv2d

class KDLoss(nn.Module):
    def __init__(self):
        super(KDLoss, self).__init__()
    def forward(self, logits_teacher, logits_student, temperature):
        kd_loss = F.binary_cross_entropy_with_logits(
            logits_student / temperature,
            torch.sigmoid(logits_teacher / temperature),
            reduction='mean'
        )
        return kd_loss
class RCLoss(nn.Module):
    def __init__(self):
        super(RCLoss, self).__init__()
    @staticmethod
    def rc(x):
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))
    def forward(self, x, y):
        return (self.rc(x) - self.rc(y)).pow(2).mean()
import warnings


def compute_filter_correlation(filters, mask_weight, gumbel_temperature=1.0):
    if torch.isnan(filters).any():
        warnings.warn("Filters contain NaN.")
    if torch.isinf(filters).any():
        warnings.warn("Filters contain Inf values.")
    if torch.isnan(mask_weight).any():
        warnings.warn("Mask weights contain NaN.")
    if torch.isinf(mask_weight).any():
        warnings.warn("Mask weights contain Inf values.")
    
    num_filters = filters.shape[0]
    if num_filters < 2:
        return torch.tensor(0.0, device=filters.device, requires_grad=True)
    
    filters_flat = filters.view(num_filters, -1)  # (C, W)
    W = filters_flat.shape[1]

    # محاسبه میانگین و مرکز کردن
    mean = filters_flat.mean(dim=1, keepdim=True)
    centered = filters_flat - mean

    # محاسبه واریانس biased (تقسیم بر W)
    variance = (centered ** 2).mean(dim=1, keepdim=True)
    std = torch.sqrt(variance + 1e-8)  # epsilon برای جلوگیری از تقسیم بر صفر

    # نرمال‌سازی
    filters_normalized = centered / std

    # 🔍 بررسی صریح NaN و Inf (همانطور که تو خواستی)
    if torch.isnan(filters_normalized).any():
        warnings.warn("Normalized filters contain NaN.")
    if torch.isinf(filters_normalized).any():
        warnings.warn("Normalized filters contain Inf values.")
    
    # ماتریس همبستگی پیرسون (biased): 1/W * sum(z_i * z_j)
    corr_matrix = torch.matmul(filters_normalized, filters_normalized.t()) / W

    if torch.isnan(corr_matrix).any():
        warnings.warn("Correlation matrix contains NaN values.")
    if torch.isinf(corr_matrix).any():
        warnings.warn("Correlation matrix contains Inf values.")

    # کلیپ برای اطمینان از بازه معتبر (اختیاری ولی توصیه‌شده)
    corr_matrix = torch.clamp(corr_matrix, -1.0, 1.0)

    # حذف قطر اصلی
    mask = ~torch.eye(num_filters, dtype=torch.bool, device=filters.device)
    off_diag_corrs = corr_matrix[mask].view(num_filters, num_filters - 1)
    correlation_scores = (off_diag_corrs ** 2).mean(dim=1)

    if torch.isnan(correlation_scores).any():
        warnings.warn("Correlation scores contain NaN values.")
    if torch.isinf(correlation_scores).any():
        warnings.warn("Correlation scores contain Inf values.")

    # احتمال نگه‌داشتن از mask
    mask_probs = F.gumbel_softmax(logits=mask_weight, tau=gumbel_temperature, hard=False, dim=1)[:, 1]
    mask_probs = mask_probs.squeeze(-1).squeeze(-1)

    if mask_probs.shape[0] != correlation_scores.shape[0]:
        warnings.warn("Shape mismatch between mask_probs and correlation_scores.")
        return torch.tensor(0.0, device=filters.device, requires_grad=True)

    correlation_loss = torch.mean(correlation_scores * mask_probs)
    return correlation_loss
    
class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()
   
    def forward(self, model):
  
        total_pruning_loss = 0.0
        num_layers = 0
        device = next(model.parameters()).device
       
        for m in model.mask_modules:
            if isinstance(m, SoftMaskedConv2d):
                filters = m.weight
                mask_weight = m.mask_weight
                gumbel_temperature = m.gumbel_temperature
                pruning_loss = compute_filter_correlation(filters, mask_weight, gumbel_temperature)
                total_pruning_loss += pruning_loss
                num_layers += 1
       
        if num_layers == 0:
            warnings.warn("No maskable layers found in the model.")
        total_loss = total_pruning_loss / num_layers
        return total_loss


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)
    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss
