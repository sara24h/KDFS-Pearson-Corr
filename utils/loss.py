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
    device = filters.device
    num_filters = filters.shape[0]
    if num_filters < 2:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0
    
    filters_flat = filters.view(num_filters, -1)
    W = filters_flat.shape[1]

    mean = filters_flat.mean(dim=1, keepdim=True)
    centered = filters_flat - mean
    variance = (centered ** 2).mean(dim=1, keepdim=True)
    std = torch.sqrt(variance + 1e-8)
    filters_normalized = centered / std

    corr_matrix = torch.matmul(filters_normalized, filters_normalized.t()) / W
    corr_matrix = torch.clamp(corr_matrix, -1.0, 1.0)

    triu_indices = torch.triu_indices(num_filters, num_filters, offset=1, device=device)
    upper_corr_values = corr_matrix[triu_indices[0], triu_indices[1]]
    mean_upper_corr = upper_corr_values.mean().item()  

    mask = ~torch.eye(num_filters, dtype=torch.bool, device=device)
    off_diag_corrs = corr_matrix[mask].view(num_filters, num_filters - 1)
    correlation_scores = (off_diag_corrs ** 2).mean(dim=1)

    mask_probs = F.gumbel_softmax(logits=mask_weight, tau=gumbel_temperature, hard=False, dim=1)[:, 1]
    mask_probs = mask_probs.squeeze(-1).squeeze(-1)

    if mask_probs.shape[0] != correlation_scores.shape[0]:
        return torch.tensor(0.0, device=device, requires_grad=True), mean_upper_corr

    correlation_loss = torch.mean(correlation_scores * mask_probs)
    return correlation_loss, mean_upper_corr
    
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
                pruning_loss, _ = compute_filter_correlation(filters, mask_weight, gumbel_temperature)
                total_pruning_loss += pruning_loss
                num_layers += 1
       
        if num_layers == 0:
            warnings.warn("No maskable layers found in the model.")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        total_loss = total_pruning_loss / num_layers
        return total_loss

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        # یک اشتباه تایپی کوچک در کد شما وجود داشت که اصلاح شد
        self.epsilon = epsilon 
        self.logsoftmax = nn.LogSoftmax(dim=1)
    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# <<<<<<<<<<<<<<<<<<<<<<<< کلاس جدید برای کنترل نرخ هرس <<<<<<<<<<<<<<<<<<<<<<
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class TargetSparsityLoss(nn.Module):
 
    def __init__(self, target_retention_ratio=0.8, coef_sparse=100.0):
        """
        Args:
            target_retention_ratio (float): میانگین هدف برای احتمال حفظ فیلترها.
                                           مثلاً 0.8 برای حفظ 80% فیلترها (یعنی 20% هرس).
            coef_sparse (float): ضریب این تابع زیان. هرچه بالاتر باشد، فشار برای رسیدن
                                 به نرخ هدف بیشتر است.
        """
        super().__init__()
        self.target_retention_ratio = target_retention_ratio
        self.coef_sparse = coef_sparse

    def forward(self, model):
        total_retention_rate = 0.0
        num_layers = 0
        device = next(model.parameters()).device

        for m in model.mask_modules:
            if isinstance(m, SoftMaskedConv2d):
                # احتمال حفظ فیلتر (برگ دوم Gumbel-Softmax)
                # m.mask_weight شامل logits برای Gumbel-Softmax است
                mask_probs = F.gumbel_softmax(m.mask_weight, tau=m.gumbel_temperature, hard=False)[:, 1]
                
                # میانگین احتمال حفظ فیلترها در این لایه
                layer_retention_rate = mask_probs.mean()
                total_retention_rate += layer_retention_rate
                num_layers += 1
        
        if num_layers == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # میانگین کل احتمالات حفظ فیلتر در تمام لایه‌ها
        current_avg_retention = total_retention_rate / num_layers
        
        # زیان مربوط به فاصله از نرخ هدف
        sparsity_loss = (current_avg_retention - self.target_retention_ratio).pow(2)
        
        return self.coef_sparse * sparsity_loss
