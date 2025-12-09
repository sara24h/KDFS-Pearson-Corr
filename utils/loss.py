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
        print('less then 2 filters')
    
    filters_flat = filters.view(num_filters, -1)

    variance = torch.var(filters_flat, dim=1)
    zero_variance_indices = torch.where(variance == 0)[0]
    if len(zero_variance_indices) > 0:
        warnings.warn(f"{len(zero_variance_indices)} filters have zero variance.")
    

    mean = torch.mean(filters_flat, dim=1, keepdim=True)
    centered = filters_flat - mean
    std = torch.std(filters_flat, dim=1, keepdim=True)
    #epsilon = 1e-4 
    filters_normalized = centered / (std)

    #norm = torch.norm(filters_normalized, dim=1, keepdim=True)
    #filters_normalized = filters_normalized / (norm + epsilon)
    
    if torch.isnan(filters_normalized).any():
        warnings.warn("Normalized filters contain NaN.")
    if torch.isinf(filters_normalized).any():
        warnings.warn("Normalized filters contain Inf values.")
    
    corr_matrix = torch.matmul(filters_normalized, filters_normalized.t())
    triu_indices = torch.triu_indices(num_filters, num_filters, offset=1, device=device)
    upper_corr_values = corr_matrix[triu_indices[0], triu_indices[1]]
    mean_upper_corr = upper_corr_values.mean().item()
    
    if torch.isnan(corr_matrix).any():
        warnings.warn("Correlation matrix contains NaN values.")
    if torch.isinf(corr_matrix).any():
        warnings.warn("Correlation matrix contains Inf values.")

    mask = ~torch.eye(num_filters, num_filters, device=filters.device).bool()
    
    correlation_scores = torch.sum((corr_matrix * mask.float())**2, dim=1)
    correlation_scores = correlation_scores / (num_filters - 1)
    
    if torch.isnan(correlation_scores).any():
        warnings.warn("Correlation scores contain NaN values.")
    if torch.isinf(correlation_scores).any():
        warnings.warn("Correlation scores contain Inf values.")

    mask_probs = F.gumbel_softmax(logits=mask_weight, tau=gumbel_temperature, hard=False, dim=1)[:, 1, :, :]
    mask_probs = mask_probs.squeeze(-1).squeeze(-1)
    
    if mask_probs.shape[0] != correlation_scores.shape[0]:
        warnings.warn("Shape mismatch between mask_probs and correlation_scores.")

    if mask_probs.shape[0] != correlation_scores.shape[0]:
        return torch.tensor(0.0, device=device, requires_grad=True), mean_upper_corr
        
    correlation_loss = torch.mean(correlation_scores * mask_probs)
    
    return correlation_loss, mean_upper_corr 

# در فایل utils/loss.py
class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()
    
    def forward(self, model):
        total_loss = 0.0
        num_layers = 0
        for m in model.mask_modules:
            if isinstance(m, SoftMaskedConv2d):
                # این زیان ساده، ماسک‌ها را به سمت صفر (حذف) سوق می‌دهد
                mask = torch.sigmoid(m.mask_weight)
                total_loss += torch.mean(mask)
                num_layers += 1
        
        if num_layers == 0:
            return torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)
        
        return total_loss / num_layers

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
