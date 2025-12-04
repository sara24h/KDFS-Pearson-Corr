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

def compute_filter_correlation(filters):
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

    return corr_matrix, mean_upper_corr

# SIMPLIFIED ThresholdLoss - ONLY FIXED THRESHOLD
class ThresholdLoss(nn.Module):
    def __init__(self, threshold_value=0.7):
        """
        Initialize threshold-based regularization loss with a fixed threshold.
        
        Args:
            threshold_value (float): Fixed threshold value for all layers.
        """
        super(ThresholdLoss, self).__init__()
        self.register_buffer('threshold', torch.tensor(threshold_value))
    
    def forward(self, model):
        total_loss = 0.0
        num_layers = 0
        device = next(model.parameters()).device
        
        for m in model.mask_modules:
            if isinstance(m, SoftMaskedConv2d):
                filters = m.weight
                corr_matrix, _ = compute_filter_correlation(filters)
                
                # Get the fixed threshold
                threshold = self.threshold
                
                # Calculate max correlation for each filter (excluding self-correlation)
                num_filters = filters.shape[0]
                mask = ~torch.eye(num_filters, dtype=torch.bool, device=device)
                max_corrs, _ = torch.max(corr_matrix[mask].view(num_filters, num_filters - 1), dim=1)
                
                # Apply threshold and ReLU
                thresholded_corrs = F.relu(max_corrs - threshold)
                
                # Square the thresholded correlations
                squared_corrs = thresholded_corrs ** 2
                
                # Average across filters
                layer_loss = torch.mean(squared_corrs)
                
                total_loss += layer_loss
                num_layers += 1
        
        if num_layers == 0:
            warnings.warn("No maskable layers found in the model.")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Average across layers
        total_loss = total_loss / num_layers
        return total_loss

# SIMPLIFIED MaskLoss - ONLY FIXED THRESHOLD
class MaskLoss(nn.Module):
    def __init__(self, use_threshold=False, threshold_value=0.7):
        """
        Initialize mask loss with optional fixed threshold-based regularization.
        
        Args:
            use_threshold (bool): Whether to use threshold-based regularization.
            threshold_value (float): Fixed threshold value.
        """
        super(MaskLoss, self).__init__()
        self.use_threshold = use_threshold
        
        if use_threshold:
            self.threshold_loss = ThresholdLoss(threshold_value)
   
    def forward(self, model):
        device = next(model.parameters()).device
       
        if self.use_threshold:
            # Use threshold-based regularization
            return self.threshold_loss(model)
        else:
            # Original implementation (if you still want to keep it as an option)
            # Note: The original compute_filter_correlation signature is different.
            # You might need to adjust this part if you intend to use it.
            # For now, I'll assume you only use the threshold method.
            warnings.warn("Original MaskLoss implementation is not fully compatible. Using threshold loss.")
            return self.threshold_loss(model) # Fallback to threshold loss

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
