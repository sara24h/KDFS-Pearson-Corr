import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
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

def compute_filter_correlation(filters, mask_weight, gumbel_temperature=1.0):
    device = filters.device
    num_filters = filters.shape[0]
    
    # محاسبه احتمال نگهداری فیلتر (m_l^i)
    mask_probs = F.gumbel_softmax(logits=mask_weight, tau=gumbel_temperature, hard=False, dim=1)[:, 1]
    mask_probs = mask_probs.squeeze(-1).squeeze(-1)
    
    # میانگین نگه‌داشتن فیلترها در این لایه (مورد نیاز برای L_sparsity)
    current_layer_retention = torch.mean(mask_probs)
    
    if num_filters < 2:
        # اگر فیلتر کافی نباشد، زیان صفر و نرخ نگه‌داشتن فعلی برگردانده می‌شود.
        return torch.tensor(0.0, device=device, requires_grad=True), current_layer_retention
    
    # --- محاسبه Score_l,i (فرمول ۶) ---
    filters_flat = filters.view(num_filters, -1)
    W = filters_flat.shape[1]

    # نرمال‌سازی فیلترها
    mean = filters_flat.mean(dim=1, keepdim=True)
    centered = filters_flat - mean
    variance = (centered ** 2).mean(dim=1, keepdim=True)
    std = torch.sqrt(variance + 1e-8)
    filters_normalized = centered / std

    # ماتریس همبستگی پیرسون
    corr_matrix = torch.matmul(filters_normalized, filters_normalized.t()) / W
    corr_matrix = torch.clamp(corr_matrix, -1.0, 1.0)
    
    # محاسبه امتیاز افزونگی (Score_l,i)
    mask = ~torch.eye(num_filters, dtype=torch.bool, device=device)
    off_diag_corrs = corr_matrix[mask].view(num_filters, num_filters - 1)
    correlation_scores = (off_diag_corrs ** 2).mean(dim=1)

    # حذف محاسبه mean_upper_corr که فقط برای نظارت بود
    # triu_indices = torch.triu_indices(num_filters, num_filters, offset=1, device=device)
    # upper_corr_values = corr_matrix[triu_indices[0], triu_indices[1]]
    # mean_upper_corr = upper_corr_values.mean().item() 

    if mask_probs.shape[0] != correlation_scores.shape[0]:
        return torch.tensor(0.0, device=device, requires_grad=True), current_layer_retention

    # L_corr (زیان همبستگی - فرمول ۷)
    correlation_loss = torch.mean(correlation_scores * mask_probs)
    
    # بازگشت L_corr و نرخ نگه‌داشت فعلی
    return correlation_loss, current_layer_retention
    
class MaskLoss(nn.Module):
    def __init__(self, target_retention=0.75, lambda_sparse=10.0, penalty_mode='asymmetric'):
        """
        Args:
            target_retention (ρ): نرخ نگه‌داشت هدف (0.75 = 75%)
            lambda_sparse: ضریب اهمیت (باید بزرگ باشد: 10-100)
            penalty_mode: نوع جریمه
                'mse': (current - target)^2
                'asymmetric': جریمه شدیدتر برای نگه‌داشت بیش از حد
                'hard': جریمه نمایی
        """
        super(MaskLoss, self).__init__()
        self.rho = target_retention
        self.lambda_sparse = lambda_sparse
        self.penalty_mode = penalty_mode
   
    def forward(self, model):
        total_corr_loss = 0.0
        total_sparsity_loss = 0.0
        num_layers = 0
        device = next(model.parameters()).device
        
        for m in model.mask_modules:
            if isinstance(m, SoftMaskedConv2d):
                filters = m.weight
                mask_weight = m.mask_weight
                gumbel_temperature = m.gumbel_temperature
                
                l_corr, current_retention = compute_filter_correlation(
                    filters, mask_weight, gumbel_temperature
                )
                
                total_corr_loss += l_corr
                
                # --- جریمه‌های مختلف برای کنترل نرخ نگه‌داشت ---
                if self.penalty_mode == 'mse':
                    # حالت معمولی: فاصله مربعی
                    layer_sparsity_loss = (current_retention - self.rho) ** 2
                
                elif self.penalty_mode == 'asymmetric':
                    # جریمه شدیدتر برای نگه‌داشت بیش از حد
                    diff = current_retention - self.rho
                    if diff > 0:  # اگر بیش از حد نگه داشته
                        layer_sparsity_loss = (diff ** 2) * 5.0  # 5 برابر جریمه
                    else:  # اگر کمتر نگه داشته
                        layer_sparsity_loss = (diff ** 2) * 1.0
                
                elif self.penalty_mode == 'hard':
                    # جریمه نمایی برای اجبار به هدف
                    diff = torch.abs(current_retention - self.rho)
                    layer_sparsity_loss = torch.exp(diff * 10) - 1
                
                elif self.penalty_mode == 'clipped':
                    # فقط وقتی جریمه که از هدف بیشتر نگه دارد
                    diff = current_retention - self.rho
                    layer_sparsity_loss = torch.relu(diff) ** 2  # فقط مثبت
                
                total_sparsity_loss += layer_sparsity_loss
                num_layers += 1
        
        if num_layers == 0:
            warnings.warn("No maskable layers found in the model.")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        avg_corr_loss = total_corr_loss / num_layers
        avg_sparsity_loss = total_sparsity_loss / num_layers
        
        # زیان نهایی
        final_loss = avg_corr_loss + (self.lambda_sparse * avg_sparsity_loss)
        
        return final_loss

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
