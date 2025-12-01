import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

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


def compute_filter_correlation(filters, continuous_mask=None):
    """
    محاسبه correlation loss برای یک لایه
    
    Args:
        filters: وزن‌های فیلترها [out_channels, in_channels, kernel_size, kernel_size]
        continuous_mask: ماسک پیوسته از Gumbel Softmax [out_channels, 1, 1, 1]
                        اگر None باشد، همه فیلترها وزن یکسانی دارند
    
    Returns:
        correlation_loss: لاس همبستگی وزن‌دار شده با ماسک
        mean_abs_corr: میانگین قدرمطلق همبستگی‌های off-diagonal (برای لاگینگ)
    """
    device = filters.device
    num_filters = filters.shape[0]
    
    if num_filters < 2:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0
    
    # Flatten filters: [num_filters, C*K*K]
    filters_flat = filters.view(num_filters, -1)
    W = filters_flat.shape[1]
    
    # Normalize filters (zero-mean, unit variance)
    mean = filters_flat.mean(dim=1, keepdim=True)
    centered = filters_flat - mean
    variance = (centered ** 2).mean(dim=1, keepdim=True)
    std = torch.sqrt(variance + 1e-8)
    filters_normalized = centered / std
    
    # Compute correlation matrix
    corr_matrix = torch.matmul(filters_normalized, filters_normalized.t()) / W
    corr_matrix = torch.clamp(corr_matrix, -1.0, 1.0)
    
    # Extract off-diagonal correlations
    mask = ~torch.eye(num_filters, dtype=torch.bool, device=device)
    off_diag_corrs = corr_matrix[mask].view(num_filters, num_filters - 1)
    
    # Correlation scores per filter (squared mean)
    correlation_scores = (off_diag_corrs ** 2).mean(dim=1)  # [num_filters]
    
    # اگر continuous_mask داده شده، از آن استفاده می‌کنیم
    if continuous_mask is not None:
        mask_probs = continuous_mask.squeeze(-1).squeeze(-1).squeeze(-1)  # [num_filters]
        
        if mask_probs.shape[0] != correlation_scores.shape[0]:
            warnings.warn(f"Mask shape mismatch: {mask_probs.shape} vs {correlation_scores.shape}")
            return torch.tensor(0.0, device=device, requires_grad=True), 0.0
        
        # وزن‌دهی correlation scores با احتمال فعال بودن فیلتر
        correlation_loss = torch.mean(correlation_scores * mask_probs)
    else:
        # اگر ماسک نداریم، میانگین ساده
        correlation_loss = torch.mean(correlation_scores)
    
    # برای لاگینگ: میانگین قدرمطلق همبستگی‌های بالامثلثی
    triu_indices = torch.triu_indices(num_filters, num_filters, offset=1, device=device)
    upper_corr_values = corr_matrix[triu_indices[0], triu_indices[1]]
    mean_abs_corr = torch.upper_corr_values.mean().item()
    
    return correlation_loss, mean_abs_corr


class MaskLoss(nn.Module):
  
    def __init__(self):
        super(MaskLoss, self).__init__()
    
    def forward(self, model):
        """
        Args:
            model: مدل دانشجو که باید continuous_mask محاسبه شده داشته باشد
        """
        from model.student.layer import SoftMaskedConv2d
        
        total_correlation_loss = 0.0
        num_layers = 0
        device = next(model.parameters()).device
        
        for m in model.mask_modules:
            if isinstance(m, SoftMaskedConv2d):
                filters = m.weight
                
                # استفاده از continuous_mask که در forward محاسبه شده
                # اگر continuous_mask وجود نداشته باشد (ticket=True)، از None استفاده می‌شود
                continuous_mask = getattr(m, 'continuous_mask', None)
                
                correlation_loss, _ = compute_filter_correlation(filters, continuous_mask)
                total_correlation_loss += correlation_loss
                num_layers += 1
        
        if num_layers == 0:
            warnings.warn("No maskable layers found in the model.")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # میانگین روی همه لایه‌ها
        total_loss = total_correlation_loss / num_layers
        
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
