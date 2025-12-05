import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# ------------- SoftMaskedConv2d -------------
class SoftMaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # Learnable logits for Gumbel-Softmax: [C, 2]
        self.mask_logits = nn.Parameter(torch.Tensor(out_channels, 2))
        nn.init.kaiming_normal_(self.mask_logits)

        self.gumbel_temperature = 1.0
        self.mask = None  # For inference

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=nn.init.calculate_gain('relu'))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def compute_mask(self, hard=False):
        """Returns soft or hard mask of shape [C, 1, 1, 1]"""
        probs = F.gumbel_softmax(self.mask_logits, tau=self.gumbel_temperature, hard=hard, dim=1)
        mask = probs[:, 1].view(-1, 1, 1, 1)  # [C, 1, 1, 1]
        return mask

    def forward(self, x):
        mask = self.compute_mask(hard=False)  # Use soft mask during training
        self.mask = mask.detach()  # Optional: for logging
        masked_weight = self.weight * mask
        return F.conv2d(x, masked_weight, self.bias, self.stride, self.padding)

    def get_current_retention(self):
        soft_mask = self.compute_mask(hard=False)
        return soft_mask.mean().detach()

# ------------- Correlation on Filter Weights -------------
def compute_weight_correlation_loss(weight, mask_logits, gumbel_temp=1.0):
    """
    weight: [C, C_in, k, k]
    mask_logits: [C, 2] (for Gumbel-Softmax)
    Returns: L_corr (scalar), current_retention (scalar)
    """
    device = weight.device
    C = weight.shape[0]
    if C < 2:
        return torch.tensor(0.0, device=device, requires_grad=True), torch.tensor(1.0, device=device)

    # --- Soft mask (m_i) ---
    mask_probs = F.gumbel_softmax(mask_logits, tau=gumbel_temp, hard=False, dim=1)[:, 1]  # [C]
    current_retention = mask_probs.mean()

    # --- Normalize filters ---
    filters_flat = weight.view(C, -1)  # [C, D]
    D = filters_flat.shape[1]

    mean = filters_flat.mean(dim=1, keepdim=True)  # [C, 1]
    centered = filters_flat - mean  # [C, D]
    std = centered.std(dim=1, keepdim=True) + 1e-8  # [C, 1]
    normalized = centered / std  # [C, D]

    # --- Pearson correlation matrix ---
    corr = torch.matmul(normalized, normalized.t()) / D  # [C, C]
    corr = torch.clamp(corr, -1.0, 1.0)

    # --- max_{j≠i} |Corr(i, j)| ---
    mask = ~torch.eye(C, dtype=torch.bool, device=device)
    off_diag = corr[mask].view(C, C - 1)  # [C, C-1]
    max_corr = off_diag.abs().max(dim=1)[0]  # [C]

    # --- L_corr = sum_i m_i * max_{j≠i} |Corr| ---
    L_corr = (mask_probs * max_corr).mean()

    return L_corr, current_retention

# ------------- Mask Loss (Solution 4 Style) -------------
class MaskLoss(nn.Module):
    def __init__(self, target_retention=0.5, lambda_sparse=1.0):
        super().__init__()
        self.rho = target_retention
        self.lambda_sparse = lambda_sparse

    def forward(self, model):
        total_corr_loss = 0.0
        total_sparsity_loss = 0.0
        num_layers = 0
        device = next(model.parameters()).device

        # Assuming model has a list `mask_modules` containing SoftMaskedConv2d layers
        for m in model.mask_modules:
            if isinstance(m, SoftMaskedConv2d):
                L_corr, curr_ret = compute_weight_correlation_loss(
                    m.weight, m.mask_logits, m.gumbel_temperature
                )
                total_corr_loss += L_corr

                # L_sparsity = (current_retention - rho)^2
                sparsity_loss = (curr_ret - self.rho) ** 2
                total_sparsity_loss += sparsity_loss

                num_layers += 1

        if num_layers == 0:
            warnings.warn("No SoftMaskedConv2d layers found.")
            return torch.tensor(0.0, device=device, requires_grad=True)

        avg_corr = total_corr_loss / num_layers
        avg_sparsity = total_sparsity_loss / num_layers

        return avg_corr + self.lambda_sparse * avg_sparsity
