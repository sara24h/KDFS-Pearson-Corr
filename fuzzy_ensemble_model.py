import torch
import torch.nn as nn
import torch.nn.functional as F
from pruned_model.Resnet_final  import ResNet_50_pruned_hardfakevsreal


class FuzzyGatingNetwork(nn.Module):

    def __init__(self, num_models, num_features=2048):
        super().__init__()
        self.num_models = num_models
        
        # Fuzzy membership layers
        self.membership_layer = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_models)
        )
        
    def forward(self, features):
        """
        Args:
            features: [batch_size, num_features]
        Returns:
            weights: [batch_size, num_models] - Fuzzy membership weights
        """
        # محاسبه membership degrees
        memberships = self.membership_layer(features)
        
        # Softmax برای نرمال‌سازی
        weights = F.softmax(memberships, dim=1)
        
        return weights, memberships


class PrunedResNetEnsemble(nn.Module):
    """
    Ensemble از مدل‌های Pruned ResNet با Fuzzy Gating
    """
    
    def __init__(self, model_paths, masks_list, means_stds, num_features=2048):
        super().__init__()
        
        self.num_models = len(model_paths)
        self.means_stds = means_stds
        
        # بارگذاری مدل‌های فریز شده
        self.models = nn.ModuleList()
        for i, (path, masks) in enumerate(zip(model_paths, masks_list)):
            print(f"  Loading Model {i+1} from {path}...")
            
            # ساخت مدل
            model = ResNet_50_pruned_hardfakevsreal(masks)
            
            # بارگذاری وزن‌ها
            checkpoint = torch.load(path, map_location='cpu')
            model.load_state_dict(checkpoint)
            
            # فریز کردن
            for param in model.parameters():
                param.requires_grad = False
            
            model.eval()
            self.models.append(model)
            print(f"    ✓ Model {i+1} loaded and frozen")
        
        # Fuzzy Gating Network (trainable)
        self.gating_network = FuzzyGatingNetwork(self.num_models, num_features)
        print(f"\n  ✓ Fuzzy Gating Network created ({self.num_models} models)")
    
    def forward(self, x, return_details=False):
        """
        Args:
            x: [batch_size, 3, H, W]
            return_details: اگر True باشد، جزئیات بیشتری برمی‌گرداند
        
        Returns:
            output: [batch_size, 2] - Final ensemble output
            (optional) details: dict با اطلاعات اضافی
        """
        batch_size = x.size(0)
        device = x.device
        
        # 1. استخراج فیچرها و پیش‌بینی‌ها از هر مدل
        all_features = []
        all_logits = []
        
        with torch.no_grad():
            for i, model in enumerate(self.models):
                # نرمال‌سازی ورودی برای این مدل
                mean, std = self.means_stds[i]
                mean_tensor = torch.tensor(mean, device=device).view(1, 3, 1, 1)
                std_tensor = torch.tensor(std, device=device).view(1, 3, 1, 1)
                x_normalized = (x - mean_tensor) / std_tensor
                
                # استخراج فیچر و logits
                features, logits = model(x_normalized, return_features=True)
                all_features.append(features)
                all_logits.append(logits)
        
        # 2. Average pooling روی features برای gating
        pooled_features = torch.mean(torch.stack(all_features), dim=0)
        
        # 3. محاسبه Fuzzy Gating Weights
        weights, memberships = self.gating_network(pooled_features)
        
        # 4. ترکیب weighted logits
        stacked_logits = torch.stack(all_logits)  # [num_models, batch_size, 2]
        weights_expanded = weights.t().unsqueeze(-1)  # [num_models, batch_size, 1]
        
        weighted_logits = stacked_logits * weights_expanded
        ensemble_output = weighted_logits.sum(dim=0)  # [batch_size, 2]
        
        if return_details:
            details = {
                'weights': weights,
                'memberships': memberships,
                'individual_logits': all_logits,
                'features': all_features
            }
            return ensemble_output, details
        
        return ensemble_output


def train_fuzzy_ensemble(ensemble_model, train_loader, val_loader, 
                         num_epochs=20, learning_rate=1e-4, device='cuda'):
    """
    تابع آموزش Fuzzy Gating Network
    """
    
    ensemble_model = ensemble_model.to(device)
    
    # فقط پارامترهای gating_network آموزش داده می‌شوند
    optimizer = torch.optim.Adam(
        ensemble_model.gating_network.parameters(),
        lr=learning_rate
    )
    
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    print(f"   Device: {device}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Optimizer: Adam")
    print()
    
    for epoch in range(num_epochs):
        # ==================== TRAINING ====================
        ensemble_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = ensemble_model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Progress
            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch [{epoch+1}/{num_epochs}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # ==================== VALIDATION ====================
        ensemble_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = ensemble_model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Print epoch summary
        print(f"\n  Epoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"    Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"    Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': ensemble_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': avg_val_loss,
            }, 'best_fuzzy_ensemble.pt')
            print(f"    ✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
        
        print()
    
    print(f"✓ Training complete! Best Val Acc: {best_val_acc:.2f}%")
    
    return history


def analyze_gating_weights(ensemble_model, data_loader, device, num_samples=1000):
    """
    تحلیل Fuzzy Gating Weights
    """
    ensemble_model.eval()
    ensemble_model = ensemble_model.to(device)
    
    all_weights = []
    all_memberships = []
    samples_processed = 0
    
    print(f"\n🔍 Analyzing gating weights on {num_samples} samples...")
    
    with torch.no_grad():
        for images, _ in data_loader:
            if samples_processed >= num_samples:
                break
            
            images = images.to(device)
            _, details = ensemble_model(images, return_details=True)
            
            all_weights.append(details['weights'].cpu())
            all_memberships.append(details['memberships'].cpu())
            
            samples_processed += images.size(0)
    
    # Concatenate
    weights = torch.cat(all_weights, dim=0)[:num_samples]
    memberships = torch.cat(all_memberships, dim=0)[:num_samples]
    
    # Statistics
    print(f"\n  Gating Weights Statistics:")
    print(f"  {'Model':<10} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("  " + "-"*50)
    
    for i in range(weights.size(1)):
        model_weights = weights[:, i]
        print(f"  Model {i+1:<4} {model_weights.mean():.4f}    "
              f"{model_weights.std():.4f}    "
              f"{model_weights.min():.4f}    "
              f"{model_weights.max():.4f}")
    
    return weights.numpy(), memberships.numpy()
