import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import argparse
from tqdm import tqdm
from data.video_data import create_uadfv_dataloaders 

def parse_args():
    parser = argparse.ArgumentParser(description='Train Teacher Model on UADFV (Video Dataset)')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to UADFV folder (contains real/ and fake/ folders)')
    parser.add_argument('--teacher_dir', type=str, default='teacher_uadfv', help='Where to save teacher model')
    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames per video')
    parser.add_argument('--image_size', type=int, default=256, help='Input image size')
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--sampling', type=str, default='uniform', choices=['uniform', 'random'])
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.teacher_dir, exist_ok=True)

    train_loader, val_loader, test_loader = create_uadfv_dataloaders(
        root_dir=args.data_dir,
        num_frames=args.num_frames,
        image_size=args.image_size,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        ddp=False,
        sampling_strategy=args.sampling
    )

    print(f"Train videos: {len(train_loader.dataset)}")
    print(f"Val videos:   {len(val_loader.dataset)}")
    print(f"Test videos:  {len(test_loader.dataset)}")

    # مدل Teacher: ResNet50
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler() if device.type == 'cuda' else None

    best_acc = 0.0
    best_path = os.path.join(args.teacher_dir, 'teacher_resnet50_uadfv_best.pth')

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")

        for videos, labels in progress_bar:
            videos = videos.to(device)           # [B, T, C, H, W]
            labels = labels.to(device).float().unsqueeze(1)  # [B, 1]

            optimizer.zero_grad()

            with autocast(device.type == 'cuda'):
                b, t, c, h, w = videos.shape
                videos_flat = videos.view(b * t, c, h, w)   # [B*T, C, H, W]

                logits = model(videos_flat)                 # [B*T, 1]
                logits = logits.view(b, t)                  # [B, T]
                logits = logits.mean(dim=1, keepdim=True)   # میانگین روی فریم‌ها → [B, 1]

                loss = criterion(logits, labels)

            if device.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix({
                'loss': f'{total_loss/(progress_bar.n+1):.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for videos, labels in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                videos = videos.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                b, t, c, h, w = videos.shape
                videos_flat = videos.view(b * t, c, h, w)

                with autocast(device.type == 'cuda'):
                    logits = model(videos_flat).view(b, t).mean(dim=1, keepdim=True)
                    preds = (torch.sigmoid(logits) > 0.5).float()

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        print(f"\n[Epoch {epoch}] Val Accuracy: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved! Acc: {val_acc:.2f}%")

    final_path = os.path.join(args.teacher_dir, 'teacher_resnet50_uadfv_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"Training finished! Best Acc: {best_acc:.2f}%")
    print(f"Best model: {best_path}")


if __name__ == "__main__":
    main()
