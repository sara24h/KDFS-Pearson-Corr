import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data.dataset import Dataset_selector
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal
from utils import meter

# DDP: تابع برای راه‌اندازی محیط توزیع‌شده
def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # مقداردهی اولیه گروه فرآیند
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # nccl برای GPUهای NVIDIA بهترین عملکرد را دارد

# DDP: تابع برای پاک‌سازی محیط توزیع‌شده
def cleanup_ddp():
    dist.destroy_process_group()

class Test:
    def __init__(self, args):
        self.args = args
        # DDP: دریافت rank و world_size از آرگومان‌ها
        self.rank = args.rank
        self.world_size = args.world_size
        self.device = torch.device(f"cuda:{self.rank}")

        self.dataset_dir = args.dataset_dir
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.arch = args.arch
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.sparsed_student_ckpt_path = args.sparsed_student_ckpt_path
        self.dataset_mode = args.dataset_mode
        self.result_dir = args.result_dir
        self.new_dataset_dir = getattr(args, 'new_dataset_dir', None)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! Please check GPU setup.")
            
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.new_test_loader = None
        self.student = None

    def dataload(self):
        # DDP: فقط در رنک 0 چاپ کن تا از تکرار جلوگیری شود
        if self.rank == 0:
            print("==> Loading datasets...")
        
        image_size = (256, 256)
        mean_140k = [0.5207, 0.4258, 0.3806]
        std_140k = [0.2490, 0.2239, 0.2212]

        transform_train_140k = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_140k, std=std_140k),
        ])

        transform_val_test_140k = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_140k, std=std_140k),
        ])

        params = {
            'dataset_mode': self.dataset_mode,
            'train_batch_size': self.train_batch_size,
            'eval_batch_size': self.test_batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'ddp': False # این پارامتر را در Dataset_selector خود مدیریت کنید
        }
        
        # ... (بخش‌های دیگر params مثل قبل هستند)
        if self.dataset_mode == 'hardfake':
            params['hardfake_csv_file'] = os.path.join(self.dataset_dir, 'data.csv')
            params['hardfake_root_dir'] = self.dataset_dir
        elif self.dataset_mode == 'rvf10k':
            params['rvf10k_train_csv'] = os.path.join(self.dataset_dir, 'train.csv')
            params['rvf10k_valid_csv'] = os.path.join(self.dataset_dir, 'valid.csv')
            params['rvf10k_root_dir'] = self.dataset_dir
        elif self.dataset_mode == '140k':
            params['realfake140k_train_csv'] = os.path.join(self.dataset_dir, 'train.csv')
            params['realfake140k_valid_csv'] = os.path.join(self.dataset_dir, 'valid.csv')
            params['realfake140k_test_csv'] = os.path.join(self.dataset_dir, 'test.csv')
            params['realfake140k_root_dir'] = self.dataset_dir
        elif self.dataset_mode == '200k':
            image_root_dir = os.path.join(self.dataset_dir, 'my_real_vs_ai_dataset', 'my_real_vs_ai_dataset')
            params['realfake200k_root_dir'] = image_root_dir
            params['realfake200k_train_csv'] = os.path.join(self.dataset_dir, 'train_labels.csv')
            params['realfake200k_val_csv'] = os.path.join(self.dataset_dir, 'val_labels.csv')
            params['realfake200k_test_csv'] = os.path.join(self.dataset_dir, 'test_labels.csv')
        elif self.dataset_mode == '190k':
            params['realfake190k_root_dir'] = self.dataset_dir
        elif self.dataset_mode == '330k':
            params['realfake330k_root_dir'] = self.dataset_dir

        dataset_manager = Dataset_selector(**params)

        if self.rank == 0:
            print("Overriding transforms to use consistent 140k normalization stats for all datasets.")
        
        train_dataset = dataset_manager.loader_train.dataset
        val_dataset = dataset_manager.loader_val.dataset
        test_dataset = dataset_manager.loader_test.dataset
        
        train_dataset.transform = transform_train_140k
        val_dataset.transform = transform_val_test_140k
        test_dataset.transform = transform_val_test_140k

        # DDP: ایجاد DistributedSampler برای هر دیتاست
        train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
        
        # DDP: ایجاد DataLoader جدید با sampler
        # نکته: batch_size در اینجا برابر با batch_size در هر GPU است
        self.train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, sampler=train_sampler, num_workers=self.num_workers, pin_memory=self.pin_memory)
        self.val_loader = DataLoader(val_dataset, batch_size=self.test_batch_size, sampler=val_sampler, num_workers=self.num_workers, pin_memory=self.pin_memory)
        self.test_loader = DataLoader(test_dataset, batch_size=self.test_batch_size, sampler=test_sampler, num_workers=self.num_workers, pin_memory=self.pin_memory)

        if self.rank == 0:
            print(f"All loaders for '{self.dataset_mode}' are now configured for DDP.")

        # Load new test dataset if provided
        if self.new_dataset_dir:
            if self.rank == 0:
                print("==> Loading new test dataset...")
            new_params = {
                'dataset_mode': 'new_test',
                'eval_batch_size': self.test_batch_size,
                'num_workers': self.num_workers,
                'pin_memory': self.pin_memory,
                'new_test_csv': os.path.join(self.new_dataset_dir, 'test.csv'),
                'new_test_root_dir': self.new_dataset_dir
            }
            new_dataset_manager = Dataset_selector(**new_params)
            new_test_dataset = new_dataset_manager.loader_test.dataset
            new_test_dataset.transform = transform_val_test_140k
            new_test_sampler = DistributedSampler(new_test_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
            self.new_test_loader = DataLoader(new_test_dataset, batch_size=self.test_batch_size, sampler=new_test_sampler, num_workers=self.num_workers, pin_memory=self.pin_memory)
            if self.rank == 0:
                print(f"New test dataset loader configured for DDP.")

    def build_model(self):
        if self.rank == 0:
            print("==> Building student model...")
        
        self.student = ResNet_50_sparse_hardfakevsreal()
        
        if not os.path.exists(self.sparsed_student_ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.sparsed_student_ckpt_path}")
            
        if self.rank == 0:
            print(f"Loading pre-trained weights from: {self.sparsed_student_ckpt_path}")
        ckpt_student = torch.load(self.sparsed_student_ckpt_path, map_location="cpu")
        state_dict = ckpt_student.get("student", ckpt_student)
        
        self.student.load_state_dict(state_dict, strict=False)
        self.student.to(self.device)
        
        # DDP: پوشاندن مدل با DDP
        self.student = DDP(self.student, device_ids=[self.rank])
        
        if self.rank == 0:
            print(f"Model wrapped with DDP and loaded on GPU {self.device}")

    def compute_metrics(self, loader, description="Test", print_metrics=True, save_confusion_matrix=True):
        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")
        all_preds = []
        all_targets = []
        sample_info = []
        
        self.student.eval()
        # DDP: دسترسی به مدل اصلی از طریق .module
        self.student.module.ticket = True
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(loader, desc=description, ncols=100, disable=(self.rank != 0))):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True).float()
                
                logits, _ = self.student(images)
                logits = logits.squeeze()
                preds = (torch.sigmoid(logits) > 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # ... (بخش sample_info مثل قبل باقی می‌ماند)
                batch_size = images.size(0)
                for i in range(batch_size):
                    try:
                        # DDP: برای محاسبه صحیح اندیس در دیتاست توزیع‌شده
                        global_idx = batch_idx * loader.batch_size + i
                        img_path = loader.dataset.samples[global_idx][0]
                    except (AttributeError, IndexError):
                        img_path = f"Sample_{global_idx}"
                    sample_info.append({
                        'id': img_path,
                        'true_label': targets[i].item(),
                        'pred_label': preds[i].item()
                    })
                
                correct = (preds == targets).sum().item()
                prec1 = 100.0 * correct / images.size(0)
                meter_top1.update(prec1, images.size(0))
        
        # DDP: جمع‌آوری نتایج از تمام GPUها
        gathered_preds = [None] * self.world_size
        gathered_targets = [None] * self.world_size
        gathered_sample_info = [None] * self.world_size
        
        # استفاده از all_gather_object برای اشیاء پیچیده‌تر
        dist.all_gather_object(gathered_preds, all_preds)
        dist.all_gather_object(gathered_targets, all_targets)
        dist.all_gather_object(gathered_sample_info, sample_info)

        # DDP: فقط در رنک 0 محاسبات نهایی و چاپ را انجام بده
        if self.rank == 0:
            all_preds = [item for sublist in gathered_preds for item in sublist]
            all_targets = [item for sublist in gathered_targets for item in sublist]
            all_sample_info = [item for sublist in gathered_sample_info for item in sublist]
            
            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)
            
            accuracy = meter_top1.avg # این مقدار میانگین محلی است، برای دقت کل باید دوباره محاسبه شود
            # محاسبه دقت کلی
            accuracy = 100.0 * np.sum(all_preds == all_targets) / len(all_preds)

            precision = precision_score(all_targets, all_preds, average='binary')
            recall = recall_score(all_targets, all_preds, average='binary')
            
            precision_per_class = precision_score(all_targets, all_preds, average=None, labels=[0, 1])
            recall_per_class = recall_score(all_targets, all_preds, average=None, labels=[0, 1])
            
            tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
            specificity_real = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificity_fake = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            if print_metrics:
                print(f"\n[{description}] Overall Metrics:")
                print(f"Accuracy: {accuracy:.2f}%")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"Specificity: {specificity_real:.4f}")
                
                print(f"\n[{description}] Per-Class Metrics:")
                print(f"Class Real (0):")
                print(f"  Precision: {precision_per_class[0]:.4f}")
                print(f"  Recall: {recall_per_class[0]:.4f}")
                print(f"  Specificity: {specificity_real:.4f}")
                print(f"Class Fake (1):")
                print(f"  Precision: {precision_per_class[1]:.4f}")
                print(f"  Recall: {recall_per_class[1]:.4f}")
                print(f"  Specificity: {specificity_fake:.4f}")
            
            cm = confusion_matrix(all_targets, all_preds)
            classes = ['Real', 'Fake']
            
            if save_confusion_matrix:
                print(f"\n[{description}] Confusion Matrix:")
                print(f"{'':>10} {'Predicted Real':>15} {'Predicted Fake':>15}")
                print(f"{'Actual Real':>10} {cm[0,0]:>15} {cm[0,1]:>15}")
                print(f"{'Actual Fake':>10} {cm[1,0]:>15} {cm[1,1]:>15}")
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
                plt.title(f'Confusion Matrix - {description}')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                
                sanitized_description = description.lower().replace(" ", "_").replace("/", "_")
                plot_path = os.path.join(self.result_dir, f'confusion_matrix_{sanitized_description}.png')
                os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                plt.savefig(plot_path)
                plt.close()
                print(f"Confusion matrix saved to: {plot_path}")
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'specificity': specificity_real,
                'precision_per_class': precision_per_class,
                'recall_per_class': recall_per_class,
                'specificity_per_class': [specificity_real, specificity_fake],
                'confusion_matrix': cm,
                'sample_info': all_sample_info
            }
        return None # برای رنک‌های دیگر None برگردان

    def display_samples(self, sample_info, description="Test", num_samples=30):
        # DDP: فقط در رنک 0 اجرا شود
        if self.rank == 0 and sample_info:
            print(f"\n[{description}] Displaying first {num_samples} test samples:")
            print(f"{'Sample ID':<50} {'True Label':<12} {'Predicted Label':<12}")
            print("-" * 80)
            for i, sample in enumerate(sample_info[:num_samples]):
                true_label = 'Real' if sample['true_label'] == 0 else 'Fake'
                pred_label = 'Real' if sample['pred_label'] == 0 else 'Fake'
                print(f"{sample['id']:<50} {true_label:<12} {pred_label:<12}")

    def finetune(self):
        if self.rank == 0:
            print("==> Fine-tuning using FEATURE EXTRACTOR strategy on 'fc' and 'layer4'...")
            if not os.path.exists(self.result_dir):
                os.makedirs(self.result_dir)
        
        # DDP: دسترسی به پارامترهای مدل اصلی از طریق .module
        for name, param in self.student.module.named_parameters():
            if 'fc' in name or 'layer4' in name:
                param.requires_grad = True
                if self.rank == 0:
                    print(f"Unfreezing for training: {name}")
            else:
                param.requires_grad = False

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.student.module.parameters()),
            lr=self.args.f_lr,
            weight_decay=self.args.f_weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        self.student.module.ticket = False
        
        best_val_acc = 0.0
        best_model_path = os.path.join(self.result_dir, f'finetuned_model_best_{self.dataset_mode}.pth')

        for epoch in range(self.args.f_epochs):
            self.student.train()
            # DDP: تنظیم epoch برای sampler تا شافل درستی در هر epoch داشته باشیم
            self.train_loader.sampler.set_epoch(epoch)
            
            meter_loss = meter.AverageMeter("Loss", ":6.4f")
            meter_top1_train = meter.AverageMeter("Train Acc@1", ":6.2f")
            
            # DDP: disable tqdm برای رنک‌های دیگر برای جلوگیری از شلوغی خروجی
            for images, targets in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.f_epochs} [Train]", ncols=100, disable=(self.rank != 0)):
                images, targets = images.to(self.device), targets.to(self.device).float()
                optimizer.zero_grad()
                logits, _ = self.student(images)
                logits = logits.squeeze()
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct = (preds == targets).sum().item()
                prec1 = 100.0 * correct / images.size(0)
                meter_loss.update(loss.item(), images.size(0))
                meter_top1_train.update(prec1, images.size(0))

            # Compute validation metrics
            val_metrics = self.compute_metrics(self.val_loader, description=f"Epoch_{epoch+1}_{self.args.f_epochs}_Val", print_metrics=False, save_confusion_matrix=False)
            
            # DDP: فقط رنک 0 نتایج را چاپ و مدل را ذخیره می‌کند
            if self.rank == 0:
                val_acc = val_metrics['accuracy']
                print(f"Epoch {epoch+1}: Train Loss: {meter_loss.avg:.4f}, Train Acc: {meter_top1_train.avg:.2f}%, Val Acc: {val_acc:.2f}%")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    print(f"New best model found with Val Acc: {best_val_acc:.2f}%. Saving to {best_model_path}")
                    # DDP: ذخیره state_dict مدل اصلی از طریق .module
                    torch.save(self.student.module.state_dict(), best_model_path)
            
            scheduler.step()
        
        if self.rank == 0:
            print(f"\nFine-tuning finished. Loading best model with Val Acc: {best_val_acc:.2f}%")
            if os.path.exists(best_model_path):
                self.student.module.load_state_dict(torch.load(best_model_path))
            else:
                print("Warning: No best model was saved. The model from the last epoch will be used for testing.")
        
        # منتظر بمان تا تمام فرآیندها به اینجا برسند
        dist.barrier()
        
        # Compute and print final test metrics after fine-tuning
        final_test_metrics = self.compute_metrics(self.test_loader, description="Final_Test", print_metrics=True, save_confusion_matrix=True)
        self.display_samples(final_test_metrics['sample_info'], "Final Test", num_samples=30)

    def main(self):
        if self.rank == 0:
            print(f"Starting pipeline with dataset mode: {self.dataset_mode} on {self.world_size} GPUs.")
        self.dataload()
        self.build_model()
        
        if self.rank == 0:
            print("\n--- Testing BEFORE fine-tuning ---")
        initial_metrics = self.compute_metrics(self.test_loader, "Initial_Test")
        self.display_samples(initial_metrics['sample_info'], "Initial Test", num_samples=30)
        
        if self.rank == 0:
            print("\n--- Starting fine-tuning ---")
        self.finetune()
        
        if self.rank == 0:
            print("\n--- Testing AFTER fine-tuning with best model ---")
        # مدل بهترین حالت در finetune بارگذاری شده است
        final_metrics = self.compute_metrics(self.test_loader, "Final_Test_Again", print_metrics=True, save_confusion_matrix=True)
        self.display_samples(final_metrics['sample_info'], "Final Test Again", num_samples=30)
        
        if self.new_test_loader:
            if self.rank == 0:
                print("\n--- Testing on NEW dataset ---")
            new_metrics = self.compute_metrics(self.new_test_loader, "New_Dataset_Test")
            self.display_samples(new_metrics['sample_info'], "New Dataset Test", num_samples=30)


# DDP: تابع اصلی که برای هر فرآیند (GPU) فراخوانی می‌شود
def main_worker(rank, world_size, args):
    setup_ddp(rank, world_size)
    args.rank = rank
    args.world_size = world_size
    
    tester = Test(args)
    tester.main()
    
    cleanup_ddp()


if __name__ == '__main__':
    # فرض می‌کنیم که شما از argparse برای پارامترها استفاده می‌کنید
    import argparse
    parser = argparse.ArgumentParser(description='DDP Test')
    # ... تمام آرگومان‌های قبلی خود را اینجا اضافه کنید
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--sparsed_student_ckpt_path', type=str, required=True)
    parser.add_argument('--result_dir', type=str, default='./results')
    parser.add_argument('--dataset_mode', type=str, default='140k')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--arch', type=str, default='resnet50')
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--f_lr', type=float, default=1e-4)
    parser.add_argument('--f_weight_decay', type=float, default=1e-5)
    parser.add_argument('--f_epochs', type=int, default=10)
    # ... و غیره

    args = parser.parse_args()
    
    # DDP: تعیین تعداد GPUها
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Warning: This script is designed for at least 2 GPUs. Running on a single GPU.")
        # می‌توانید منطقی برای اجرای تک GPU اینجا اضافه کنید
        # اما برای سادگی، اجرا متوقف می‌شود
        exit()

    # DDP: اجرای فرآیندها با mp.spawn
    mp.spawn(main_worker,
             args=(world_size, args),
             nprocs=world_size,
             join=True)
