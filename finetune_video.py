import os
import random
import utils  
import time
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import cv2
from pathlib import Path
from data.video_data import create_uadfv_dataloaders
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from model.student.ResNet_sparse_video import ResNet_50_sparse_uadfv
class FinetuneDDP:
    def __init__(self, args):
        """Initialize FinetuneDDP with provided arguments."""
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.dataset_mode = args.dataset_mode

        self.num_frames = args.num_frames
        self.image_size = args.image_size
        self.sampling_strategy = args.sampling_strategy
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.arch = args.arch
        self.seed = args.seed
        self.result_dir = args.result_dir
        self.finetune_train_batch_size = args.finetune_train_batch_size
        self.finetune_eval_batch_size = args.finetune_eval_batch_size
        self.finetune_student_ckpt_path = args.finetune_student_ckpt_path
        self.finetune_num_epochs = args.finetune_num_epochs
        self.finetune_lr = args.finetune_lr
        self.finetune_warmup_steps = args.finetune_warmup_steps
        self.finetune_warmup_start_lr = args.finetune_warmup_start_lr
        self.finetune_lr_decay_T_max = args.finetune_lr_decay_T_max
        self.finetune_lr_decay_eta_min = args.finetune_lr_decay_eta_min
        self.finetune_weight_decay = args.finetune_weight_decay
        self.finetune_resume = args.finetune_resume
        self.sparsed_student_ckpt_path = args.sparsed_student_ckpt_path
        self.start_epoch = 0
        self.best_prec1_after_finetune = 0
        self.world_size = 0
        self.local_rank = -1
        self.rank = -1

    def dist_init(self):
        """Initialize distributed training with NCCL backend."""
        # برای اجرای محلی، DDP رو غیرفعال می‌کنیم تا نیاز به تنظیمات محیطی نداشته باشه
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            dist.init_process_group("nccl")
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(self.local_rank)
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
        print(f"Running on rank {self.rank}/{self.world_size}")


    def result_init(self):
        """Initialize logging and TensorBoard writer for rank 0."""
        if self.rank == 0:
            os.makedirs(self.result_dir, exist_ok=True)
            self.writer = SummaryWriter(self.result_dir)
            self.logger = utils.get_logger(os.path.join(self.result_dir, "finetune_logger.log"), "finetune_logger")
            self.logger.info("Finetune configuration:")
            self.logger.info(str(vars(self.args)))
            utils.record_config(self.args, os.path.join(self.result_dir, "finetune_config.txt"))
            self.logger.info("--------- Finetune -----------")

    def setup_seed(self):
        """Set random seeds for reproducibility."""
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        self.seed += self.rank
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def dataload(self):
        """Load dataset based on dataset_mode."""
        # --- تغییر جدید: اضافه شدن حالت 'uadfv' ---
        if self.dataset_mode == 'uadfv':
            self.train_loader, self.val_loader, self.test_loader = create_uadfv_dataloaders(
                root_dir=self.dataset_dir,
                num_frames=self.num_frames,
                image_size=self.image_size,
                train_batch_size=self.finetune_train_batch_size,
                eval_batch_size=self.finetune_eval_batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                ddp=(self.world_size > 1), # اگر DDP فعال بود
                sampling_strategy=self.sampling_strategy,
                seed=self.seed
            )
        else:
            raise ValueError(f"Unsupported dataset_mode: {self.dataset_mode}. Please use 'uadfv'.")

        if self.rank == 0:
            self.logger.info("Dataset loaded successfully!")

    def build_model(self):
        """Build and load the student model."""
        if self.rank == 0:
            self.logger.info("==> Building model...")
            self.logger.info("Loading student model")
        
        # اگر چک‌پوینت وجود داشت، لود می‌کنیم، در غیر این صورت مدل رو از نو می‌سازیم
        self.student = ResNet_50_sparse_uadfv()
        if os.path.exists(self.finetune_student_ckpt_path):
            ckpt_student = torch.load(self.finetune_student_ckpt_path, map_location="cpu")
            # فرض می‌کنیم چک‌پوینت state_dict مدل را تحت کلید 'student' دارد
            if 'student' in ckpt_student:
                self.student.load_state_dict(ckpt_student["student"])
                if self.rank == 0:
                    self.best_prec1_before_finetune = ckpt_student.get("best_prec1", 0.0)
            else:
                self.student.load_state_dict(ckpt_student)
        else:
            if self.rank == 0:
                self.logger.warning(f"Checkpoint not found at {self.finetune_student_ckpt_path}. Training from scratch.")
                self.best_prec1_before_finetune = 0.0

        if torch.cuda.is_available():
            self.student = self.student.cuda()
        
        if self.world_size > 1:
            self.student = DDP(self.student, device_ids=[self.local_rank], find_unused_parameters=True)

    def define_loss(self):
        """Define the loss function."""
        self.ori_loss = nn.BCEWithLogitsLoss()

    def define_optim(self):
        """Define optimizer and scheduler."""
        weight_params = [p for n, p in self.student.named_parameters() if p.requires_grad and "mask" not in n]
        self.finetune_optim_weight = torch.optim.Adamax(
            weight_params, lr=self.finetune_lr, weight_decay=self.finetune_weight_decay, eps=1e-7,
        )
        self.finetune_scheduler_student_weight = scheduler.CosineAnnealingLRWarmup(
            self.finetune_optim_weight, T_max=self.finetune_lr_decay_T_max,
            eta_min=self.finetune_lr_decay_eta_min, last_epoch=-1,
            warmup_steps=self.finetune_warmup_steps, warmup_start_lr=self.finetune_warmup_start_lr,
        )

    def resume_student_ckpt(self):
        """Resume training from a checkpoint."""
        if os.path.exists(self.finetune_resume):
            ckpt_student = torch.load(self.finetune_resume, map_location="cpu")
            self.best_prec1_after_finetune = ckpt_student["best_prec1_after_finetune"]
            self.start_epoch = ckpt_student["start_epoch"]
            self.student.module.load_state_dict(ckpt_student["student"])
            self.finetune_optim_weight.load_state_dict(ckpt_student["finetune_optim_weight"])
            self.finetune_scheduler_student_weight.load_state_dict(ckpt_student["finetune_scheduler_student_weight"])
            if self.rank == 0:
                self.logger.info(f"=> Resuming from epoch {self.start_epoch}...")

    def save_student_ckpt(self, is_best):
        """Save model checkpoint."""
        if self.rank == 0:
            folder = os.path.join(self.result_dir, "student_model")
            os.makedirs(folder, exist_ok=True)
            model_to_save = self.student.module if self.world_size > 1 else self.student
            ckpt_student = {
                "best_prec1_after_finetune": self.best_prec1_after_finetune,
                "start_epoch": self.start_epoch,
                "student": model_to_save.state_dict(),
                "finetune_optim_weight": self.finetune_optim_weight.state_dict(),
                "finetune_scheduler_student_weight": self.finetune_scheduler_student_weight.state_dict(),
            }
            if is_best:
                torch.save(ckpt_student, os.path.join(folder, f"finetune_{self.arch}_sparse_best.pt"))
            torch.save(ckpt_student, os.path.join(folder, f"finetune_{self.arch}_sparse_last.pt"))

    def reduce_tensor(self, tensor):
        """Reduce tensor across all processes in DDP."""
        if self.world_size > 1:
            rt = tensor.clone()
            dist.all_reduce(rt, op=dist.ReduceOp.SUM)
            rt /= self.world_size
            return rt
        return tensor

    def finetune(self):
        """Perform finetuning of the student model."""
        if torch.cuda.is_available():
            self.ori_loss = self.ori_loss.cuda()
        
        if self.finetune_resume:
            self.resume_student_ckpt()

        if self.rank == 0:
            meter_oriloss = meter.AverageMeter("OriLoss", ":.4e")
            meter_loss = meter.AverageMeter("Loss", ":.4e")
            meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")

        for epoch in range(self.start_epoch + 1, self.finetune_num_epochs + 1):
            if self.world_size > 1:
                self.train_loader.sampler.set_epoch(epoch)
            
            model_to_train = self.student.module if self.world_size > 1 else self.student
            model_to_train.train()

            if self.rank == 0:
                meter_oriloss.reset()
                meter_loss.reset()
                meter_top1.reset()
                finetune_lr = self.finetune_optim_weight.param_groups[0]['lr']

            with tqdm(total=len(self.train_loader), ncols=100, disable=(self.rank != 0)) as _tqdm:
                if self.rank == 0:
                    _tqdm.set_description(f"Epoch: {epoch}/{self.finetune_num_epochs}")
                
                for images, targets in self.train_loader:
                    self.finetune_optim_weight.zero_grad()
                    
                    if torch.cuda.is_available():
                        images = images.cuda(non_blocking=True)
                        targets = targets.cuda(non_blocking=True).float()
                    
                    # --- تغییر اصلی: پردازش ویدیو ---
                    # images شکل [B, T, C, H, W] دارد
                    B, T, C, H, W = images.shape
                    # ابعاد را ادغام می‌کنیم تا همه فریم‌ها یکجا به مدل داده شوند
                    images_flat = images.view(B * T, C, H, W)
                    
                    # مدل را روی تمام فریم‌ها اجرا می‌کنیم
                    logits_student_raw, _ = self.student(images_flat)
                    
                    # خروجی را به شکل [B, T] برمی‌گردانیم
                    logits_student_raw = logits_student_raw.view(B, T)
                    
                    # میانگین‌گیری در طول زمان برای رسیدن به یک پیش‌بینی برای هر ویدیو
                    logits_student = logits_student_raw.mean(dim=1)
                    
                    # محاسبه loss با لیبل ویدیو
                    ori_loss = self.ori_loss(logits_student.squeeze(1), targets)
                    total_loss = ori_loss
                    total_loss.backward()
                    self.finetune_optim_weight.step()

                    preds = (torch.sigmoid(logits_student) > 0.5).float()
                    correct = (preds.squeeze(1) == targets).sum().item()
                    prec1 = torch.tensor(100. * correct / B, device=images.device)

                    n = B
                    reduced_ori_loss = self.reduce_tensor(ori_loss)
                    reduced_total_loss = self.reduce_tensor(total_loss)
                    reduced_prec1 = self.reduce_tensor(prec1)

                    if self.rank == 0:
                        meter_oriloss.update(reduced_ori_loss.item(), n)
                        meter_loss.update(reduced_total_loss.item(), n)
                        meter_top1.update(reduced_prec1.item(), n)
                        _tqdm.set_postfix(loss=f"{meter_loss.avg:.4f}", top1=f"{meter_top1.avg:.2f}")
                        _tqdm.update(1)

            self.finetune_scheduler_student_weight.step()

            # --- بخش اعتبارسنجی (Validation) ---
            if self.rank == 0:
                model_to_train.eval()
                meter_top1.reset()
                with torch.no_grad():
                    with tqdm(total=len(self.val_loader), ncols=100) as _tqdm:
                        _tqdm.set_description(f"Val Epoch: {epoch}/{self.finetune_num_epochs}")
                        for images, targets in self.val_loader:
                            if torch.cuda.is_available():
                                images = images.cuda(non_blocking=True)
                                targets = targets.cuda(non_blocking=True).float()
                            
                            # --- تغییر اصلی: پردازش ویدیو در Validation ---
                            B, T, C, H, W = images.shape
                            images_flat = images.view(B * T, C, H, W)
                            logits_student_raw, _ = self.student(images_flat)
                            logits_student_raw = logits_student_raw.view(B, T)
                            logits_student = logits_student_raw.mean(dim=1)
                            
                            preds = (torch.sigmoid(logits_student) > 0.5).float()
                            correct = (preds.squeeze(1) == targets).sum().item()
                            prec1 = 100. * correct / B
                            meter_top1.update(prec1, B)
                            _tqdm.set_postfix(top1=f"{meter_top1.avg:.2f}")
                            _tqdm.update(1)
                
                self.writer.add_scalar("finetune_val/acc/top1", meter_top1.avg, epoch)
                self.logger.info(f"[Finetune_val] Epoch {epoch}: Accuracy@1 {meter_top1.avg:.2f}")
                
                self.start_epoch += 1
                if self.best_prec1_after_finetune < meter_top1.avg:
                    self.best_prec1_after_finetune = meter_top1.avg
                    self.save_student_ckpt(True)
                else:
                    self.save_student_ckpt(False)
                
                self.logger.info(f" => Best Accuracy@1 before finetune: {self.best_prec1_before_finetune:.2f}")
                self.logger.info(f" => Best Accuracy@1 after finetune: {self.best_prec1_after_finetune:.2f}")

        if self.rank == 0:
            self.logger.info("Finetuning completed!")
            self.logger.info(f"Best Accuracy@1: {self.best_prec1_after_finetune:.2f}")

    def main(self):
        """Main function to orchestrate finetuning process."""
        self.dist_init()
        self.result_init()
        self.setup_seed()
        self.dataload()
        self.build_model()
        self.define_loss()
        self.define_optim()
        self.finetune()
        if self.world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    # یک کلاس ساده برای شبیه‌سازی argparse
    class Args:
        def __init__(self):
            # --- پارامترهای جدید برای ویدیو ---
            self.dataset_mode = 'uadfv'  # حالت جدید برای دیتاست ویدیویی
            self.num_frames = 16
            self.image_size = 256
            self.sampling_strategy = 'uniform'
            
            # --- پارامترهای اصلی ---
            self.dataset_dir = "/kaggle/input/uadfv-dataset/UADFV"  # <-- مسیر دیتاست خود را اینجا قرار دهید
            self.result_dir = "/kaggle/working/results/uadfv_finetune"
            self.arch = "ResNet_50"
            self.seed = 42
            
            self.num_workers = 2
            self.pin_memory = True
            
            self.finetune_train_batch_size = 4  # با توجه به حافظه GPU تغییر دهید
            self.finetune_eval_batch_size = 8   # با توجه به حافظه GPU تغییر دهید
            
            self.finetune_student_ckpt_path = "" # اگر چک‌پوینت اولیه دارید، مسیر آن را وارد کنید
            self.finetune_num_epochs = 5
            self.finetune_lr = 1e-4
            self.finetune_warmup_steps = 100
            self.finetune_warmup_start_lr = 1e-6
            self.finetune_lr_decay_T_max = 1000
            self.finetune_lr_decay_eta_min = 1e-6
            self.finetune_weight_decay = 0.0
            self.finetune_resume = "" # اگر از ادامه آموزش می‌خواهید استفاده کنید
            self.sparsed_student_ckpt_path = "" # این در کد اصلی شما استفاده شده ولی در اینجا لازم نیست

    args = Args()

    
    print("Starting fine-tuning process...")
    # ایجاد یک دیتاست ساختگی برای تست
    fake_dataset_path = args.dataset_dir
    if not os.path.exists(fake_dataset_path):
        print(f"Creating a dummy dataset at {fake_dataset_path} for demonstration.")
        os.makedirs(os.path.join(fake_dataset_path, 'fake'), exist_ok=True)
        os.makedirs(os.path.join(fake_dataset_path, 'real'), exist_ok=True)
        # ایجاد چند فایل خالی .mp4 برای جلوگیری از خطا
        for i in range(10):
            open(os.path.join(fake_dataset_path, 'fake', f'fake_{i}.mp4'), 'a').close()
            open(os.path.join(fake_dataset_path, 'real', f'real_{i}.mp4'), 'a').close()
        print("Dummy dataset created. Please replace it with the real UADFV dataset.")
    
    finetuner = FinetuneDDP(args)
    finetuner.main()
