import json
import os
import random
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from data.video_data import create_uadfv_dataloaders
import cv2
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from model.student.ResNet_sparse_video import (ResNet_50_sparse_uadfv, SoftMaskedConv2d)
from model.student.MobileNetV2_sparse import MobileNetV2_sparse_deepfake
from model.student.GoogleNet_sparse import GoogLeNet_sparse_deepfake
from utils import utils, loss, meter, scheduler
from thop import profile
from model.teacher.ResNet import ResNet_50_hardfakevsreal
from model.teacher.Mobilenetv2 import MobileNetV2_deepfake
from model.teacher.GoogleNet import GoogLeNet_deepfake
from utils.loss import compute_filter_correlation
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
class TrainDDP:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.dataset_mode = args.dataset_mode
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.seed = args.seed
        self.result_dir = args.result_dir
        self.teacher_ckpt_path = args.teacher_ckpt_path
        self.num_epochs = args.num_epochs
        
        # Video-specific parameters
        self.num_frames = getattr(args, 'num_frames', 16)
        self.frame_sampling = getattr(args, 'frame_sampling', 'uniform')
        self.split_ratio = getattr(args, 'split_ratio', (0.7, 0.15, 0.15))
        
        self.lr = args.lr
        self.warmup_steps = args.warmup_steps
        self.warmup_start_lr = args.warmup_start_lr
        self.lr_decay_T_max = args.lr_decay_T_max
        self.lr_decay_eta_min = args.lr_decay_eta_min
        self.weight_decay = args.weight_decay
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        
        self.target_temperature = args.target_temperature
        self.gumbel_start_temperature = args.gumbel_start_temperature
        self.gumbel_end_temperature = args.gumbel_end_temperature
        
        self.coef_kdloss = args.coef_kdloss
        self.coef_rcloss = args.coef_rcloss
        self.coef_maskloss = args.coef_maskloss
        
        self.resume = args.resume
        self.start_epoch = 0
        self.best_prec1 = 0
        
        self.world_size = 0
        self.local_rank = -1
        self.rank = -1
        
        self.device = None  # Initialize as None, set later
        
        if self.dataset_mode == "uadfv":
            self.args.dataset_type = "uadfv"
            self.num_classes = 1
            self.image_size = 256
        else:
            raise ValueError("dataset_mode must be 'uadfv' for this script")
        
        self.arch = args.arch.lower().replace('_', '')
        if self.arch not in ['resnet50', 'mobilenetv2', 'googlenet']:
            raise ValueError(f"Unsupported architecture: '{args.arch}'. "
                           "It must be 'resnet50', 'mobilenetv2', or 'googlenet'.")
    def dist_init(self):
        dist.init_process_group("nccl")
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)
    def result_init(self):
        if self.rank == 0:
            if not os.path.exists(self.result_dir):
                os.makedirs(self.result_dir)
            
            self.writer = SummaryWriter(self.result_dir)
            self.logger = utils.get_logger(
                os.path.join(self.result_dir, "train_logger.log"), "train_logger"
            )
            self.logger.info("train config:")
            self.logger.info(str(json.dumps(vars(self.args), indent=4)))
            utils.record_config(
                self.args, os.path.join(self.result_dir, "train_config.txt")
            )
            self.logger.info("--------- Train -----------")
    def setup_seed(self):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        
        self.seed = self.seed + self.rank
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = True
    def dataload(self):
        if self.dataset_mode == "uadfv":
            # Dataset loader is now defined in this file
            if self.rank == 0:
                self.logger.info(f"Loading UADFV dataset from: {self.dataset_dir}")
                self.logger.info(f"Number of frames per video: {self.num_frames}")
                self.logger.info(f"Frame sampling strategy: {self.frame_sampling}")
                self.logger.info(f"Split ratio (train/val/test): {self.split_ratio}")
            
            self.train_loader, self.val_loader, self.test_loader = create_uadfv_dataloaders(
                root_dir=self.dataset_dir,
                num_frames=self.num_frames,
                image_size=self.image_size,
                train_batch_size=self.train_batch_size,
                eval_batch_size=self.eval_batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                ddp=True,
                seed=self.seed,
                sampling_strategy=self.frame_sampling
            )
            
            if self.rank == 0:
                self.logger.info("UADFV Dataset has been loaded!")
    def build_model(self):
        if self.rank == 0:
            self.logger.info("==> Building model..")
            self.logger.info("Loading teacher model")
        
        # Teacher model
        if self.arch == 'resnet50':
            teacher_model = ResNet_50_hardfakevsreal()
        elif self.arch == 'mobilenetv2':
            teacher_model = MobileNetV2_deepfake()
        elif self.arch == 'googlenet':
            teacher_model = GoogLeNet_deepfake()
        else:
            raise ValueError(f"Unsupported architecture: {self.arch}")
        
        ckpt_teacher = torch.load(self.teacher_ckpt_path, map_location="cpu")
        state_dict = ckpt_teacher.get('config_state_dict',
                                     ckpt_teacher.get('student', ckpt_teacher))
        
        if list(state_dict.keys())[0].startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '', 1)
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        teacher_model.load_state_dict(state_dict, strict=True)
        self.teacher = teacher_model.to(self.device)
        
        # Student model
        if self.rank == 0:
            self.logger.info("Building student model")
        
        if self.arch == 'resnet50':
            StudentModelClass = ResNet_50_sparse_uadfv
        elif self.arch == 'mobilenetv2':
            StudentModelClass = MobileNetV2_sparse_deepfake
        elif self.arch == 'googlenet':
            StudentModelClass = GoogLeNet_sparse_deepfake
        else:
            raise ValueError(f"Unsupported architecture for student: {self.arch}")
        
        self.student = StudentModelClass(
            gumbel_start_temperature=self.gumbel_start_temperature,
            gumbel_end_temperature=self.gumbel_end_temperature,
            num_epochs=self.num_epochs,
        ).to(self.device)
        self.student.dataset_type = self.args.dataset_type
        
        # Modify final layer for binary classification
        if self.arch == 'mobilenetv2':
            num_ftrs = self.student.classifier.in_features
            self.student.classifier = nn.Linear(num_ftrs, 1).to(self.device)
        elif self.arch == 'googlenet':
            num_ftrs = self.student.fc.in_features
            self.student.fc = nn.Linear(num_ftrs, 1).to(self.device)
        else: # resnet50
            num_ftrs = self.student.fc.in_features
            self.student.fc = nn.Linear(num_ftrs, 1).to(self.device)
        
        self.student = DDP(self.student, device_ids=[self.local_rank], find_unused_parameters=True)
    def define_loss(self):
        self.ori_loss = nn.BCEWithLogitsLoss().to(self.device)
        self.kd_loss = loss.KDLoss().to(self.device)
        self.rc_loss = loss.RCLoss().to(self.device)
        self.mask_loss = loss.MaskLoss().to(self.device)
    def define_optim(self):
        weight_params = [p[1] for p in self.student.named_parameters() if p[1].requires_grad and "mask" not in p[0]]
        mask_params = [p[1] for p in self.student.named_parameters() if p[1].requires_grad and "mask" in p[0]]
        
        self.optim_weight = torch.optim.Adamax(weight_params,
                                             lr=self.lr,
                                             weight_decay=self.weight_decay,
                                             eps=1e-7)
        self.optim_mask = torch.optim.Adamax(mask_params, lr=self.lr, eps=1e-7)
        
        self.scheduler_student_weight = scheduler.CosineAnnealingLRWarmup(
            self.optim_weight, T_max=self.lr_decay_T_max,
            eta_min=self.lr_decay_eta_min, last_epoch=-1,
            warmup_steps=self.warmup_steps,
            warmup_start_lr=self.warmup_start_lr)
        
        self.scheduler_student_mask = scheduler.CosineAnnealingLRWarmup(
            self.optim_mask, T_max=self.lr_decay_T_max,
            eta_min=self.lr_decay_eta_min, last_epoch=-1,
            warmup_steps=self.warmup_steps,
            warmup_start_lr=self.warmup_start_lr)
    def resume_student_ckpt(self):
        if not os.path.exists(self.resume):
            raise FileNotFoundError(f"Checkpoint file not found: {self.resume}")
        
        ckpt_student = torch.load(self.resume, map_location=self.device, weights_only=True)
        self.best_prec1 = ckpt_student["best_prec1"]
        self.start_epoch = ckpt_student["start_epoch"]
        self.student.load_state_dict(ckpt_student["student"])
        self.optim_weight.load_state_dict(ckpt_student["optim_weight"])
        self.optim_mask.load_state_dict(ckpt_student["optim_mask"])
        self.scheduler_student_weight.load_state_dict(ckpt_student["scheduler_student_weight"])
        self.scheduler_student_mask.load_state_dict(ckpt_student["scheduler_student_mask"])
        
        if self.rank == 0:
            self.logger.info(f"=> Continue from epoch {self.start_epoch + 1}...")
    def save_student_ckpt(self, is_best, epoch):
        if self.rank == 0:
            folder = os.path.join(self.result_dir, "student_model")
            if not os.path.exists(folder):
                os.makedirs(folder)
            
            ckpt_student = {
                "best_prec1": self.best_prec1,
                "start_epoch": epoch,
                "student": self.student.state_dict(),
                "optim_weight": self.optim_weight.state_dict(),
                "optim_mask": self.optim_mask.state_dict(),
                "scheduler_student_weight": self.scheduler_student_weight.state_dict(),
                "scheduler_student_mask": self.scheduler_student_mask.state_dict(),
            }
            
            if is_best:
                torch.save(ckpt_student,
                         os.path.join(folder, self.arch + "_sparse_best.pt"))
            
            torch.save(ckpt_student,
                     os.path.join(folder, self.arch + "_sparse_last.pt"))
    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.world_size
        return rt
    def get_mask_averages(self):
        """Get average mask values for each layer"""
        mask_avgs = []
        for m in self.student.mask_modules:
            if isinstance(m, SoftMaskedConv2d):
                with torch.no_grad():
                    mask = torch.sigmoid(m.mask_weight)
                    mask_avgs.append(round(mask.mean().item(), 2))
        return mask_avgs
    def validate(self, epoch):
        """Validation function that runs on all GPUs"""
        self.student.eval()
        self.student.ticket = True
        
        val_meter = meter.AverageMeter("Acc@1", ":6.2f")
        
        with torch.no_grad():
            for val_videos, val_targets in self.val_loader:
                val_videos = val_videos.to(self.device, non_blocking=True)
                val_targets = val_targets.to(self.device, non_blocking=True).float()
                
                val_batch_size, val_num_frames, C, H, W = val_videos.shape
                val_frames = val_videos.view(-1, C, H, W)
                
                val_logits, _ = self.student(val_frames)
                val_logits = val_logits.squeeze(1)
                val_logits = val_logits.view(val_batch_size, val_num_frames).mean(dim=1)
                
                val_preds = (torch.sigmoid(val_logits) > 0.5).float()
                correct = (val_preds == val_targets).sum().item()
                acc1 = 100.0 * correct / val_batch_size
                val_meter.update(acc1, val_batch_size)
        
        # Synchronize validation accuracy across all GPUs
        dist.barrier()
        val_acc_tensor = torch.tensor(val_meter.avg, device=self.device)
        reduced_val_acc = self.reduce_tensor(val_acc_tensor)
        
        # Get mask averages and FLOPs (only on rank 0)
        if self.rank == 0:
            mask_avgs = self.get_mask_averages()
            val_flops = self.student.get_flops()
            
            self.logger.info(f"[Val] Epoch {epoch} : Val_Acc {reduced_val_acc.item():.2f}")
            self.logger.info(f"[Val mask avg] Epoch {epoch} : {mask_avgs}")
            self.logger.info(f"[Val model Flops] Epoch {epoch} : {val_flops/1e6:.6f}M")
        else:
            mask_avgs = None
            val_flops = None
        
        return reduced_val_acc.item(), mask_avgs, val_flops
    def train(self):
        if self.rank == 0:
            self.logger.info(f"Starting training from epoch: {self.start_epoch + 1}")
        
        torch.cuda.empty_cache()
        self.teacher.eval()
        scaler = GradScaler('cuda')
        
        if self.resume:
            self.resume_student_ckpt()
        
        # Initialize meters (only on rank 0)
        if self.rank == 0:
            meter_oriloss = meter.AverageMeter("OriLoss", ":.4e")
            meter_kdloss = meter.AverageMeter("KDLoss", ":.4e")
            meter_rcloss = meter.AverageMeter("RCLoss", ":.4e")
            meter_maskloss = meter.AverageMeter("MaskLoss", ":.6e")
            meter_loss = meter.AverageMeter("Loss", ":.4e")
            meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")
        
        for epoch in range(self.start_epoch + 1, self.num_epochs + 1):
            self.train_loader.sampler.set_epoch(epoch)
            self.student.train()
            self.student.ticket = False
            
            if self.rank == 0:
                meter_oriloss.reset()
                meter_kdloss.reset()
                meter_rcloss.reset()
                meter_maskloss.reset()
                meter_loss.reset()
                meter_top1.reset()
                current_lr = self.optim_weight.param_groups[0]['lr']
            
            self.student.update_gumbel_temperature(epoch)
            current_gumbel_temp = self.student.gumbel_temperature
            
            with tqdm(total=len(self.train_loader), ncols=100, disable=self.rank != 0) as _tqdm:
                if self.rank == 0:
                    _tqdm.set_description(f"epoch: {epoch}/{self.num_epochs}")
                
                for batch_idx, batch_data in enumerate(self.train_loader):
                    self.optim_weight.zero_grad()
                    self.optim_mask.zero_grad()
                    
                    # Video dataset processing
                    videos, targets = batch_data
                    videos = videos.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True).float()
                    
                    batch_size, num_frames, C, H, W = videos.shape
                    images = videos.view(-1, C, H, W) # [B*T, C, H, W]
                    
                    # Check for invalid inputs
                    if torch.isnan(images).any() or torch.isinf(images).any() or \
                       torch.isnan(targets).any() or torch.isinf(targets).any():
                        if self.rank == 0:
                            self.logger.warning("Invalid input detected (NaN or Inf)")
                        continue
                    
                    with autocast(device_type='cuda'):
                        # ============ Student forward ============
                        logits_student, feature_list_student = self.student(images)
                        logits_student = logits_student.squeeze(1) # [B*T]
                        logits_student = logits_student.view(batch_size, num_frames).mean(dim=1) # [B]
                        
                        # ============ Teacher forward ============
                        with torch.no_grad():
                            logits_teacher, feature_list_teacher = self.teacher(images)
                            logits_teacher = logits_teacher.squeeze(1) # [B*T]
                            logits_teacher = logits_teacher.view(batch_size, num_frames).mean(dim=1) # [B]
                        
                        # ============ Original Loss ============
                        ori_loss = self.ori_loss(logits_student, targets)
                        
                        # ============ KD Loss ============
                        kd_loss = (self.target_temperature ** 2) * self.kd_loss(
                            logits_teacher, logits_student, self.target_temperature)
                        
                        # ============ RC Loss (FIXED) ============
                        rc_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                        
                        if len(feature_list_student) > 0:
                            num_layers = len(feature_list_student)
                            
                            for i in range(num_layers):
                                feat_s = feature_list_student[i] # [B*T, C, H, W]
                                feat_t = feature_list_teacher[i] # [B*T, C, H, W]
                                
                                # Reshape to [B, T, C, H, W] then average over time
                                _, c, h, w = feat_s.shape
                                feat_s_video = feat_s.view(batch_size, num_frames, c, h, w).mean(dim=1) # [B, C, H, W]
                                feat_t_video = feat_t.view(batch_size, num_frames, c, h, w).mean(dim=1) # [B, C, H, W]
                                
                                # Compute RC loss for this layer
                                layer_rc_loss = self.rc_loss(feat_s_video, feat_t_video)
                                rc_loss = rc_loss + layer_rc_loss
                            
                            # Average over all layers
                            rc_loss = rc_loss / num_layers
                        
                        # ============ Mask Loss ============
                        mask_loss = self.mask_loss(self.student)
                        
                        # ============ Total Loss ============
                        total_loss = (
                            ori_loss +
                            self.coef_kdloss * kd_loss +
                            self.coef_rcloss * rc_loss +
                            self.coef_maskloss * mask_loss
                        )
                    
                    # ============ Backward & Optimizer step ============
                    scaler.scale(total_loss).backward()
                    scaler.step(self.optim_weight)
                    scaler.step(self.optim_mask)
                    scaler.update()
                    
                    # ============ Compute accuracy ============
                    preds = (torch.sigmoid(logits_student) > 0.5).float()
                    correct = (preds == targets).sum().item()
                    prec1 = 100.0 * correct / batch_size
                    
                    # ============ Reduce metrics across GPUs ============
                    dist.barrier()
                    reduced_ori_loss = self.reduce_tensor(ori_loss.detach())
                    reduced_kd_loss = self.reduce_tensor(kd_loss.detach())
                    reduced_rc_loss = self.reduce_tensor(rc_loss.detach())
                    reduced_mask_loss = self.reduce_tensor(mask_loss.detach())
                    reduced_total_loss = self.reduce_tensor(total_loss.detach())
                    reduced_prec1 = self.reduce_tensor(torch.tensor(prec1, device=self.device))
                    
                    # ============ Update meters (rank 0 only) ============
                    if self.rank == 0:
                        meter_oriloss.update(reduced_ori_loss.item(), batch_size)
                        meter_kdloss.update(self.coef_kdloss * reduced_kd_loss.item(), batch_size)
                        meter_rcloss.update(self.coef_rcloss * reduced_rc_loss.item(), batch_size)
                        meter_maskloss.update(self.coef_maskloss * reduced_mask_loss.item(), batch_size)
                        meter_loss.update(reduced_total_loss.item(), batch_size)
                        meter_top1.update(reduced_prec1.item(), batch_size)
                        
                        _tqdm.set_postfix(
                            loss=f"{meter_loss.avg:.4f}",
                            acc=f"{meter_top1.avg:.2f}"
                        )
                        _tqdm.update(1)
            
            # ============ Log training results ============
            if self.rank == 0:
                # Calculate train FLOPs
                self.student.ticket = False
                train_flops = self.student.get_flops()
                
                self.logger.info(
                    f"[Train] Epoch {epoch} : Gumbel_temp {current_gumbel_temp:.2f} "
                    f"LR {current_lr:.6f} OriLoss {meter_oriloss.avg:.4f} "
                    f"KDLoss {meter_kdloss.avg:.4f} RCLoss {meter_rcloss.avg:.6f} "
                    f"MaskLoss {meter_maskloss.avg:.6f} TotalLoss {meter_loss.avg:.4f} "
                    f"Train_Acc {meter_top1.avg:.2f}"
                )
                self.logger.info(f"[Train model Flops] Epoch {epoch} : {train_flops/1e6:.6f}M")
            
            # ============ Validation (runs on ALL GPUs) ============
            val_acc, mask_avgs, val_flops = self.validate(epoch)
            
            # ============ Update schedulers & save checkpoints (rank 0 only) ============
            if self.rank == 0:
                # Update schedulers
                self.scheduler_student_weight.step()
                self.scheduler_student_mask.step()
                
                # TensorBoard logging
                self.writer.add_scalar("train/lr", current_lr, epoch)
                self.writer.add_scalar("train/gumbel_temp", current_gumbel_temp, epoch)
                self.writer.add_scalar("train/acc", meter_top1.avg, epoch)
                self.writer.add_scalar("train/loss", meter_loss.avg, epoch)
                self.writer.add_scalar("train/ori_loss", meter_oriloss.avg, epoch)
                self.writer.add_scalar("train/kd_loss", meter_kdloss.avg, epoch)
                self.writer.add_scalar("train/rc_loss", meter_rcloss.avg, epoch)
                self.writer.add_scalar("train/mask_loss", meter_maskloss.avg, epoch)
                self.writer.add_scalar("train/flops", train_flops, epoch)
                self.writer.add_scalar("val/acc", val_acc, epoch)
                self.writer.add_scalar("val/flops", val_flops, epoch)
                
                # Save checkpoint
                if val_acc > self.best_prec1:
                    self.best_prec1 = val_acc
                    self.logger.info(f" => Best top1 accuracy on validation: {self.best_prec1:.2f}")
                    self.save_student_ckpt(is_best=True, epoch=epoch)
                else:
                    self.save_student_ckpt(is_best=False, epoch=epoch)
        
        if self.rank == 0:
            self.logger.info("Training finished!")
            self.writer.close()
    def main(self):
        self.dist_init()
        self.device = torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')
        self.result_init()
        self.setup_seed()
        self.dataload()
        self.build_model()
        self.define_loss()
        self.define_optim()
        self.train()
