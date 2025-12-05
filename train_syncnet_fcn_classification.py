#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script for FCN-SyncNet CLASSIFICATION model.

Key differences from regression training:
- Uses CrossEntropyLoss instead of MSE
- Treats offset as discrete classes (-15 to +15 = 31 classes)
- Tracks classification accuracy as primary metric
- Avoids regression-to-mean problem

Usage:
    python train_syncnet_fcn_classification.py --data_dir /path/to/dataset
    python train_syncnet_fcn_classification.py --data_dir /path/to/dataset --epochs 50 --lr 1e-4
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import subprocess
from scipy.io import wavfile
import python_speech_features
import cv2
from pathlib import Path

from SyncNetModel_FCN_Classification import (
    SyncNetFCN_Classification,
    StreamSyncFCN_Classification,
    create_classification_criterion,
    train_step_classification,
    validate_classification
)


class AVSyncDataset(Dataset):
    """
    Dataset for audio-video sync classification.
    
    Generates training samples with artificial offsets for data augmentation.
    """
    
    def __init__(self, video_dir, max_offset=15, num_samples_per_video=10,
                 frame_size=(112, 112), num_frames=25, cache_features=True):
        """
        Args:
            video_dir: Directory containing video files
            max_offset: Maximum offset in frames (creates 2*max_offset+1 classes)
            num_samples_per_video: Number of samples to generate per video
            frame_size: Target frame size (H, W)
            num_frames: Number of frames per sample
            cache_features: Cache extracted features for faster training
        """
        self.video_dir = video_dir
        self.max_offset = max_offset
        self.num_samples_per_video = num_samples_per_video
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.cache_features = cache_features
        self.feature_cache = {}
        
        # Find all video files
        self.video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.mpg', '*.mpeg']:
            self.video_files.extend(Path(video_dir).glob(f'**/{ext}'))
        
        if not self.video_files:
            raise ValueError(f"No video files found in {video_dir}")
        
        print(f"Found {len(self.video_files)} video files")
        
        # Generate sample list (video_idx, offset)
        self.samples = []
        for vid_idx in range(len(self.video_files)):
            for _ in range(num_samples_per_video):
                # Random offset within range
                offset = np.random.randint(-max_offset, max_offset + 1)
                self.samples.append((vid_idx, offset))
        
        print(f"Generated {len(self.samples)} training samples")
    
    def __len__(self):
        return len(self.samples)
    
    def extract_features(self, video_path):
        """Extract audio MFCC and video frames."""
        video_path = str(video_path)
        
        # Check cache
        if self.cache_features and video_path in self.feature_cache:
            return self.feature_cache[video_path]
        
        # Extract audio
        temp_audio = f'temp_audio_{os.getpid()}_{np.random.randint(10000)}.wav'
        try:
            cmd = ['ffmpeg', '-y', '-i', video_path, '-ac', '1', '-ar', '16000',
                   '-vn', '-acodec', 'pcm_s16le', temp_audio]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            sample_rate, audio = wavfile.read(temp_audio)
            
            # Validate audio length (need at least num_frames * 4 MFCC frames)
            min_audio_samples = (self.num_frames * 4 + self.max_offset * 4) * 160  # 160 samples per MFCC frame at 16kHz
            if len(audio) < min_audio_samples:
                raise ValueError(f"Audio too short: {len(audio)} samples, need {min_audio_samples}")
            
            mfcc = python_speech_features.mfcc(audio, sample_rate, numcep=13)
            
            # Validate MFCC length
            min_mfcc_frames = self.num_frames * 4 + abs(self.max_offset) * 4
            if len(mfcc) < min_mfcc_frames:
                raise ValueError(f"MFCC too short: {len(mfcc)} frames, need {min_mfcc_frames}")
        finally:
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
        
        # Extract video frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype(np.float32) / 255.0)
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames extracted from {video_path}")
        
        result = (mfcc, np.stack(frames))
        
        # Cache if enabled
        if self.cache_features:
            self.feature_cache[video_path] = result
        
        return result
    
    def apply_offset(self, mfcc, frames, offset):
        """
        Apply temporal offset between audio and video.
        
        Positive offset: audio is ahead (shift audio forward / video backward)
        Negative offset: video is ahead (shift video forward / audio backward)
        """
        # MFCC is at 100Hz (10ms per frame), video at 25fps (40ms per frame)
        # 1 video frame = 4 MFCC frames
        mfcc_offset = offset * 4
        
        num_video_frames = min(self.num_frames, len(frames) - abs(offset))
        num_mfcc_frames = num_video_frames * 4
        
        if offset >= 0:
            # Audio ahead: start audio later
            video_start = 0
            mfcc_start = mfcc_offset
        else:
            # Video ahead: start video later
            video_start = abs(offset)
            mfcc_start = 0
        
        # Extract aligned segments
        video_segment = frames[video_start:video_start + num_video_frames]
        mfcc_segment = mfcc[mfcc_start:mfcc_start + num_mfcc_frames]
        
        # Pad if needed
        if len(video_segment) < self.num_frames:
            pad_frames = self.num_frames - len(video_segment)
            video_segment = np.concatenate([
                video_segment,
                np.repeat(video_segment[-1:], pad_frames, axis=0)
            ], axis=0)
        
        target_mfcc_len = self.num_frames * 4
        if len(mfcc_segment) < target_mfcc_len:
            pad_mfcc = target_mfcc_len - len(mfcc_segment)
            mfcc_segment = np.concatenate([
                mfcc_segment,
                np.repeat(mfcc_segment[-1:], pad_mfcc, axis=0)
            ], axis=0)
        
        return mfcc_segment[:target_mfcc_len], video_segment[:self.num_frames]
    
    def __getitem__(self, idx):
        vid_idx, offset = self.samples[idx]
        video_path = self.video_files[vid_idx]
        
        try:
            mfcc, frames = self.extract_features(video_path)
            mfcc, frames = self.apply_offset(mfcc, frames, offset)
            
            # Convert to tensors
            audio_tensor = torch.FloatTensor(mfcc.T).unsqueeze(0)  # [1, 13, T]
            video_tensor = torch.FloatTensor(frames).permute(3, 0, 1, 2)  # [3, T, H, W]
            offset_tensor = torch.tensor(offset, dtype=torch.long)
            
            return audio_tensor, video_tensor, offset_tensor
            
        except Exception as e:
            # Return None for bad samples (filtered by collate_fn)
            return None


def collate_fn_skip_none(batch):
    """Custom collate function that skips None and invalid samples."""
    # Filter out None samples
    batch = [b for b in batch if b is not None]
    
    # Filter out samples with empty tensors (0-length MFCC from videos without audio)
    valid_batch = []
    for b in batch:
        audio, video, offset = b
        # Check if audio and video have valid sizes
        if audio.size(-1) > 0 and video.size(1) > 0:
            valid_batch.append(b)
    
    if len(valid_batch) == 0:
        # Return None if all samples are bad
        return None
    
    # Stack valid samples
    audio = torch.stack([b[0] for b in valid_batch])
    video = torch.stack([b[1] for b in valid_batch])
    offset = torch.stack([b[2] for b in valid_batch])
    
    return audio, video, offset


def train_epoch(model, dataloader, criterion, optimizer, device, max_offset):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Skip None batches (all samples were invalid)
        if batch is None:
            continue
        
        audio, video, target_offset = batch
        audio = audio.to(device)
        video = video.to(device)
        target_class = (target_offset + max_offset).long().to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if hasattr(model, 'fcn_model'):
            class_logits, _, _ = model(audio, video)
        else:
            class_logits, _, _ = model(audio, video)
        
        # Compute loss
        loss = criterion(class_logits, target_class)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * audio.size(0)
        predicted_class = class_logits.argmax(dim=1)
        total_correct += (predicted_class == target_class).sum().item()
        total_samples += audio.size(0)
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}: Loss={loss.item():.4f}, "
                  f"Acc={(predicted_class == target_class).float().mean().item():.2%}")
    
    return total_loss / total_samples, total_correct / total_samples


def validate(model, dataloader, criterion, device, max_offset):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    total_error = 0
    
    with torch.no_grad():
        for audio, video, target_offset in dataloader:
            audio = audio.to(device)
            video = video.to(device)
            target_class = (target_offset + max_offset).long().to(device)
            
            if hasattr(model, 'fcn_model'):
                class_logits, _, _ = model(audio, video)
            else:
                class_logits, _, _ = model(audio, video)
            
            loss = criterion(class_logits, target_class)
            total_loss += loss.item() * audio.size(0)
            
            predicted_class = class_logits.argmax(dim=1)
            total_correct += (predicted_class == target_class).sum().item()
            total_samples += audio.size(0)
            
            # Mean absolute error in frames
            predicted_offset = predicted_class - max_offset
            actual_offset = target_class - max_offset
            total_error += (predicted_offset - actual_offset).abs().sum().item()
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    mae = total_error / total_samples
    
    return avg_loss, accuracy, mae


def main():
    parser = argparse.ArgumentParser(description='Train FCN-SyncNet Classification Model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing training videos')
    parser.add_argument('--val_dir', type=str, default=None,
                       help='Directory containing validation videos (optional)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_classification',
                       help='Directory to save checkpoints')
    parser.add_argument('--pretrained', type=str, default='data/syncnet_v2.model',
                       help='Path to pretrained SyncNet weights')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Training parameters (defaults optimized for RTX A5000 24GB, GRID corpus)
    parser.add_argument('--epochs', type=int, default=25,
                       help='25 epochs for GRID corpus (~2.5-3 hrs on A5000)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='64 fits in A5000 24GB, faster throughput')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Higher LR for faster convergence with large dataset')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Slightly lower dropout for classification')
    
    # Model parameters
    parser.add_argument('--max_offset', type=int, default=15,
                       help='±15 frames for GRID corpus (31 classes)')
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--num_frames', type=int, default=25)
    parser.add_argument('--samples_per_video', type=int, default=5,
                       help='5 samples/video for GRID corpus')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Data loading workers (4 for RTX A5000 system)')
    
    # Training options
    parser.add_argument('--freeze_conv', action='store_true', default=True,
                       help='Freeze pretrained conv layers')
    parser.add_argument('--no_freeze_conv', dest='freeze_conv', action='store_false')
    parser.add_argument('--unfreeze_epoch', type=int, default=20,
                       help='Epoch to unfreeze conv layers for fine-tuning')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create model
    print("Creating model...")
    model = StreamSyncFCN_Classification(
        embedding_dim=args.embedding_dim,
        max_offset=args.max_offset,
        pretrained_syncnet_path=args.pretrained if os.path.exists(args.pretrained) else None,
        auto_load_pretrained=True,
        dropout=args.dropout
    )
    
    if args.freeze_conv:
        print("Conv layers frozen (will unfreeze at epoch {})".format(args.unfreeze_epoch))
    
    model = model.to(device)
    
    # Create dataset
    print("Loading dataset...")
    train_dataset = AVSyncDataset(
        video_dir=args.data_dir,
        max_offset=args.max_offset,
        num_samples_per_video=args.samples_per_video,
        num_frames=args.num_frames,
        cache_features=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=collate_fn_skip_none
    )
    
    val_loader = None
    if args.val_dir and os.path.exists(args.val_dir):
        val_dataset = AVSyncDataset(
            video_dir=args.val_dir,
            max_offset=args.max_offset,
            num_samples_per_video=5,
            num_frames=args.num_frames,
            cache_features=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False,
            persistent_workers=True if args.num_workers > 0 else False,
            collate_fn=collate_fn_skip_none
        )
    
    # Loss and optimizer
    criterion = create_classification_criterion(
        max_offset=args.max_offset,
        label_smoothing=args.label_smoothing
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Resume from checkpoint
    start_epoch = 0
    best_accuracy = 0
    
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint.get('best_accuracy', 0)
        print(f"Resumed from epoch {start_epoch}, best accuracy: {best_accuracy:.2%}")
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        # Unfreeze conv layers after specified epoch
        if args.freeze_conv and epoch == args.unfreeze_epoch:
            print("Unfreezing conv layers for fine-tuning...")
            model.unfreeze_all_layers()
        
        # Train
        start_time = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, args.max_offset
        )
        train_time = time.time() - start_time
        
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2%}, Time: {train_time:.1f}s")
        
        # Validate
        if val_loader:
            val_loss, val_acc, val_mae = validate(
                model, val_loader, criterion, device, args.max_offset
            )
            print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.2%}, MAE: {val_mae:.2f} frames")
            scheduler.step(val_acc)
            is_best = val_acc > best_accuracy
            best_accuracy = max(val_acc, best_accuracy)
        else:
            scheduler.step(train_acc)
            is_best = train_acc > best_accuracy
            best_accuracy = max(train_acc, best_accuracy)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'best_accuracy': best_accuracy
        }
        
        checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        if is_best:
            best_path = os.path.join(args.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model! Accuracy: {best_accuracy:.2%}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best accuracy: {best_accuracy:.2%}")
    print("="*60)


if __name__ == '__main__':
    main()
