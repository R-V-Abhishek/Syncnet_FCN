#!/usr/bin/env python
"""
Continue training from epoch 2 checkpoint.

This script resumes training from checkpoints/syncnet_fcn_epoch2.pth
which uses SyncNet_TransferLearning with 31-class classification (±15 frames).

Usage:
    python train_continue_epoch2.py --data_dir "E:\voxc2\vox2_dev_mp4_partaa~\dev\mp4" --hours 5
"""

import os
import sys
import argparse
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import subprocess
from scipy.io import wavfile
import python_speech_features

from SyncNet_TransferLearning import SyncNet_TransferLearning


class AVSyncDataset(Dataset):
    """Dataset for audio-video sync classification."""
    
    def __init__(self, video_dir, max_offset=15, num_samples_per_video=2,
                 frame_size=(112, 112), num_frames=25, max_videos=None):
        self.video_dir = video_dir
        self.max_offset = max_offset
        self.num_samples_per_video = num_samples_per_video
        self.frame_size = frame_size
        self.num_frames = num_frames
        
        # Find all video files
        self.video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            self.video_files.extend(Path(video_dir).glob(f'**/{ext}'))
        
        # Limit number of videos if specified
        if max_videos and len(self.video_files) > max_videos:
            np.random.shuffle(self.video_files)
            self.video_files = self.video_files[:max_videos]
        
        if not self.video_files:
            raise ValueError(f"No video files found in {video_dir}")
        
        print(f"Using {len(self.video_files)} video files")
        
        # Generate sample list
        self.samples = []
        for vid_idx in range(len(self.video_files)):
            for _ in range(num_samples_per_video):
                offset = np.random.randint(-max_offset, max_offset + 1)
                self.samples.append((vid_idx, offset))
        
        print(f"Generated {len(self.samples)} training samples")
    
    def __len__(self):
        return len(self.samples)
    
    def extract_features(self, video_path):
        """Extract audio MFCC and video frames."""
        video_path = str(video_path)
        
        # Extract audio
        temp_audio = f'temp_audio_{os.getpid()}_{np.random.randint(10000)}.wav'
        try:
            cmd = ['ffmpeg', '-y', '-i', video_path, '-ac', '1', '-ar', '16000',
                   '-vn', '-acodec', 'pcm_s16le', temp_audio]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            sample_rate, audio = wavfile.read(temp_audio)
            
            # Validate audio length
            min_audio_samples = (self.num_frames * 4 + self.max_offset * 4) * 160
            if len(audio) < min_audio_samples:
                raise ValueError(f"Audio too short: {len(audio)} samples")
            
            mfcc = python_speech_features.mfcc(audio, sample_rate, numcep=13)
            
            min_mfcc_frames = self.num_frames * 4 + abs(self.max_offset) * 4
            if len(mfcc) < min_mfcc_frames:
                raise ValueError(f"MFCC too short: {len(mfcc)} frames")
        finally:
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
        
        # Extract video frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < self.num_frames + abs(self.max_offset) + 10:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype(np.float32) / 255.0)
        cap.release()
        
        if len(frames) < self.num_frames + abs(self.max_offset):
            raise ValueError(f"Video too short: {len(frames)} frames")
        
        return mfcc, np.stack(frames)
    
    def apply_offset(self, mfcc, frames, offset):
        """Apply temporal offset between audio and video."""
        mfcc_offset = offset * 4
        
        num_video_frames = min(self.num_frames, len(frames) - abs(offset))
        num_mfcc_frames = num_video_frames * 4
        
        if offset >= 0:
            video_start = 0
            mfcc_start = mfcc_offset
        else:
            video_start = abs(offset)
            mfcc_start = 0
        
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
            
            audio_tensor = torch.FloatTensor(mfcc.T).unsqueeze(0)  # [1, 13, T]
            video_tensor = torch.FloatTensor(frames).permute(3, 0, 1, 2)  # [3, T, H, W]
            offset_tensor = torch.tensor(offset, dtype=torch.long)
            
            return audio_tensor, video_tensor, offset_tensor
        except Exception as e:
            return None


def collate_fn_skip_none(batch):
    """Skip None samples."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    audio = torch.stack([b[0] for b in batch])
    video = torch.stack([b[1] for b in batch])
    offset = torch.stack([b[2] for b in batch])
    return audio, video, offset


def train_epoch(model, dataloader, criterion, optimizer, device, max_offset):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if batch is None:
            continue
        
        audio, video, target_offset = batch
        audio = audio.to(device)
        video = video.to(device)
        target_class = (target_offset + max_offset).long().to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        sync_probs, _, _ = model(audio, video)
        
        # Global average pooling over time
        sync_logits = torch.log(sync_probs + 1e-8).mean(dim=2)  # [B, 31]
        
        # Compute loss
        loss = criterion(sync_logits, target_class)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * audio.size(0)
        predicted_class = sync_logits.argmax(dim=1)
        total_correct += (predicted_class == target_class).sum().item()
        total_samples += audio.size(0)
        
        if batch_idx % 10 == 0:
            acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0
            print(f"  Batch {batch_idx}/{len(dataloader)}: Loss={loss.item():.4f}, Acc={acc:.2f}%")
    
    return total_loss / total_samples, total_correct / total_samples


def main():
    parser = argparse.ArgumentParser(description='Continue training from epoch 2')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/syncnet_fcn_epoch2.pth')
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--hours', type=float, default=5.0, help='Training time in hours')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_videos', type=int, default=None, 
                       help='Limit number of videos (for faster training)')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    max_offset = 15  # ±15 frames, 31 classes
    
    # Create model
    print("Creating model...")
    model = SyncNet_TransferLearning(
        video_backbone='fcn',
        audio_backbone='fcn', 
        embedding_dim=512,
        max_offset=max_offset,
        freeze_backbone=False
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Load model state
    model_state = checkpoint['model_state_dict']
    # Remove 'fcn_model.' prefix if present
    new_state = {}
    for k, v in model_state.items():
        if k.startswith('fcn_model.'):
            new_state[k[10:]] = v  # Remove 'fcn_model.' prefix
        else:
            new_state[k] = v
    
    model.load_state_dict(new_state, strict=False)
    start_epoch = checkpoint.get('epoch', 2)
    print(f"Resuming from epoch {start_epoch}")
    
    model = model.to(device)
    
    # Dataset
    print("Loading dataset...")
    dataset = AVSyncDataset(
        video_dir=args.data_dir,
        max_offset=max_offset,
        num_samples_per_video=2,
        max_videos=args.max_videos
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_skip_none,
        pin_memory=True
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Training loop with time limit
    os.makedirs(args.output_dir, exist_ok=True)
    
    max_seconds = args.hours * 3600
    start_time = time.time()
    epoch = start_epoch
    best_acc = 0
    
    print(f"\n{'='*60}")
    print(f"Starting training for {args.hours} hours...")
    print(f"{'='*60}")
    
    while True:
        elapsed = time.time() - start_time
        remaining = max_seconds - elapsed
        
        if remaining <= 0:
            print(f"\nTime limit reached ({args.hours} hours)")
            break
        
        epoch += 1
        print(f"\nEpoch {epoch} (Time remaining: {remaining/3600:.2f} hours)")
        print("-" * 40)
        
        train_loss, train_acc = train_epoch(
            model, dataloader, criterion, optimizer, device, max_offset
        )
        
        print(f"Epoch {epoch}: Loss={train_loss:.4f}, Acc={100*train_acc:.2f}%")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f'syncnet_fcn_epoch{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'accuracy': train_acc * 100,
        }, checkpoint_path)
        print(f"Saved: {checkpoint_path}")
        
        # Save best
        if train_acc > best_acc:
            best_acc = train_acc
            best_path = os.path.join(args.output_dir, 'syncnet_fcn_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'accuracy': train_acc * 100,
            }, best_path)
            print(f"New best model saved: {best_path}")
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Final epoch: {epoch}")
    print(f"Best accuracy: {100*best_acc:.2f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
