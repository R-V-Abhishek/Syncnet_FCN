#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
IMPROVED Training Script for SyncNetFCN on VoxCeleb2

Key Fixes:
1. Corrected loss function: CrossEntropyLoss for offset prediction (31 classes)
2. Removed dummy data fallback
3. Reduced logging overhead
4. Added proper metrics tracking (exact accuracy, ±1 frame accuracy, MAE)
5. Added temporal consistency regularization
6. Better learning rate scheduling

Usage:
    python train_syncnet_fcn_improved.py --data_dir E:/voxceleb2_dataset/VoxCeleb2/dev --pretrained_model data/syncnet_v2.model --checkpoint checkpoints/syncnet_fcn_epoch2.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import numpy as np
from SyncNetModel_FCN import StreamSyncFCN
import glob
import random
import cv2
import subprocess
from scipy.io import wavfile
import python_speech_features
import time


class VoxCeleb2DatasetImproved(Dataset):
    """Improved VoxCeleb2 dataset loader with fixed label format and no dummy data."""
    
    def __init__(self, data_dir, max_offset=15, video_length=25, temp_dir='temp_dataset'):
        """
        Args:
            data_dir: Path to VoxCeleb2 root directory
            max_offset: Maximum frame offset for negative samples
            video_length: Number of frames per clip
            temp_dir: Temporary directory for audio extraction
        """
        self.data_dir = data_dir
        self.max_offset = max_offset
        self.video_length = video_length
        self.temp_dir = temp_dir
        
        os.makedirs(temp_dir, exist_ok=True)
        
        # Find all video files
        self.video_files = glob.glob(os.path.join(data_dir, '**', '*.mp4'), recursive=True)
        print(f"Found {len(self.video_files)} videos in dataset")
        
        # Track failed samples
        self.failed_samples = set()
    
    def __len__(self):
        return len(self.video_files)
    
    def _extract_audio_mfcc(self, video_path):
        """Extract audio and compute MFCC features."""
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(self.temp_dir, f'{video_id}_audio.wav')
        
        try:
            # Extract audio using FFmpeg
            cmd = ['ffmpeg', '-y', '-i', video_path, '-ac', '1', '-ar', '16000',
                   '-vn', '-acodec', 'pcm_s16le', audio_path]
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed")
            
            # Read audio and compute MFCC
            sample_rate, audio = wavfile.read(audio_path)
            
            # Ensure audio is 1D
            if isinstance(audio, np.ndarray) and len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            if not isinstance(audio, np.ndarray) or audio.size == 0:
                raise ValueError(f"Audio data is empty")
            
            # Compute MFCC
            mfcc = python_speech_features.mfcc(audio, sample_rate, numcep=13)
            mfcc_tensor = torch.FloatTensor(mfcc.T).unsqueeze(0).unsqueeze(0)
            
            # Clean up temp file
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception:
                    pass
            
            return mfcc_tensor
            
        except Exception as e:
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception:
                    pass
            raise RuntimeError(f"Failed to extract audio: {e}")
    
    def _extract_video_frames(self, video_path, target_size=(112, 112)):
        """Extract video frames as tensor."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype(np.float32) / 255.0)
        
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames extracted")
        
        frames_array = np.stack(frames, axis=0)
        video_tensor = torch.FloatTensor(frames_array).permute(3, 0, 1, 2).unsqueeze(0)
        
        return video_tensor
    
    def _crop_or_pad_video(self, video_tensor, target_length):
        """Crop or pad video to target length."""
        B, C, T, H, W = video_tensor.shape
        
        if T > target_length:
            start = random.randint(0, T - target_length)
            return video_tensor[:, :, start:start+target_length, :, :]
        elif T < target_length:
            pad_length = target_length - T
            last_frame = video_tensor[:, :, -1:, :, :].repeat(1, 1, pad_length, 1, 1)
            return torch.cat([video_tensor, last_frame], dim=2)
        else:
            return video_tensor
    
    def _crop_or_pad_audio(self, audio_tensor, target_length):
        """Crop or pad audio to target length."""
        B, C, F, T = audio_tensor.shape
        
        if T > target_length:
            start = random.randint(0, T - target_length)
            return audio_tensor[:, :, :, start:start+target_length]
        elif T < target_length:
            pad_length = target_length - T
            padding = torch.zeros(B, C, F, pad_length)
            return torch.cat([audio_tensor, padding], dim=3)
        else:
            return audio_tensor
    
    def __getitem__(self, idx):
        """
        Returns:
            audio: [1, 13, T] MFCC features
            video: [3, T_frames, H, W] video frames
            offset: Ground truth offset in frames (integer from -15 to +15)
        """
        video_path = self.video_files[idx]
        
        # Skip previously failed samples
        if idx in self.failed_samples:
            return self.__getitem__((idx + 1) % len(self))
        
        # Balanced offset distribution
        # 20% synced (offset=0), 80% distributed across other offsets
        if random.random() < 0.2:
            offset = 0
        else:
            # Exclude 0 from choices
            offset_choices = [o for o in range(-self.max_offset, self.max_offset + 1) if o != 0]
            offset = random.choice(offset_choices)
        
        # Log occasionally (every 1000 samples instead of random 1%)
        if idx % 1000 == 0:
            print(f"[INFO] Processing sample {idx}: offset={offset}")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Extract audio MFCC features
                audio = self._extract_audio_mfcc(video_path)
                
                # Extract video frames
                video = self._extract_video_frames(video_path)
                
                # Apply temporal offset for negative samples
                if offset != 0:
                    if offset > 0:
                        # Shift video forward (cut from beginning)
                        video = video[:, :, offset:, :, :]
                    else:
                        # Shift video backward (cut from end)
                        video = video[:, :, :offset, :, :]
                
                # Crop/pad to fixed length
                video = self._crop_or_pad_video(video, self.video_length)
                audio = self._crop_or_pad_audio(audio, self.video_length * 4)
                
                # Remove batch dimension
                audio = audio.squeeze(0)  # [1, 13, T]
                video = video.squeeze(0)  # [3, T, H, W]
                
                # Validate shapes
                if audio.shape[0] != 1 or audio.shape[1] != 13:
                    raise ValueError(f"Audio MFCC shape mismatch: {audio.shape}")
                if audio.shape[2] != self.video_length * 4:
                     # Force fix length if mismatch (should be handled by crop_or_pad but double check)
                     audio = self._crop_or_pad_audio(audio.unsqueeze(0), self.video_length * 4).squeeze(0)
                
                if video.shape[0] != 3 or video.shape[2] != 112 or video.shape[3] != 112:
                    raise ValueError(f"Video frame shape mismatch: {video.shape}")
                if video.shape[1] != self.video_length:
                    # Force fix length
                    video = self._crop_or_pad_video(video.unsqueeze(0), self.video_length).squeeze(0)

                # Final check
                if audio.shape != (1, 13, 100) or video.shape != (3, 25, 112, 112):
                     raise ValueError(f"Final shape mismatch: Audio {audio.shape}, Video {video.shape}")
                
                return {
                    'audio': audio,
                    'video': video,
                    'offset': torch.tensor(offset, dtype=torch.long),  # Integer offset, not binary
                }
                
            except Exception as e:
                if attempt == max_retries - 1:
                    # Mark as failed and try next sample
                    self.failed_samples.add(idx)
                    if idx % 100 == 0:  # Only log occasionally
                        print(f"[WARN] Sample {idx} failed after {max_retries} attempts: {str(e)[:100]}")
                    return self.__getitem__((idx + 1) % len(self))
                continue


class OffsetRegressionLoss(nn.Module):
    """L1 regression loss for continuous offset prediction."""
    
    def __init__(self):
        super(OffsetRegressionLoss, self).__init__()
        self.l1 = nn.L1Loss()  # More robust to outliers than MSE
    
    def forward(self, predicted_offsets, target_offsets):
        """
        Args:
            predicted_offsets: [B, 1, T] - model output (continuous offset predictions)
            target_offsets: [B] - ground truth offset in frames (float)
        
        Returns:
            loss: scalar
        """
        B, C, T = predicted_offsets.shape
        
        # Average over time dimension
        predicted_offsets_avg = predicted_offsets.mean(dim=2).squeeze(1)  # [B]
        
        # L1 loss
        loss = self.l1(predicted_offsets_avg, target_offsets.float())
        
        return loss


def temporal_consistency_loss(predicted_offsets):
    """
    Encourage smooth predictions over time.
    
    Args:
        predicted_offsets: [B, 1, T]
    
    Returns:
        consistency_loss: scalar
    """
    # Compute difference between adjacent timesteps
    temporal_diff = predicted_offsets[:, :, 1:] - predicted_offsets[:, :, :-1]
    consistency_loss = (temporal_diff ** 2).mean()
    return consistency_loss


def compute_metrics(predicted_offsets, target_offsets, max_offset=125):
    """
    Compute comprehensive metrics for offset regression.
    
    Args:
        predicted_offsets: [B, 1, T]
        target_offsets: [B]
    
    Returns:
        dict with metrics
    """
    B, C, T = predicted_offsets.shape
    
    # Average over time
    predicted_offsets_avg = predicted_offsets.mean(dim=2).squeeze(1)  # [B]
    
    # Mean absolute error
    mae = torch.abs(predicted_offsets_avg - target_offsets).mean()
    
    # Root mean squared error
    rmse = torch.sqrt(((predicted_offsets_avg - target_offsets) ** 2).mean())
    
    # Error buckets
    acc_1frame = (torch.abs(predicted_offsets_avg - target_offsets) <= 1).float().mean()
    acc_1sec = (torch.abs(predicted_offsets_avg - target_offsets) <= 25).float().mean()
    
    # Strict Sync Score (1 - error/25_frames)
    # 1.0 = perfect sync
    # 0.0 = >1 second error (unusable)
    abs_error = torch.abs(predicted_offsets_avg - target_offsets)
    sync_score = 1.0 - (abs_error / 25.0) # 25 frames = 1 second
    sync_score = torch.clamp(sync_score, 0.0, 1.0).mean()
    
    return {
        'mae': mae.item(),
        'rmse': rmse.item(),
        'acc_1frame': acc_1frame.item(),
        'acc_1sec': acc_1sec.item(),
        'sync_score': sync_score.item()
    }


def train_epoch(model, dataloader, optimizer, criterion, device, epoch_num):
    """Train for one epoch with regression metrics."""
    model.train()
    total_loss = 0
    total_offset_loss = 0
    total_consistency_loss = 0
    
    metrics_accum = {'mae': 0, 'rmse': 0, 'acc_1frame': 0, 'acc_1sec': 0, 'sync_score': 0}
    num_batches = 0
    
    import gc
    for batch_idx, batch in enumerate(dataloader):
        audio = batch['audio'].to(device)
        video = batch['video'].to(device)
        offsets = batch['offset'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predicted_offsets, _, _ = model(audio, video)
        
        # Compute losses
        offset_loss = criterion(predicted_offsets, offsets)
        consistency_loss = temporal_consistency_loss(predicted_offsets)
        
        # Combined loss
        loss = offset_loss + 0.1 * consistency_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        total_offset_loss += offset_loss.item()
        total_consistency_loss += consistency_loss.item()
        
        # Compute metrics
        with torch.no_grad():
            metrics = compute_metrics(predicted_offsets, offsets)
            for key in metrics_accum:
                metrics_accum[key] += metrics[key]
        
        num_batches += 1
        
        # Log every 10 batches
        if batch_idx % 10 == 0:
            print(f'  Batch {batch_idx}/{len(dataloader)}, '
                  f'Loss: {loss.item():.4f}, '
                  f'MAE: {metrics["mae"]:.2f} frames, '
                  f'Score: {metrics["sync_score"]:.4f}')
        
        # Clean up
        del audio, video, offsets, predicted_offsets
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Average metrics
    avg_loss = total_loss / num_batches
    avg_offset_loss = total_offset_loss / num_batches
    avg_consistency_loss = total_consistency_loss / num_batches
    
    for key in metrics_accum:
        metrics_accum[key] /= num_batches
    
    return avg_loss, avg_offset_loss, avg_consistency_loss, metrics_accum


def main():
    parser = argparse.ArgumentParser(description='Train SyncNetFCN (Improved)')
    parser.add_argument('--data_dir', type=str, required=True, help='VoxCeleb2 root directory')
    parser.add_argument('--pretrained_model', type=str, default='data/syncnet_v2.model', 
                       help='Pretrained SyncNet model')
    parser.add_argument('--checkpoint', type=str, default=None, 
                       help='Resume from checkpoint (optional)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (default: 4)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate (lowered from 0.001)')
    parser.add_argument('--output_dir', type=str, default='checkpoints_improved', help='Output directory')
    parser.add_argument('--use_attention', action='store_true', help='Use attention model')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument('--max_offset', type=int, default=125, help='Max offset in frames (default: 125)')
    parser.add_argument('--unfreeze_epoch', type=int, default=10, help='Epoch to unfreeze all layers (default: 10)')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model with transfer learning (max_offset=125 for ±5 seconds)
    print(f'Creating model with max_offset={args.max_offset}...')
    model = StreamSyncFCN(
        max_offset=args.max_offset,  # ±5 seconds at 25fps
        pretrained_syncnet_path=args.pretrained_model,
        auto_load_pretrained=True,
        use_attention=args.use_attention
    )
    
    # Load from checkpoint if provided
    start_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f'Loading checkpoint: {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f'Resuming from epoch {start_epoch}')
    
    model = model.to(device)
    print(f'Model created. Pretrained conv layers loaded and frozen.')
    
    # Dataset and dataloader
    print(f'Loading dataset with max_offset={args.max_offset}...')
    dataset = VoxCeleb2DatasetImproved(args.data_dir, max_offset=args.max_offset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                           num_workers=args.num_workers, pin_memory=True)
    
    # Loss and optimizer (REGRESSION)
    criterion = OffsetRegressionLoss()
    
    # Only optimize non-frozen parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=5,       # Restart every 5 epochs
        T_mult=2,    # Double restart period each time
        eta_min=1e-7 # Minimum LR
    )
    
    print(f'Trainable parameters: {sum(p.numel() for p in trainable_params):,}')
    print(f'Frozen parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}')
    print(f'Learning rate: {args.lr}')
    
    # Training loop
    print('\\nStarting training...')
    print('='*80)
    
    best_tolerance_acc = 0
    
    for epoch in range(start_epoch, start_epoch + args.epochs):
        print(f'\\nEpoch {epoch+1}/{start_epoch + args.epochs}')
        print('-'*80)
        
        # Unfreeze layers if reached unfreeze_epoch
        if epoch + 1 == args.unfreeze_epoch:
            print(f'\\n🔓 Unfreezing all layers for fine-tuning at epoch {epoch+1}...')
            model.unfreeze_all_layers()
            
            # Lower learning rate for fine-tuning
            new_lr = args.lr * 0.1
            print(f'📉 Lowering learning rate to {new_lr} for fine-tuning')
            
            # Re-initialize optimizer with all parameters
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.Adam(trainable_params, lr=new_lr)
            
            # Re-initialize scheduler
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=5, T_mult=2, eta_min=1e-8
            )
            print(f'Trainable parameters now: {sum(p.numel() for p in trainable_params):,}')
        
        avg_loss, avg_offset_loss, avg_consistency_loss, metrics = train_epoch(
            model, dataloader, optimizer, criterion, device, epoch
        )
        
        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'  Total Loss: {avg_loss:.4f}')
        print(f'  Offset Loss: {avg_offset_loss:.4f}')
        print(f'  Consistency Loss: {avg_consistency_loss:.4f}')
        print(f'  MAE: {metrics["mae"]:.2f} frames ({metrics["mae"]/25:.3f} seconds)')
        print(f'  RMSE: {metrics["rmse"]:.2f} frames')
        print(f'  Sync Score: {metrics["sync_score"]:.4f} (1.0=Perfect, 0.0=>1s Error)')
        print(f'  <1 Frame Acc: {metrics["acc_1frame"]*100:.2f}%')
        print(f'  <1 Second Acc: {metrics["acc_1sec"]*100:.2f}%')
        print(f'  Learning Rate: {current_lr:.2e}')
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f'syncnet_fcn_improved_epoch{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'offset_loss': avg_offset_loss,
            'metrics': metrics,
        }, checkpoint_path)
        print(f'  Checkpoint saved: {checkpoint_path}')
        
        # Save best model based on Sync Score
        if metrics['sync_score'] > best_tolerance_acc:
            best_tolerance_acc = metrics['sync_score']
            best_path = os.path.join(args.output_dir, 'syncnet_fcn_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'metrics': metrics,
            }, best_path)
            print(f'  ✓ New best model saved! (Score: {best_tolerance_acc:.4f})')
    
    print('\n' + '='*80)
    print('Training complete!')
    print(f'Best Sync Score: {best_tolerance_acc:.4f}')
    print(f'Models saved to: {args.output_dir}')


if __name__ == '__main__':
    main()

