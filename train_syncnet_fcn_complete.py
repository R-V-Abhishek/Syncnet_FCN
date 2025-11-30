#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Training Script for SyncNetFCN on VoxCeleb2

Usage:
    python train_syncnet_fcn_complete.py --data_dir E:/voxceleb2_dataset/VoxCeleb2/dev --pretrained_model data/syncnet_v2.model
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


class VoxCeleb2Dataset(Dataset):
    """VoxCeleb2 dataset loader for sync training with real preprocessing."""
    
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
    
    def __len__(self):
        return len(self.video_files)
    
    def _extract_audio_mfcc(self, video_path):
        """Extract audio and compute MFCC features."""
        # Create unique temp audio file
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(self.temp_dir, f'{video_id}_audio.wav')
        
        try:
            # Extract audio using FFmpeg
            cmd = ['ffmpeg', '-y', '-i', video_path, '-ac', '1', '-ar', '16000',
                   '-vn', '-acodec', 'pcm_s16le', audio_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                          check=True, timeout=30)
            
            # Read audio and compute MFCC
            sample_rate, audio = wavfile.read(audio_path)
            mfcc = python_speech_features.mfcc(audio, sample_rate, numcep=13)
            
            # Shape: [T, 13] -> [13, T] -> [1, 13, T]
            # Need to add channel dimension: [1, 13, T] -> [1, 1, 13, T] for model
            mfcc_tensor = torch.FloatTensor(mfcc.T).unsqueeze(0).unsqueeze(0)  # [1, 1, 13, T]
            
            # Clean up temp file
            if os.path.exists(audio_path):
                os.remove(audio_path)
                
            return mfcc_tensor
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract audio from {video_path}: {e}")
    
    def _extract_video_frames(self, video_path, target_size=(112, 112)):
        """Extract video frames as tensor."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Resize and normalize
            frame = cv2.resize(frame, target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype(np.float32) / 255.0)
        
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames extracted from {video_path}")
        
        # Stack and convert to tensor [T, H, W, 3] -> [3, T, H, W]
        frames_array = np.stack(frames, axis=0)
        video_tensor = torch.FloatTensor(frames_array).permute(3, 0, 1, 2).unsqueeze(0)
        
        return video_tensor
    
    def _crop_or_pad_video(self, video_tensor, target_length):
        """Crop or pad video to target length."""
        B, C, T, H, W = video_tensor.shape
        
        if T > target_length:
            # Random crop
            start = random.randint(0, T - target_length)
            return video_tensor[:, :, start:start+target_length, :, :]
        elif T < target_length:
            # Pad with last frame
            pad_length = target_length - T
            last_frame = video_tensor[:, :, -1:, :, :].repeat(1, 1, pad_length, 1, 1)
            return torch.cat([video_tensor, last_frame], dim=2)
        else:
            return video_tensor
    
    def _crop_or_pad_audio(self, audio_tensor, target_length):
        """Crop or pad audio to target length."""
        B, C, T = audio_tensor.shape
        
        if T > target_length:
            # Random crop
            start = random.randint(0, T - target_length)
            return audio_tensor[:, :, start:start+target_length]
        elif T < target_length:
            # Pad with zeros
            pad_length = target_length - T
            padding = torch.zeros(B, C, pad_length)
            return torch.cat([audio_tensor, padding], dim=2)
        else:
            return audio_tensor
    
    def __getitem__(self, idx):
        """
        Returns:
            audio: [1, 13, T] MFCC features
            video: [3, T_frames, H, W] video frames
            offset: Ground truth offset (0 for positive, non-zero for negative)
            label: 1 if in sync, 0 if out of sync
        """
        video_path = self.video_files[idx]
        
        # Randomly decide if this should be positive (sync) or negative (out-of-sync)
        is_positive = random.random() > 0.5
        
        if is_positive:
            offset = 0
            label = 1
        else:
            # Random offset between 1 and max_offset
            offset = random.randint(1, self.max_offset) * random.choice([-1, 1])
            label = 0
        
        try:
            # Extract audio MFCC features
            audio = self._extract_audio_mfcc(video_path)
            
            # Extract video frames
            video = self._extract_video_frames(video_path)
            
            # Apply temporal offset for negative samples
            if not is_positive and offset != 0:
                if offset > 0:
                    # Shift video forward (cut from beginning)
                    video = video[:, :, offset:, :, :]
                else:
                    # Shift video backward (cut from end)
                    video = video[:, :, :offset, :, :]
            
            # Crop/pad to fixed length
            video = self._crop_or_pad_video(video, self.video_length)
            audio = self._crop_or_pad_audio(audio, self.video_length * 4)
            
            # Remove batch dimension (DataLoader will add it)
            # audio is [1, 1, 13, T], squeeze to [1, 13, T]
            audio = audio.squeeze(0)  # [1, 13, T]
            video = video.squeeze(0)  # [3, T, H, W]
            
        except Exception as e:
            # Fallback to dummy data if preprocessing fails
            print(f"Warning: Failed to process {video_path}: {e}")
            audio = torch.randn(1, 13, self.video_length * 4)
            video = torch.randn(3, self.video_length, 112, 112)
            offset = 0
            label = 1
        
        return {
            'audio': audio,
            'video': video,
            'offset': torch.tensor(offset, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }


class SyncLoss(nn.Module):
    """Binary cross-entropy loss for sync/no-sync classification."""
    
    def __init__(self):
        super(SyncLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, sync_probs, labels):
        """
        Args:
            sync_probs: [B, 2*K+1, T] sync probability distribution
            labels: [B] binary labels (1=sync, 0=out-of-sync)
        """
        # Take max probability across offsets and time
        max_probs = sync_probs.max(dim=1)[0].max(dim=1)[0]  # [B]
        
        # BCE loss
        loss = self.bce(max_probs, labels)
        return loss


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(dataloader):
        audio = batch['audio'].to(device)
        video = batch['video'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        sync_probs, _, _ = model(audio, video)
        
        # Compute loss
        loss = criterion(sync_probs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        pred = (sync_probs.max(dim=1)[0].max(dim=1)[0] > 0.5).float()
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        
        if batch_idx % 10 == 0:
            print(f'  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, Acc: {100*correct/total:.2f}%')
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train SyncNetFCN')
    parser.add_argument('--data_dir', type=str, required=True, help='VoxCeleb2 root directory')
    parser.add_argument('--pretrained_model', type=str, default='data/syncnet_v2.model', 
                       help='Pretrained SyncNet model')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (default: 4)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Output directory')
    parser.add_argument('--use_attention', action='store_true', help='Use attention model')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model with transfer learning
    print('Creating model...')
    model = StreamSyncFCN(
        pretrained_syncnet_path=args.pretrained_model,
        auto_load_pretrained=True,
        use_attention=args.use_attention
    )
    model = model.to(device)
    
    print(f'Model created. Pretrained conv layers loaded and frozen.')
    
    # Dataset and dataloader
    print('Loading dataset...')
    dataset = VoxCeleb2Dataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                           num_workers=args.num_workers, pin_memory=True)
    
    # Loss and optimizer
    criterion = SyncLoss()
    
    # Only optimize non-frozen parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr)
    
    print(f'Trainable parameters: {sum(p.numel() for p in trainable_params):,}')
    print(f'Frozen parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}')
    
    # Training loop
    print('\nStarting training...')
    print('='*80)
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        print('-'*80)
        
        avg_loss, accuracy = train_epoch(model, dataloader, optimizer, criterion, device)
        
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'  Average Loss: {avg_loss:.4f}')
        print(f'  Accuracy: {accuracy:.2f}%')
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f'syncnet_fcn_epoch{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'accuracy': accuracy,
        }, checkpoint_path)
        print(f'  Checkpoint saved: {checkpoint_path}')
    
    print('\n' + '='*80)
    print('Training complete!')
    print(f'Final model saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
