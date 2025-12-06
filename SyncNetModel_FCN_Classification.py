#!/usr/bin/python
#-*- coding: utf-8 -*-

"""
Fully Convolutional SyncNet (FCN-SyncNet) - CLASSIFICATION VERSION

Key difference from regression version:
- Output: Probability distribution over discrete offset classes
- Loss: CrossEntropyLoss instead of MSE
- Avoids regression-to-mean problem

Offset classes: -15 to +15 frames (31 classes total)
Class 0 = -15 frames, Class 15 = 0 frames, Class 30 = +15 frames

Author: Enhanced version based on original SyncNet
Date: 2025-12-04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import cv2
import os
import subprocess
from scipy.io import wavfile
import python_speech_features


class TemporalCorrelation(nn.Module):
    """
    Compute correlation between audio and video features across time.
    """
    def __init__(self, max_displacement=15):
        super(TemporalCorrelation, self).__init__()
        self.max_displacement = max_displacement
        
    def forward(self, feat1, feat2):
        """
        Args:
            feat1: [B, C, T] - visual features
            feat2: [B, C, T] - audio features
        Returns:
            correlation: [B, 2*max_displacement+1, T] - correlation map
        """
        B, C, T = feat1.shape
        max_disp = self.max_displacement
        
        # Normalize features
        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1)
        
        # Pad feat2 for shifting
        feat2_padded = F.pad(feat2, (max_disp, max_disp), mode='replicate')
        
        corr_list = []
        for offset in range(-max_disp, max_disp + 1):
            shifted_feat2 = feat2_padded[:, :, offset+max_disp:offset+max_disp+T]
            corr = (feat1 * shifted_feat2).sum(dim=1, keepdim=True)
            corr_list.append(corr)
        
        correlation = torch.cat(corr_list, dim=1)
        return correlation


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, t = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class TemporalAttention(nn.Module):
    """Self-attention over temporal dimension."""
    def __init__(self, channels):
        super(TemporalAttention, self).__init__()
        self.query_conv = nn.Conv1d(channels, channels // 8, 1)
        self.key_conv = nn.Conv1d(channels, channels // 8, 1)
        self.value_conv = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, T = x.size()
        query = self.query_conv(x).permute(0, 2, 1)
        key = self.key_conv(x)
        value = self.value_conv(x)
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = self.gamma * out + x
        return out


class FCN_AudioEncoder(nn.Module):
    """Fully convolutional audio encoder."""
    def __init__(self, output_channels=512):
        super(FCN_AudioEncoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 192, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(1,2)),
            
            nn.Conv2d(192, 384, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
            
            nn.Conv2d(256, 512, kernel_size=(5,1), stride=(5,1), padding=(0,0)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        self.channel_conv = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, output_channels, kernel_size=1),
            nn.BatchNorm1d(output_channels),
        )
        
        self.channel_attn = ChannelAttention(output_channels)
        
    def forward(self, x):
        x = self.conv_layers(x)
        B, C, F, T = x.size()
        x = x.view(B, C * F, T)
        x = self.channel_conv(x)
        x = self.channel_attn(x)
        return x


class FCN_VideoEncoder(nn.Module):
    """Fully convolutional video encoder."""
    def __init__(self, output_channels=512):
        super(FCN_VideoEncoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv3d(3, 96, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3)),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            
            nn.Conv3d(96, 256, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            
            nn.Conv3d(256, 256, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(256, 256, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(256, 256, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            
            nn.Conv3d(256, 512, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((None, 1, 1))
        )
        
        self.channel_conv = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, output_channels, kernel_size=1),
            nn.BatchNorm1d(output_channels),
        )
        
        self.channel_attn = ChannelAttention(output_channels)
        
    def forward(self, x):
        x = self.conv_layers(x)
        B, C, T, H, W = x.size()
        x = x.view(B, C, T)
        x = self.channel_conv(x)
        x = self.channel_attn(x)
        return x


class SyncNetFCN_Classification(nn.Module):
    """
    Fully Convolutional SyncNet with CLASSIFICATION output.
    
    Treats offset detection as a multi-class classification problem:
    - num_classes = 2 * max_offset + 1 (e.g., 251 classes for max_offset=125)
    - Class index = offset + max_offset (e.g., offset -5 → class 120)
    - Uses CrossEntropyLoss for training
    - Default: ±125 frames = ±5 seconds at 25fps
    
    This avoids the regression-to-mean problem encountered with MSE loss.
    
    Architecture:
    1. Audio encoder: MFCC → temporal features
    2. Video encoder: frames → temporal features
    3. Correlation layer: compute audio-video similarity over time
    4. Classifier: predict offset class probabilities
    """
    def __init__(self, embedding_dim=512, max_offset=125, dropout=0.3):
        super(SyncNetFCN_Classification, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.max_offset = max_offset
        self.num_classes = 2 * max_offset + 1  # -15 to +15 = 31 classes
        
        # Encoders
        self.audio_encoder = FCN_AudioEncoder(output_channels=embedding_dim)
        self.video_encoder = FCN_VideoEncoder(output_channels=embedding_dim)
        
        # Temporal correlation
        self.correlation = TemporalCorrelation(max_displacement=max_offset)
        
        # Classifier head (replaces regressor)
        self.classifier = nn.Sequential(
            nn.Conv1d(self.num_classes, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # Output: class logits for each timestep
            nn.Conv1d(64, self.num_classes, kernel_size=1),
        )
        
        # Global classifier (for single prediction from sequence)
        self.global_classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.num_classes, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, self.num_classes),
        )
        
    def forward_audio(self, audio_mfcc):
        """Extract audio features."""
        return self.audio_encoder(audio_mfcc)
    
    def forward_video(self, video_frames):
        """Extract video features."""
        return self.video_encoder(video_frames)
    
    def forward(self, audio_mfcc, video_frames, return_temporal=False):
        """
        Forward pass with audio-video offset classification.
        
        Args:
            audio_mfcc: [B, 1, F, T] - MFCC features
            video_frames: [B, 3, T', H, W] - video frames
            return_temporal: If True, also return per-timestep predictions
            
        Returns:
            class_logits: [B, num_classes] - global offset class logits
            temporal_logits: [B, num_classes, T] - per-timestep logits (if return_temporal)
            audio_features: [B, C, T_a] - audio embeddings
            video_features: [B, C, T_v] - video embeddings
        """
        # Extract features
        if audio_mfcc.dim() == 3:
            audio_mfcc = audio_mfcc.unsqueeze(1)
            
        audio_features = self.audio_encoder(audio_mfcc)
        video_features = self.video_encoder(video_frames)
        
        # Align temporal dimensions
        min_time = min(audio_features.size(2), video_features.size(2))
        audio_features = audio_features[:, :, :min_time]
        video_features = video_features[:, :, :min_time]
        
        # Compute correlation
        correlation = self.correlation(video_features, audio_features)
        
        # Per-timestep classification
        temporal_logits = self.classifier(correlation)
        
        # Global classification (aggregate over time)
        class_logits = self.global_classifier(temporal_logits)
        
        if return_temporal:
            return class_logits, temporal_logits, audio_features, video_features
        return class_logits, audio_features, video_features
    
    def predict_offset(self, class_logits):
        """
        Convert class logits to offset prediction.
        
        Args:
            class_logits: [B, num_classes] - classification logits
            
        Returns:
            offsets: [B] - predicted offset in frames
            confidences: [B] - prediction confidence (softmax probability)
        """
        probs = F.softmax(class_logits, dim=1)
        predicted_class = probs.argmax(dim=1)
        offsets = predicted_class - self.max_offset  # Convert class to offset
        confidences = probs.max(dim=1).values
        return offsets, confidences
    
    def offset_to_class(self, offset):
        """Convert offset value to class index."""
        return offset + self.max_offset
    
    def class_to_offset(self, class_idx):
        """Convert class index to offset value."""
        return class_idx - self.max_offset


class StreamSyncFCN_Classification(nn.Module):
    """
    Streaming-capable FCN SyncNet with classification output.
    
    Includes preprocessing, transfer learning, and inference utilities.
    """
    
    def __init__(self, embedding_dim=512, max_offset=125,
                 window_size=25, stride=5, buffer_size=100,
                 pretrained_syncnet_path=None, auto_load_pretrained=True,
                 dropout=0.3):
        super(StreamSyncFCN_Classification, self).__init__()
        
        self.window_size = window_size
        self.stride = stride
        self.buffer_size = buffer_size
        self.max_offset = max_offset
        self.num_classes = 2 * max_offset + 1
        
        # Initialize classification model
        self.fcn_model = SyncNetFCN_Classification(
            embedding_dim=embedding_dim,
            max_offset=max_offset,
            dropout=dropout
        )
        
        # Auto-load pretrained weights
        if auto_load_pretrained and pretrained_syncnet_path:
            self.load_pretrained_syncnet(pretrained_syncnet_path)
        
        self.reset_buffers()
    
    def reset_buffers(self):
        """Reset temporal buffers."""
        self.logits_buffer = []
        self.frame_count = 0
    
    def load_pretrained_syncnet(self, syncnet_model_path, freeze_conv=True, verbose=True):
        """Load conv layers from original SyncNet."""
        if verbose:
            print(f"Loading pretrained SyncNet from: {syncnet_model_path}")
        
        try:
            pretrained = torch.load(syncnet_model_path, map_location='cpu')
            if isinstance(pretrained, dict):
                pretrained_dict = pretrained.get('model_state_dict', pretrained.get('state_dict', pretrained))
            else:
                pretrained_dict = pretrained.state_dict()
            
            fcn_dict = self.fcn_model.state_dict()
            loaded_count = 0
            
            for key in list(pretrained_dict.keys()):
                if key.startswith('netcnnaud.'):
                    idx = key.split('.')[1]
                    param = '.'.join(key.split('.')[2:])
                    new_key = f'audio_encoder.conv_layers.{idx}.{param}'
                    if new_key in fcn_dict and pretrained_dict[key].shape == fcn_dict[new_key].shape:
                        fcn_dict[new_key] = pretrained_dict[key]
                        loaded_count += 1
                
                elif key.startswith('netcnnlip.'):
                    idx = key.split('.')[1]
                    param = '.'.join(key.split('.')[2:])
                    new_key = f'video_encoder.conv_layers.{idx}.{param}'
                    if new_key in fcn_dict and pretrained_dict[key].shape == fcn_dict[new_key].shape:
                        fcn_dict[new_key] = pretrained_dict[key]
                        loaded_count += 1
            
            self.fcn_model.load_state_dict(fcn_dict, strict=False)
            
            if verbose:
                print(f"✓ Loaded {loaded_count} pretrained conv parameters")
            
            if freeze_conv:
                for name, param in self.fcn_model.named_parameters():
                    if 'conv_layers' in name:
                        param.requires_grad = False
                if verbose:
                    print("✓ Froze pretrained conv layers")
        
        except Exception as e:
            if verbose:
                print(f"⚠ Could not load pretrained weights: {e}")
    
    def load_fcn_checkpoint(self, checkpoint_path, verbose=True):
        """Load FCN classification checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Try to load directly first
        try:
            self.fcn_model.load_state_dict(state_dict, strict=True)
            if verbose:
                print(f"✓ Loaded full checkpoint from {checkpoint_path}")
        except:
            # Load only matching keys
            model_dict = self.fcn_model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() 
                             if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.fcn_model.load_state_dict(model_dict, strict=False)
            if verbose:
                print(f"✓ Loaded {len(pretrained_dict)}/{len(state_dict)} parameters from {checkpoint_path}")
        
        return checkpoint.get('epoch', None)
    
    def unfreeze_all_layers(self, verbose=True):
        """Unfreeze all layers for fine-tuning."""
        for param in self.fcn_model.parameters():
            param.requires_grad = True
        if verbose:
            print("✓ Unfrozen all layers for fine-tuning")
    
    def forward(self, audio_mfcc, video_frames, return_temporal=False):
        """Forward pass through FCN model."""
        return self.fcn_model(audio_mfcc, video_frames, return_temporal)
    
    def extract_audio_mfcc(self, video_path, temp_dir='temp'):
        """Extract audio and compute MFCC."""
        os.makedirs(temp_dir, exist_ok=True)
        audio_path = os.path.join(temp_dir, 'temp_audio.wav')
        
        cmd = ['ffmpeg', '-y', '-i', video_path, '-ac', '1', '-ar', '16000',
               '-vn', '-acodec', 'pcm_s16le', audio_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        sample_rate, audio = wavfile.read(audio_path)
        mfcc = python_speech_features.mfcc(audio, sample_rate, numcep=13).T
        mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0).unsqueeze(0)
        
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        return mfcc_tensor
    
    def extract_video_frames(self, video_path, target_size=(112, 112)):
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
            raise ValueError(f"No frames extracted from {video_path}")
        
        frames_array = np.stack(frames, axis=0)
        video_tensor = torch.FloatTensor(frames_array).permute(3, 0, 1, 2).unsqueeze(0)
        
        return video_tensor
    
    def detect_offset(self, video_path, temp_dir='temp', verbose=True):
        """
        Detect AV offset using classification approach.
        
        Args:
            video_path: Path to video file
            temp_dir: Temporary directory for audio extraction
            verbose: Print progress information
            
        Returns:
            offset: Predicted offset in frames (positive = audio ahead)
            confidence: Classification confidence (0-1)
            class_probs: Full probability distribution over offset classes
        """
        if verbose:
            print(f"Processing: {video_path}")
        
        # Extract features
        mfcc = self.extract_audio_mfcc(video_path, temp_dir)
        video = self.extract_video_frames(video_path)
        
        if verbose:
            print(f"  Audio MFCC: {mfcc.shape}, Video: {video.shape}")
        
        # Run inference
        self.fcn_model.eval()
        with torch.no_grad():
            class_logits, _, _ = self.fcn_model(mfcc, video)
            offset, confidence = self.fcn_model.predict_offset(class_logits)
            class_probs = F.softmax(class_logits, dim=1)
        
        offset = offset.item()
        confidence = confidence.item()
        
        if verbose:
            print(f"  Detected offset: {offset:+d} frames")
            print(f"  Confidence: {confidence:.4f}")
        
        return offset, confidence, class_probs.squeeze(0).numpy()
    
    def process_video_file(self, video_path, temp_dir='temp', verbose=True):
        """Alias for detect_offset for compatibility."""
        offset, confidence, _ = self.detect_offset(video_path, temp_dir, verbose)
        return offset, confidence


def create_classification_criterion(max_offset=125, label_smoothing=0.1):
    """
    Create loss function for classification training.
    
    Args:
        max_offset: Maximum offset value
        label_smoothing: Label smoothing factor (0 = no smoothing)
        
    Returns:
        criterion: CrossEntropyLoss with optional label smoothing
    """
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def train_step_classification(model, audio, video, target_offset, criterion, optimizer, device):
    """
    Single training step for classification model.
    
    Args:
        model: SyncNetFCN_Classification or StreamSyncFCN_Classification
        audio: [B, 1, F, T] audio MFCC
        video: [B, 3, T, H, W] video frames
        target_offset: [B] target offset in frames (-max_offset to +max_offset)
        criterion: CrossEntropyLoss
        optimizer: Optimizer
        device: torch device
        
    Returns:
        loss: Training loss value
        accuracy: Classification accuracy
    """
    model.train()
    optimizer.zero_grad()
    
    audio = audio.to(device)
    video = video.to(device)
    
    # Convert offset to class index
    if hasattr(model, 'fcn_model'):
        target_class = target_offset + model.fcn_model.max_offset
    else:
        target_class = target_offset + model.max_offset
    target_class = target_class.long().to(device)
    
    # Forward pass
    if hasattr(model, 'fcn_model'):
        class_logits, _, _ = model(audio, video)
    else:
        class_logits, _, _ = model(audio, video)
    
    # Compute loss
    loss = criterion(class_logits, target_class)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Compute accuracy
    predicted_class = class_logits.argmax(dim=1)
    accuracy = (predicted_class == target_class).float().mean().item()
    
    return loss.item(), accuracy


def validate_classification(model, dataloader, criterion, device, max_offset=125):
    """
    Validate classification model.
    
    Returns:
        avg_loss: Average validation loss
        accuracy: Classification accuracy
        mean_error: Mean absolute error in frames
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
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
            correct += (predicted_class == target_class).sum().item()
            total += audio.size(0)
            
            # Mean absolute error
            predicted_offset = predicted_class - max_offset
            target_offset_dev = target_class - max_offset
            total_error += (predicted_offset - target_offset_dev).abs().sum().item()
    
    return total_loss / total, correct / total, total_error / total


if __name__ == "__main__":
    print("Testing SyncNetFCN_Classification...")
    
    # Test model creation (use smaller offset for quick testing)
    model = SyncNetFCN_Classification(embedding_dim=512, max_offset=125)
    print(f"Number of classes: {model.num_classes}")
    
    # Test forward pass
    audio_input = torch.randn(2, 1, 13, 100)
    video_input = torch.randn(2, 3, 25, 112, 112)
    
    class_logits, audio_feat, video_feat = model(audio_input, video_input)
    print(f"Class logits: {class_logits.shape}")
    print(f"Audio features: {audio_feat.shape}")
    print(f"Video features: {video_feat.shape}")
    
    # Test prediction
    offsets, confidences = model.predict_offset(class_logits)
    print(f"Predicted offsets: {offsets}")
    print(f"Confidences: {confidences}")
    
    # Test with temporal output
    class_logits, temporal_logits, _, _ = model(audio_input, video_input, return_temporal=True)
    print(f"Temporal logits: {temporal_logits.shape}")
    
    # Test training step
    print("\nTesting training step...")
    criterion = create_classification_criterion(max_offset=125, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    target_offset = torch.tensor([3, -5])  # Example target offsets
    
    loss, acc = train_step_classification(
        model, audio_input, video_input, target_offset,
        criterion, optimizer, 'cpu'
    )
    print(f"Training loss: {loss:.4f}, Accuracy: {acc:.2%}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\nTesting StreamSyncFCN_Classification...")
    stream_model = StreamSyncFCN_Classification(
        embedding_dim=512, max_offset=125,
        pretrained_syncnet_path=None, auto_load_pretrained=False
    )
    
    class_logits, _, _ = stream_model(audio_input, video_input)
    print(f"Stream model class logits: {class_logits.shape}")
    
    print("\n✓ All tests passed!")
