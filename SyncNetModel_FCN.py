#!/usr/bin/python
#-*- coding: utf-8 -*-

"""
Fully Convolutional SyncNet (FCN-SyncNet)

Key improvements:
1. Fully convolutional architecture (no FC layers)
2. Temporal feature maps instead of single embeddings
3. Correlation-based audio-video fusion
4. Dense sync probability predictions over time
5. Multi-scale feature extraction
6. Attention mechanisms

Author: Enhanced version based on original SyncNet
Date: 2025-11-22
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
from collections import OrderedDict


class TemporalCorrelation(nn.Module):
    """
    Compute correlation between audio and video features across time.
    Inspired by FlowNet correlation layer.
    """
    def __init__(self, max_displacement=10):
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
            # Shift audio features
            shifted_feat2 = feat2_padded[:, :, offset+max_disp:offset+max_disp+T]
            
            # Compute correlation (cosine similarity)
            corr = (feat1 * shifted_feat2).sum(dim=1, keepdim=True)  # [B, 1, T]
            corr_list.append(corr)
        
        # Stack all correlations
        correlation = torch.cat(corr_list, dim=1)  # [B, 2*max_disp+1, T]
        
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
        """
        Args:
            x: [B, C, T]
        """
        B, C, T = x.size()
        
        # Generate query, key, value
        query = self.query_conv(x).permute(0, 2, 1)  # [B, T, C']
        key = self.key_conv(x)  # [B, C', T]
        value = self.value_conv(x)  # [B, C, T]
        
        # Attention weights
        attention = torch.bmm(query, key)  # [B, T, T]
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention
        out = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C, T]
        out = self.gamma * out + x
        
        return out


class FCN_AudioEncoder(nn.Module):
    """
    Fully convolutional audio encoder.
    Input: MFCC or Mel spectrogram [B, 1, F, T]
    Output: Feature map [B, C, T']
    """
    def __init__(self, output_channels=512):
        super(FCN_AudioEncoder, self).__init__()
        
        # Convolutional layers (preserve temporal dimension)
        self.conv_layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Layer 2
            nn.Conv2d(64, 192, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(1,2)),  # Reduce frequency, keep time
            
            # Layer 3
            nn.Conv2d(192, 384, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            # Layer 4
            nn.Conv2d(384, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Layer 5
            nn.Conv2d(256, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
            
            # Layer 6 - Reduce frequency dimension to 1
            nn.Conv2d(256, 512, kernel_size=(5,1), stride=(5,1), padding=(0,0)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # 1×1 conv to adjust channels (replaces FC layer)
        self.channel_conv = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, output_channels, kernel_size=1),
            nn.BatchNorm1d(output_channels),
        )
        
        # Channel attention
        self.channel_attn = ChannelAttention(output_channels)
        
    def forward(self, x):
        """
        Args:
            x: [B, 1, F, T] - MFCC features
        Returns:
            features: [B, C, T'] - temporal feature map
        """
        # Convolutional encoding
        x = self.conv_layers(x)  # [B, 512, F', T']
        
        # Collapse frequency dimension
        B, C, F, T = x.size()
        x = x.view(B, C * F, T)  # Flatten frequency into channels
        
        # Reduce to output_channels
        x = self.channel_conv(x)  # [B, output_channels, T']
        
        # Apply attention
        x = self.channel_attn(x)
        
        return x


class FCN_VideoEncoder(nn.Module):
    """
    Fully convolutional video encoder.
    Input: Video clip [B, 3, T, H, W]
    Output: Feature map [B, C, T']
    """
    def __init__(self, output_channels=512):
        super(FCN_VideoEncoder, self).__init__()
        
        # 3D Convolutional layers
        self.conv_layers = nn.Sequential(
            # Layer 1
            nn.Conv3d(3, 96, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3)),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            
            # Layer 2
            nn.Conv3d(96, 256, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            
            # Layer 3
            nn.Conv3d(256, 256, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            # Layer 4
            nn.Conv3d(256, 256, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            # Layer 5
            nn.Conv3d(256, 256, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            
            # Layer 6 - Reduce spatial dimension
            nn.Conv3d(256, 512, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            # Adaptive pooling to 1x1 spatial
            nn.AdaptiveAvgPool3d((None, 1, 1))  # Keep temporal, pool spatial to 1x1
        )
        
        # 1×1 conv to adjust channels (replaces FC layer)
        self.channel_conv = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, output_channels, kernel_size=1),
            nn.BatchNorm1d(output_channels),
        )
        
        # Channel attention
        self.channel_attn = ChannelAttention(output_channels)
        
    def forward(self, x):
        """
        Args:
            x: [B, 3, T, H, W] - video frames
        Returns:
            features: [B, C, T'] - temporal feature map
        """
        # Convolutional encoding
        x = self.conv_layers(x)  # [B, 512, T', 1, 1]
        
        # Remove spatial dimensions
        B, C, T, H, W = x.size()
        x = x.view(B, C, T)  # [B, 512, T']
        
        # Reduce to output_channels
        x = self.channel_conv(x)  # [B, output_channels, T']
        
        # Apply attention
        x = self.channel_attn(x)
        
        return x


class SyncNetFCN(nn.Module):
    """
    Fully Convolutional SyncNet with temporal outputs (REGRESSION VERSION).
    
    Architecture:
    1. Audio encoder: MFCC → temporal features
    2. Video encoder: frames → temporal features
    3. Correlation layer: compute audio-video similarity over time
    4. Offset regressor: predict continuous offset value for each frame
    
    Changes from classification version:
    - Output: [B, 1, T] continuous offset values (not probability distribution)
    - Default max_offset: 125 frames (±5 seconds at 25fps) for streaming
    - Loss: L1/MSE instead of CrossEntropy
    """
    def __init__(self, embedding_dim=512, max_offset=125):
        super(SyncNetFCN, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.max_offset = max_offset
        
        # Encoders
        self.audio_encoder = FCN_AudioEncoder(output_channels=embedding_dim)
        self.video_encoder = FCN_VideoEncoder(output_channels=embedding_dim)
        
        # Temporal correlation
        self.correlation = TemporalCorrelation(max_displacement=max_offset)
        
        # Offset regressor (processes correlation map) - REGRESSION OUTPUT
        self.offset_regressor = nn.Sequential(
            nn.Conv1d(2*max_offset+1, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 1, kernel_size=1),  # Output: single continuous offset value
        )
        
        # Optional: Temporal smoothing with dilated convolutions
        self.temporal_smoother = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=1),
        )
        
    def forward_audio(self, audio_mfcc):
        """Extract audio features."""
        return self.audio_encoder(audio_mfcc)
    
    def forward_video(self, video_frames):
        """Extract video features."""
        return self.video_encoder(video_frames)
    
    def forward(self, audio_mfcc, video_frames):
        """
        Forward pass with audio-video offset regression.
        
        Args:
            audio_mfcc: [B, 1, F, T] - MFCC features
            video_frames: [B, 3, T', H, W] - video frames
            
        Returns:
            predicted_offsets: [B, 1, T''] - predicted offset in frames for each timestep
            audio_features: [B, C, T_a] - audio embeddings
            video_features: [B, C, T_v] - video embeddings
        """
        # Extract features
        if audio_mfcc.dim() == 3:
            audio_mfcc = audio_mfcc.unsqueeze(1)  # [B, 1, F, T]
            
        audio_features = self.audio_encoder(audio_mfcc)  # [B, C, T_a]
        video_features = self.video_encoder(video_frames)  # [B, C, T_v]
        
        # Align temporal dimensions (if needed)
        min_time = min(audio_features.size(2), video_features.size(2))
        audio_features = audio_features[:, :, :min_time]
        video_features = video_features[:, :, :min_time]
        
        # Compute correlation
        correlation = self.correlation(video_features, audio_features)  # [B, 2*K+1, T]
        
        # Predict offset (regression)
        offset_logits = self.offset_regressor(correlation)  # [B, 1, T]
        predicted_offsets = self.temporal_smoother(offset_logits)  # Temporal smoothing
        
        # Clamp to valid range
        predicted_offsets = torch.clamp(predicted_offsets, -self.max_offset, self.max_offset)
        
        return predicted_offsets, audio_features, video_features
    
    def compute_offset(self, predicted_offsets):
        """
        Extract offset and confidence from regression predictions.
        
        Args:
            predicted_offsets: [B, 1, T] - predicted offsets
            
        Returns:
            offsets: [B, T] - predicted offset for each frame
            confidences: [B, T] - confidence scores (inverse of variance)
        """
        # Remove channel dimension
        offsets = predicted_offsets.squeeze(1)  # [B, T]
        
        # Confidence = inverse of temporal variance (stable predictions = high confidence)
        temporal_variance = torch.var(offsets, dim=1, keepdim=True) + 1e-6  # [B, 1]
        confidences = 1.0 / temporal_variance  # [B, 1]
        confidences = confidences.expand_as(offsets)  # [B, T]
        
        # Normalize confidence to [0, 1]
        confidences = torch.sigmoid(confidences - 5.0)  # Shift to reasonable range
        
        return offsets, confidences


class SyncNetFCN_WithAttention(SyncNetFCN):
    """
    Enhanced version with cross-modal attention.
    Audio and video features attend to each other before correlation.
    """
    def __init__(self, embedding_dim=512, max_offset=15):
        super(SyncNetFCN_WithAttention, self).__init__(embedding_dim, max_offset)
        
        # Cross-modal attention
        self.audio_to_video_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            batch_first=False
        )
        
        self.video_to_audio_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            batch_first=False
        )
        
        # Self-attention for temporal modeling
        self.audio_self_attn = TemporalAttention(embedding_dim)
        self.video_self_attn = TemporalAttention(embedding_dim)
        
    def forward(self, audio_mfcc, video_frames):
        """
        Forward pass with attention mechanisms.
        """
        # Extract features
        if audio_mfcc.dim() == 3:
            audio_mfcc = audio_mfcc.unsqueeze(1)  # [B, 1, F, T]
            
        audio_features = self.audio_encoder(audio_mfcc)  # [B, C, T_a]
        video_features = self.video_encoder(video_frames)  # [B, C, T_v]
        
        # Self-attention
        audio_features = self.audio_self_attn(audio_features)
        video_features = self.video_self_attn(video_features)
        
        # Align temporal dimensions
        min_time = min(audio_features.size(2), video_features.size(2))
        audio_features = audio_features[:, :, :min_time]
        video_features = video_features[:, :, :min_time]
        
        # Cross-modal attention
        # Reshape for attention: [T, B, C]
        audio_t = audio_features.permute(2, 0, 1)
        video_t = video_features.permute(2, 0, 1)
        
        # Audio attends to video
        audio_attended, _ = self.audio_to_video_attn(
            query=audio_t, key=video_t, value=video_t
        )
        audio_features = audio_features + audio_attended.permute(1, 2, 0)
        
        # Video attends to audio
        video_attended, _ = self.video_to_audio_attn(
            query=video_t, key=audio_t, value=audio_t
        )
        video_features = video_features + video_attended.permute(1, 2, 0)
        
        # Compute correlation
        correlation = self.correlation(video_features, audio_features)
        
        # Predict offset (regression)
        offset_logits = self.offset_regressor(correlation)
        predicted_offsets = self.temporal_smoother(offset_logits)
        
        # Clamp to valid range
        predicted_offsets = torch.clamp(predicted_offsets, -self.max_offset, self.max_offset)
        
        return predicted_offsets, audio_features, video_features


class StreamSyncFCN(nn.Module):
    """
    StreamSync-style FCN with built-in preprocessing and transfer learning.
    
    Features:
    1. Sliding window processing for streams
    2. HLS stream support (.m3u8)
    3. Raw video file processing (MP4, AVI, etc.)
    4. Automatic transfer learning from Sync NetModel.py
    5. Temporal buffering and smoothing
    """
    
    def __init__(self, embedding_dim=512, max_offset=15,
                 window_size=25, stride=5, buffer_size=100,
                 use_attention=False, pretrained_syncnet_path=None,
                 auto_load_pretrained=True):
        """
        Args:
            embedding_dim: Feature dimension
            max_offset: Maximum temporal offset (frames)
            window_size: Frames per processing window
            stride: Window stride
            buffer_size: Temporal buffer size
            use_attention: Use attention model
            pretrained_syncnet_path: Path to original SyncNet weights
            auto_load_pretrained: Auto-load pretrained weights if path provided
        """
        super(StreamSyncFCN, self).__init__()
        
        self.window_size = window_size
        self.stride = stride
        self.buffer_size = buffer_size
        self.max_offset = max_offset
        
        # Initialize FCN model
        if use_attention:
            self.fcn_model = SyncNetFCN_WithAttention(embedding_dim, max_offset)
        else:
            self.fcn_model = SyncNetFCN(embedding_dim, max_offset)
        
        # Auto-load pretrained weights
        if auto_load_pretrained and pretrained_syncnet_path:
            self.load_pretrained_syncnet(pretrained_syncnet_path)
        
        self.reset_buffers()
    
    def reset_buffers(self):
        """Reset temporal buffers."""
        self.offset_buffer = []
        self.confidence_buffer = []
        self.frame_count = 0
    
    def load_pretrained_syncnet(self, syncnet_model_path, freeze_conv=True, verbose=True):
        """
        Load conv layers from original SyncNet (SyncNetModel.py).
        Maps: netcnnaud.* → audio_encoder.conv_layers.*
              netcnnlip.* → video_encoder.conv_layers.*
        """
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
            
            # Map audio conv layers
            for key in list(pretrained_dict.keys()):
                if key.startswith('netcnnaud.'):
                    idx = key.split('.')[1]
                    param = '.'.join(key.split('.')[2:])
                    new_key = f'audio_encoder.conv_layers.{idx}.{param}'
                    if new_key in fcn_dict and pretrained_dict[key].shape == fcn_dict[new_key].shape:
                        fcn_dict[new_key] = pretrained_dict[key]
                        loaded_count += 1
                
                # Map video conv layers
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
    
    def unfreeze_all_layers(self, verbose=True):
        """Unfreeze all layers for fine-tuning."""
        for param in self.fcn_model.parameters():
            param.requires_grad = True
        if verbose:
            print("✓ Unfrozen all layers for fine-tuning")
    
    def forward(self, audio_mfcc, video_frames):
        """Forward pass through FCN model."""
        return self.fcn_model(audio_mfcc, video_frames)
    
    def process_window(self, audio_window, video_window):
        """Process single window."""
        with torch.no_grad():
            sync_probs, _, _ = self.fcn_model(audio_window, video_window)
            offsets, confidences = self.fcn_model.compute_offset(sync_probs)
        return offsets[0].mean().item(), confidences[0].mean().item()
    
    def process_stream(self, audio_stream, video_stream, return_trace=False):
        """Process full stream with sliding windows."""
        self.reset_buffers()
        
        video_frames = video_stream.shape[2]
        audio_frames = audio_stream.shape[3] // 4
        min_frames = min(video_frames, audio_frames)
        num_windows = max(1, (min_frames - self.window_size) // self.stride + 1)
        
        trace = {'offsets': [], 'confidences': [], 'timestamps': []}
        
        for win_idx in range(num_windows):
            start = win_idx * self.stride
            end = min(start + self.window_size, min_frames)
            
            video_win = video_stream[:, :, start:end, :, :]
            audio_win = audio_stream[:, :, :, start*4:end*4]
            
            offset, confidence = self.process_window(audio_win, video_win)
            
            self.offset_buffer.append(offset)
            self.confidence_buffer.append(confidence)
            
            if return_trace:
                trace['offsets'].append(offset)
                trace['confidences'].append(confidence)
                trace['timestamps'].append(start)
            
            if len(self.offset_buffer) > self.buffer_size:
                self.offset_buffer.pop(0)
                self.confidence_buffer.pop(0)
            
            self.frame_count = end
        
        final_offset, final_conf = self.get_smoothed_prediction()
        
        return (final_offset, final_conf, trace) if return_trace else (final_offset, final_conf)
    
    def get_smoothed_prediction(self, method='confidence_weighted'):
        """Compute smoothed offset from buffer."""
        if not self.offset_buffer:
            return 0.0, 0.0
        
        offsets = torch.tensor(self.offset_buffer)
        confs = torch.tensor(self.confidence_buffer)
        
        if method == 'confidence_weighted':
            weights = confs / (confs.sum() + 1e-8)
            offset = (offsets * weights).sum().item()
        elif method == 'median':
            offset = torch.median(offsets).item()
        else:
            offset = torch.mean(offsets).item()
        
        return offset, torch.mean(confs).item()
    
    def extract_audio_mfcc(self, video_path, temp_dir='temp'):
        """Extract audio and compute MFCC."""
        os.makedirs(temp_dir, exist_ok=True)
        audio_path = os.path.join(temp_dir, 'temp_audio.wav')
        
        cmd = ['ffmpeg', '-y', '-i', video_path, '-ac', '1', '-ar', '16000',
               '-vn', '-acodec', 'pcm_s16le', audio_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        sample_rate, audio = wavfile.read(audio_path)
        mfcc = python_speech_features.mfcc(audio, sample_rate).T
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
    
    def process_video_file(self, video_path, return_trace=False, temp_dir='temp',
                          target_size=(112, 112), verbose=True):
        """
        Process raw video file (MP4, AVI, MOV, etc.).
        
        Args:
            video_path: Path to video file
            return_trace: Return per-window predictions
            temp_dir: Temporary directory
            target_size: Video frame size
            verbose: Print progress
            
        Returns:
            offset: Detected offset (frames)
            confidence: Detection confidence
            trace: (optional) Per-window data
            
        Example:
            >>> model = StreamSyncFCN(pretrained_syncnet_path='data/syncnet_v2.model')
            >>> offset, conf = model.process_video_file('video.mp4')
        """
        if verbose:
            print(f"Processing: {video_path}")
        
        mfcc = self.extract_audio_mfcc(video_path, temp_dir)
        video = self.extract_video_frames(video_path, target_size)
        
        if verbose:
            print(f"  Audio: {mfcc.shape}, Video: {video.shape}")
        
        result = self.process_stream(mfcc, video, return_trace)
        
        if verbose:
            offset, conf = result[:2]
            print(f"  Offset: {offset:.2f} frames, Confidence: {conf:.3f}")
        
        return result
    
    def detect_offset_correlation(self, video_path, calibration_offset=3, calibration_scale=-0.5, 
                                    calibration_baseline=-15, temp_dir='temp', verbose=True):
        """
        Detect AV offset using correlation-based method with calibration.
        
        This method uses the trained audio-video encoders to compute temporal
        correlation and find the best matching offset. A linear calibration
        is applied to correct for systematic bias in the model.
        
        Calibration formula: calibrated = calibration_offset + calibration_scale * (raw - calibration_baseline)
        Default values determined empirically from test videos.
        
        Args:
            video_path: Path to video file
            calibration_offset: Baseline expected offset (default: 3)
            calibration_scale: Scale factor for raw offset (default: -0.5)
            calibration_baseline: Baseline raw offset (default: -15)
            temp_dir: Temporary directory for audio extraction
            verbose: Print progress information
            
        Returns:
            offset: Calibrated offset in frames (positive = audio ahead)
            confidence: Detection confidence (correlation strength)
            raw_offset: Uncalibrated raw offset from correlation
            
        Example:
            >>> model = StreamSyncFCN(pretrained_syncnet_path='data/syncnet_v2.model')
            >>> offset, conf, raw = model.detect_offset_correlation('video.mp4')
            >>> print(f"Detected offset: {offset} frames")
        """
        import python_speech_features
        from scipy.io import wavfile
        
        if verbose:
            print(f"Processing: {video_path}")
        
        # Extract audio MFCC
        os.makedirs(temp_dir, exist_ok=True)
        audio_path = os.path.join(temp_dir, 'temp_audio.wav')
        
        cmd = ['ffmpeg', '-y', '-i', video_path, '-ac', '1', '-ar', '16000',
               '-vn', '-acodec', 'pcm_s16le', audio_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        sample_rate, audio = wavfile.read(audio_path)
        mfcc = python_speech_features.mfcc(audio, sample_rate, numcep=13)
        audio_tensor = torch.FloatTensor(mfcc.T).unsqueeze(0).unsqueeze(0)
        
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        # Extract video frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (112, 112))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype(np.float32) / 255.0)
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames extracted from {video_path}")
        
        video_tensor = torch.FloatTensor(np.stack(frames)).permute(3, 0, 1, 2).unsqueeze(0)
        
        if verbose:
            print(f"  Audio MFCC: {audio_tensor.shape}, Video: {video_tensor.shape}")
        
        # Compute correlation-based offset
        with torch.no_grad():
            # Get features from encoders
            audio_feat = self.fcn_model.audio_encoder(audio_tensor)
            video_feat = self.fcn_model.video_encoder(video_tensor)
            
            # Align temporal dimensions
            min_t = min(audio_feat.shape[2], video_feat.shape[2])
            audio_feat = audio_feat[:, :, :min_t]
            video_feat = video_feat[:, :, :min_t]
            
            # Compute correlation map
            correlation = self.fcn_model.correlation(video_feat, audio_feat)
            
            # Average over time dimension
            corr_avg = correlation.mean(dim=2).squeeze(0)
            
            # Find best offset (argmax of correlation)
            best_idx = corr_avg.argmax().item()
            raw_offset = best_idx - self.max_offset
            
            # Compute confidence as peak prominence
            corr_np = corr_avg.numpy()
            peak_val = corr_np[best_idx]
            median_val = np.median(corr_np)
            confidence = peak_val - median_val
            
            # Apply linear calibration: calibrated = offset + scale * (raw - baseline)
            calibrated_offset = int(round(calibration_offset + calibration_scale * (raw_offset - calibration_baseline)))
        
        if verbose:
            print(f"  Raw offset: {raw_offset}, Calibrated: {calibrated_offset}")
            print(f"  Confidence: {confidence:.4f}")
        
        return calibrated_offset, confidence, raw_offset

    def process_hls_stream(self, hls_url, segment_duration=10, return_trace=False,
                          temp_dir='temp_hls', verbose=True):
        """
        Process HLS stream (.m3u8 playlist).
        
        Args:
            hls_url: URL to .m3u8 playlist
            segment_duration: Seconds to capture
            return_trace: Return per-window predictions
            temp_dir: Temporary directory
            verbose: Print progress
            
        Returns:
            offset: Detected offset
            confidence: Detection confidence
            trace: (optional) Per-window data
            
        Example:
            >>> model = StreamSyncFCN(pretrained_syncnet_path='data/syncnet_v2.model')
           >>> offset, conf = model.process_hls_stream('http://example.com/stream.m3u8')
        """
        if verbose:
            print(f"Processing HLS: {hls_url}")
        
        os.makedirs(temp_dir, exist_ok=True)
        temp_video = os.path.join(temp_dir, 'hls_segment.mp4')
        
        try:
            cmd = ['ffmpeg', '-y', '-i', hls_url, '-t', str(segment_duration),
                   '-c', 'copy', temp_video]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                          check=True, timeout=segment_duration + 30)
            
            result = self.process_video_file(temp_video, return_trace, temp_dir, verbose=verbose)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"HLS processing failed: {e}")
        finally:
            if os.path.exists(temp_video):
                os.remove(temp_video)


# Utility functions
def save_model(model, filename):
    """Save model to file."""
    with open(filename, "wb") as f:
        torch.save(model.state_dict(), f)
        print(f"{filename} saved.")


def load_model(model, filename):
    """Load model from file."""
    state_dict = torch.load(filename, map_location='cpu')
    model.load_state_dict(state_dict)
    print(f"{filename} loaded.")
    return model


if __name__ == "__main__":
    # Test the models
    print("Testing FCN_AudioEncoder...")
    audio_encoder = FCN_AudioEncoder(output_channels=512)
    audio_input = torch.randn(2, 1, 13, 100)  # [B, 1, MFCC_dim, Time]
    audio_out = audio_encoder(audio_input)
    print(f"Audio input: {audio_input.shape} → Audio output: {audio_out.shape}")
    
    print("\nTesting FCN_VideoEncoder...")
    video_encoder = FCN_VideoEncoder(output_channels=512)
    video_input = torch.randn(2, 3, 25, 112, 112)  # [B, 3, T, H, W]
    video_out = video_encoder(video_input)
    print(f"Video input: {video_input.shape} → Video output: {video_out.shape}")
    
    print("\nTesting SyncNetFCN...")
    model = SyncNetFCN(embedding_dim=512, max_offset=15)
    sync_probs, audio_feat, video_feat = model(audio_input, video_input)
    print(f"Sync probs: {sync_probs.shape}")
    print(f"Audio features: {audio_feat.shape}")
    print(f"Video features: {video_feat.shape}")
    
    offsets, confidences = model.compute_offset(sync_probs)
    print(f"Offsets: {offsets.shape}")
    print(f"Confidences: {confidences.shape}")
    
    print("\nTesting SyncNetFCN_WithAttention...")
    model_attn = SyncNetFCN_WithAttention(embedding_dim=512, max_offset=15)
    sync_probs, audio_feat, video_feat = model_attn(audio_input, video_input)
    print(f"Sync probs (with attention): {sync_probs.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    total_params_attn = sum(p.numel() for p in model_attn.parameters())
    print(f"\nTotal parameters (FCN): {total_params:,}")
    print(f"Total parameters (FCN+Attention): {total_params_attn:,}")
