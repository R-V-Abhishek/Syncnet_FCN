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
            
            # Layer 6 - Spatial pooling to 1×1
            nn.Conv3d(256, 512, kernel_size=(3,6,6), stride=(1,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(512),
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
    Fully Convolutional SyncNet with temporal outputs.
    
    Architecture:
    1. Audio encoder: MFCC → temporal features
    2. Video encoder: frames → temporal features
    3. Correlation layer: compute audio-video similarity over time
    4. Sync predictor: predict sync probability for each frame
    """
    def __init__(self, embedding_dim=512, max_offset=15):
        super(SyncNetFCN, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.max_offset = max_offset
        
        # Encoders
        self.audio_encoder = FCN_AudioEncoder(output_channels=embedding_dim)
        self.video_encoder = FCN_VideoEncoder(output_channels=embedding_dim)
        
        # Temporal correlation
        self.correlation = TemporalCorrelation(max_displacement=max_offset)
        
        # Sync predictor (processes correlation map)
        self.sync_predictor = nn.Sequential(
            nn.Conv1d(2*max_offset+1, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 2*max_offset+1, kernel_size=1),  # Output: prob for each offset
        )
        
        # Optional: Temporal smoothing with dilated convolutions
        self.temporal_smoother = nn.Sequential(
            nn.Conv1d(2*max_offset+1, 64, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 2*max_offset+1, kernel_size=1),
        )
        
    def forward_audio(self, audio_mfcc):
        """Extract audio features."""
        return self.audio_encoder(audio_mfcc)
    
    def forward_video(self, video_frames):
        """Extract video features."""
        return self.video_encoder(video_frames)
    
    def forward(self, audio_mfcc, video_frames):
        """
        Forward pass with audio-video sync prediction.
        
        Args:
            audio_mfcc: [B, 1, F, T] - MFCC features
            video_frames: [B, 3, T', H, W] - video frames
            
        Returns:
            sync_probs: [B, 2*max_offset+1, T''] - sync probability for each offset and time
            audio_features: [B, C, T_a] - audio embeddings
            video_features: [B, C, T_v] - video embeddings
        """
        # Extract features
        audio_features = self.audio_encoder(audio_mfcc)  # [B, C, T_a]
        video_features = self.video_encoder(video_frames)  # [B, C, T_v]
        
        # Align temporal dimensions (if needed)
        min_time = min(audio_features.size(2), video_features.size(2))
        audio_features = audio_features[:, :, :min_time]
        video_features = video_features[:, :, :min_time]
        
        # Compute correlation
        correlation = self.correlation(video_features, audio_features)  # [B, 2*K+1, T]
        
        # Predict sync probabilities
        sync_logits = self.sync_predictor(correlation)  # [B, 2*K+1, T]
        sync_logits = self.temporal_smoother(sync_logits)  # Temporal smoothing
        
        # Apply softmax over offset dimension
        sync_probs = F.softmax(sync_logits, dim=1)  # [B, 2*K+1, T]
        
        return sync_probs, audio_features, video_features
    
    def compute_offset(self, sync_probs):
        """
        Compute offset from sync probability map.
        
        Args:
            sync_probs: [B, 2*K+1, T] - sync probabilities
            
        Returns:
            offsets: [B, T] - predicted offset for each frame
            confidences: [B, T] - confidence scores
        """
        # Find most likely offset for each time step
        max_probs, max_indices = torch.max(sync_probs, dim=1)  # [B, T]
        
        # Convert indices to offsets
        offsets = self.max_offset - max_indices  # [B, T]
        
        # Confidence = max_prob - median_prob
        median_probs = torch.median(sync_probs, dim=1)[0]  # [B, T]
        confidences = max_probs - median_probs  # [B, T]
        
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
        
        # Predict sync probabilities
        sync_logits = self.sync_predictor(correlation)
        sync_logits = self.temporal_smoother(sync_logits)
        sync_probs = F.softmax(sync_logits, dim=1)
        
        return sync_probs, audio_features, video_features


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
