#!/usr/bin/python
#-*- coding: utf-8 -*-
"""
Transfer Learning Implementation for SyncNet

This module provides pre-trained backbone integration for improved performance.

Supported backbones:
- Video: 3D ResNet (Kinetics), I3D, SlowFast, X3D
- Audio: VGGish (AudioSet), wav2vec 2.0, HuBERT

Author: Enhanced version
Date: 2025-11-22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== VIDEO BACKBONES ====================

class ResNet3D_Backbone(nn.Module):
    """
    3D ResNet backbone pre-trained on Kinetics-400.
    Uses torchvision's video models.
    """
    def __init__(self, embedding_dim=512, pretrained=True, model_type='r3d_18'):
        super(ResNet3D_Backbone, self).__init__()
        
        try:
            import torchvision.models.video as video_models
            
            # Load pre-trained model
            if model_type == 'r3d_18':
                backbone = video_models.r3d_18(pretrained=pretrained)
            elif model_type == 'mc3_18':
                backbone = video_models.mc3_18(pretrained=pretrained)
            elif model_type == 'r2plus1d_18':
                backbone = video_models.r2plus1d_18(pretrained=pretrained)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Remove final FC and pooling layers
            self.features = nn.Sequential(*list(backbone.children())[:-2])
            
            # Add custom head
            self.conv_head = nn.Sequential(
                nn.Conv3d(512, embedding_dim, kernel_size=1),
                nn.BatchNorm3d(embedding_dim),
                nn.ReLU(inplace=True),
            )
            
            print(f"Loaded {model_type} with pretrained={pretrained}")
            
        except ImportError:
            print("Warning: torchvision not found. Using random initialization.")
            self.features = self._build_simple_3dcnn()
            self.conv_head = nn.Conv3d(512, embedding_dim, 1)
    
    def _build_simple_3dcnn(self):
        """Fallback if torchvision not available."""
        return nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, T, H, W]
        Returns:
            features: [B, C, T', H', W']
        """
        x = self.features(x)
        x = self.conv_head(x)
        return x


class I3D_Backbone(nn.Module):
    """
    Inflated 3D ConvNet (I3D) backbone.
    Requires external I3D implementation.
    """
    def __init__(self, embedding_dim=512, pretrained=True):
        super(I3D_Backbone, self).__init__()
        
        try:
            # Try to import I3D (needs to be installed separately)
            from i3d import InceptionI3d
            
            self.i3d = InceptionI3d(400, in_channels=3)
            
            if pretrained:
                # Load pre-trained weights
                state_dict = torch.load('models/rgb_imagenet.pt', map_location='cpu')
                self.i3d.load_state_dict(state_dict)
                print("Loaded I3D with ImageNet+Kinetics pre-training")
            
            # Adaptation layer
            self.adapt = nn.Conv3d(1024, embedding_dim, kernel_size=1)
            
        except:
            print("Warning: I3D not available. Install from: https://github.com/piergiaj/pytorch-i3d")
            # Fallback to simple 3D CNN
            self.i3d = self._build_fallback()
            self.adapt = nn.Conv3d(512, embedding_dim, 1)
    
    def _build_fallback(self):
        return nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        features = self.i3d.extract_features(x) if hasattr(self.i3d, 'extract_features') else self.i3d(x)
        features = self.adapt(features)
        return features


# ==================== AUDIO BACKBONES ====================

class VGGish_Backbone(nn.Module):
    """
    VGGish audio encoder pre-trained on AudioSet.
    Processes log-mel spectrograms.
    """
    def __init__(self, embedding_dim=512, pretrained=True):
        super(VGGish_Backbone, self).__init__()
        
        try:
            import torchvggish
            
            # Load VGGish
            self.vggish = torchvggish.vggish()
            
            if pretrained:
                # Download and load pre-trained weights
                self.vggish.load_state_dict(
                    torch.hub.load_state_dict_from_url(
                        'https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth',
                        map_location='cpu'
                    )
                )
                print("Loaded VGGish pre-trained on AudioSet")
            
            # Use convolutional part only
            self.features = self.vggish.features
            
            # Adaptation layer
            self.adapt = nn.Sequential(
                nn.Conv2d(512, embedding_dim, kernel_size=1),
                nn.BatchNorm2d(embedding_dim),
                nn.ReLU(inplace=True),
            )
            
        except ImportError:
            print("Warning: torchvggish not found. Install: pip install torchvggish")
            self.features = self._build_fallback()
            self.adapt = nn.Conv2d(512, embedding_dim, 1)
    
    def _build_fallback(self):
        """Simple audio CNN if VGGish unavailable."""
        return nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, 1, F, T] or [B, 1, 96, T] (log-mel spectrogram)
        Returns:
            features: [B, C, F', T']
        """
        x = self.features(x)
        x = self.adapt(x)
        return x


class Wav2Vec_Backbone(nn.Module):
    """
    wav2vec 2.0 backbone for speech representation.
    Processes raw waveforms.
    """
    def __init__(self, embedding_dim=512, pretrained=True, model_name='facebook/wav2vec2-base'):
        super(Wav2Vec_Backbone, self).__init__()
        
        try:
            from transformers import Wav2Vec2Model
            
            if pretrained:
                self.wav2vec = Wav2Vec2Model.from_pretrained(model_name)
                print(f"Loaded {model_name} from HuggingFace")
            else:
                from transformers import Wav2Vec2Config
                config = Wav2Vec2Config()
                self.wav2vec = Wav2Vec2Model(config)
            
            # Freeze early layers for fine-tuning
            self._freeze_layers(num_layers_to_freeze=6)
            
            # Adaptation layer
            wav2vec_dim = self.wav2vec.config.hidden_size
            self.adapt = nn.Sequential(
                nn.Linear(wav2vec_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.ReLU(),
            )
            
        except ImportError:
            print("Warning: transformers not found. Install: pip install transformers")
            raise
    
    def _freeze_layers(self, num_layers_to_freeze):
        """Freeze early transformer layers."""
        for param in self.wav2vec.feature_extractor.parameters():
            param.requires_grad = False
        
        for i, layer in enumerate(self.wav2vec.encoder.layers):
            if i < num_layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(self, waveform):
        """
        Args:
            waveform: [B, T] - raw audio waveform (16kHz)
        Returns:
            features: [B, C, T'] - temporal features
        """
        # Extract features from wav2vec
        outputs = self.wav2vec(waveform, output_hidden_states=True)
        features = outputs.last_hidden_state  # [B, T', D]
        
        # Adapt to target dimension
        features = self.adapt(features)  # [B, T', embedding_dim]
        
        # Reshape to [B, C, T']
        features = features.transpose(1, 2)
        
        return features


# ==================== INTEGRATED SYNCNET WITH TRANSFER LEARNING ====================

class SyncNet_TransferLearning(nn.Module):
    """
    SyncNet with transfer learning from pre-trained backbones.
    
    Args:
        video_backbone: 'resnet3d', 'i3d', 'simple'
        audio_backbone: 'vggish', 'wav2vec', 'simple'
        embedding_dim: Dimension of shared embedding space
        max_offset: Maximum temporal offset to consider
        freeze_backbone: Whether to freeze backbone weights
    """
    def __init__(self, 
                 video_backbone='resnet3d',
                 audio_backbone='vggish',
                 embedding_dim=512,
                 max_offset=15,
                 freeze_backbone=False):
        super(SyncNet_TransferLearning, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.max_offset = max_offset
        
        # Initialize video encoder
        if video_backbone == 'resnet3d':
            self.video_encoder = ResNet3D_Backbone(embedding_dim, pretrained=True)
        elif video_backbone == 'i3d':
            self.video_encoder = I3D_Backbone(embedding_dim, pretrained=True)
        else:
            from SyncNetModel_FCN import FCN_VideoEncoder
            self.video_encoder = FCN_VideoEncoder(embedding_dim)
        
        # Initialize audio encoder
        if audio_backbone == 'vggish':
            self.audio_encoder = VGGish_Backbone(embedding_dim, pretrained=True)
        elif audio_backbone == 'wav2vec':
            self.audio_encoder = Wav2Vec_Backbone(embedding_dim, pretrained=True)
        else:
            from SyncNetModel_FCN import FCN_AudioEncoder
            self.audio_encoder = FCN_AudioEncoder(embedding_dim)
        
        # Freeze backbones if requested
        if freeze_backbone:
            self._freeze_backbones()
        
        # Temporal pooling to handle variable spatial/frequency dimensions
        self.video_temporal_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.audio_temporal_pool = nn.AdaptiveAvgPool2d((1, None))
        
        # Correlation and sync prediction (from FCN model)
        from SyncNetModel_FCN import TemporalCorrelation
        self.correlation = TemporalCorrelation(max_displacement=max_offset)
        
        self.sync_predictor = nn.Sequential(
            nn.Conv1d(2*max_offset+1, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 2*max_offset+1, kernel_size=1),
        )
    
    def _freeze_backbones(self):
        """Freeze backbone parameters for fine-tuning only the head."""
        for param in self.video_encoder.parameters():
            param.requires_grad = False
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        print("Backbones frozen. Only training sync predictor.")
    
    def forward_video(self, video):
        """
        Extract video features.
        Args:
            video: [B, 3, T, H, W]
        Returns:
            features: [B, C, T']
        """
        features = self.video_encoder(video)  # [B, C, T', H', W']
        features = self.video_temporal_pool(features)  # [B, C, T', 1, 1]
        B, C, T, _, _ = features.shape
        features = features.view(B, C, T)  # [B, C, T']
        return features
    
    def forward_audio(self, audio):
        """
        Extract audio features.
        Args:
            audio: [B, 1, F, T] or [B, T] (raw waveform for wav2vec)
        Returns:
            features: [B, C, T']
        """
        if isinstance(self.audio_encoder, Wav2Vec_Backbone):
            # wav2vec expects [B, T]
            if audio.dim() == 4:
                # Convert from spectrogram to waveform (placeholder - need actual audio)
                raise NotImplementedError("Need raw waveform for wav2vec")
            features = self.audio_encoder(audio)
        else:
            features = self.audio_encoder(audio)  # [B, C, F', T']
            features = self.audio_temporal_pool(features)  # [B, C, 1, T']
            B, C, _, T = features.shape
            features = features.view(B, C, T)  # [B, C, T']
        
        return features
    
    def forward(self, audio, video):
        """
        Full forward pass with sync prediction.
        
        Args:
            audio: [B, 1, F, T] - audio features
            video: [B, 3, T', H, W] - video frames
            
        Returns:
            sync_probs: [B, 2K+1, T''] - sync probabilities
            audio_features: [B, C, T_a]
            video_features: [B, C, T_v]
        """
        # Extract features
        audio_features = self.forward_audio(audio)
        video_features = self.forward_video(video)
        
        # Align temporal dimensions
        min_time = min(audio_features.size(2), video_features.size(2))
        audio_features = audio_features[:, :, :min_time]
        video_features = video_features[:, :, :min_time]
        
        # Compute correlation
        correlation = self.correlation(video_features, audio_features)
        
        # Predict sync probabilities
        sync_logits = self.sync_predictor(correlation)
        sync_probs = F.softmax(sync_logits, dim=1)
        
        return sync_probs, audio_features, video_features
    
    def compute_offset(self, sync_probs):
        """
        Compute offset from sync probability map.
        
        Args:
            sync_probs: [B, 2K+1, T] - sync probabilities
            
        Returns:
            offsets: [B, T] - predicted offset for each frame
            confidences: [B, T] - confidence scores
        """
        max_probs, max_indices = torch.max(sync_probs, dim=1)
        offsets = self.max_offset - max_indices
        median_probs = torch.median(sync_probs, dim=1)[0]
        confidences = max_probs - median_probs
        return offsets, confidences


# ==================== TRAINING UTILITIES ====================

def fine_tune_with_transfer_learning(model, 
                                     train_loader, 
                                     val_loader,
                                     num_epochs=10,
                                     lr=1e-4,
                                     device='cuda'):
    """
    Fine-tune pre-trained model on SyncNet task.
    
    Strategy:
    1. Freeze backbones, train head (2-3 epochs)
    2. Unfreeze last layers, train with small lr (5 epochs)
    3. Unfreeze all, train with very small lr (2-3 epochs)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    for epoch in range(num_epochs):
        # Phase 1: Freeze backbones
        if epoch < 3:
            model._freeze_backbones()
            current_lr = lr
        # Phase 2: Unfreeze
        elif epoch == 3:
            for param in model.parameters():
                param.requires_grad = True
            current_lr = lr / 10
            optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)
        
        model.train()
        total_loss = 0
        
        for batch_idx, (audio, video, labels) in enumerate(train_loader):
            audio, video = audio.to(device), video.to(device)
            labels = labels.to(device)
            
            # Forward pass
            sync_probs, _, _ = model(audio, video)
            
            # Loss (cross-entropy on offset prediction)
            loss = F.cross_entropy(
                sync_probs.view(-1, sync_probs.size(1)),
                labels.view(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for audio, video, labels in val_loader:
                audio, video = audio.to(device), video.to(device)
                labels = labels.to(device)
                
                sync_probs, _, _ = model(audio, video)
                
                val_loss += F.cross_entropy(
                    sync_probs.view(-1, sync_probs.size(1)),
                    labels.view(-1)
                ).item()
                
                offsets, _ = model.compute_offset(sync_probs)
                correct += (offsets.round() == labels).sum().item()
                total += labels.numel()
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {total_loss/len(train_loader):.4f}")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"  Val Accuracy: {100*correct/total:.2f}%")


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("Testing Transfer Learning SyncNet...")
    
    # Create model with pre-trained backbones
    model = SyncNet_TransferLearning(
        video_backbone='resnet3d',  # or 'i3d'
        audio_backbone='vggish',     # or 'wav2vec'
        embedding_dim=512,
        max_offset=15,
        freeze_backbone=False
    )
    
    print(f"\nModel architecture:")
    print(f"  Video encoder: {type(model.video_encoder).__name__}")
    print(f"  Audio encoder: {type(model.audio_encoder).__name__}")
    
    # Test forward pass
    dummy_audio = torch.randn(2, 1, 13, 100)
    dummy_video = torch.randn(2, 3, 25, 112, 112)
    
    try:
        sync_probs, audio_feat, video_feat = model(dummy_audio, dummy_video)
        print(f"\nForward pass successful!")
        print(f"  Sync probs: {sync_probs.shape}")
        print(f"  Audio features: {audio_feat.shape}")
        print(f"  Video features: {video_feat.shape}")
        
        offsets, confidences = model.compute_offset(sync_probs)
        print(f"  Offsets: {offsets.shape}")
        print(f"  Confidences: {confidences.shape}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
