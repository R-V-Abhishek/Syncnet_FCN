#!/usr/bin/python
#-*- coding: utf-8 -*-
"""
Practical Example: Using Enhanced SyncNet Models

This script demonstrates how to use the improved SyncNet models with
different configurations and backbones.

Usage:
    python example_usage.py --videofile video.mp4 --model fcn
    python example_usage.py --videofile video.mp4 --model transfer --video_backbone resnet3d
    python example_usage.py --videofile video.mp4 --model original

Author: Enhanced version
Date: 2025-11-22
"""

import torch
import argparse
import os
import sys


def example_1_basic_fcn():
    """Example 1: Basic Fully Convolutional SyncNet"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Fully Convolutional SyncNet")
    print("="*60)
    
    from SyncNetModel_FCN import SyncNetFCN
    
    # Create model
    model = SyncNetFCN(embedding_dim=512, max_offset=15)
    model.eval()
    
    # Dummy inputs
    audio = torch.randn(2, 1, 13, 100)  # [Batch, Channels, MFCC_dim, Time]
    video = torch.randn(2, 3, 25, 112, 112)  # [Batch, Channels, Time, Height, Width]
    
    print(f"Input shapes:")
    print(f"  Audio: {audio.shape}")
    print(f"  Video: {video.shape}")
    
    # Forward pass
    with torch.no_grad():
        sync_probs, audio_feat, video_feat = model(audio, video)
        offsets, confidences = model.compute_offset(sync_probs)
    
    print(f"\nOutput shapes:")
    print(f"  Sync probabilities: {sync_probs.shape}")
    print(f"  Audio features: {audio_feat.shape}")
    print(f"  Video features: {video_feat.shape}")
    print(f"  Offsets: {offsets.shape}")
    print(f"  Confidences: {confidences.shape}")
    
    print(f"\nPredicted offsets: {offsets[0, :10].numpy()}")
    print(f"Confidences: {confidences[0, :10].numpy()}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")


def example_2_attention_fcn():
    """Example 2: FCN with Cross-Modal Attention"""
    print("\n" + "="*60)
    print("EXAMPLE 2: FCN with Cross-Modal Attention")
    print("="*60)
    
    from SyncNetModel_FCN import SyncNetFCN_WithAttention
    
    # Create model with attention
    model = SyncNetFCN_WithAttention(embedding_dim=512, max_offset=15)
    model.eval()
    
    # Dummy inputs
    audio = torch.randn(2, 1, 13, 100)
    video = torch.randn(2, 3, 25, 112, 112)
    
    print(f"Model includes:")
    print(f"  ✓ Audio encoder")
    print(f"  ✓ Video encoder")
    print(f"  ✓ Self-attention layers")
    print(f"  ✓ Cross-modal attention")
    print(f"  ✓ Temporal correlation")
    print(f"  ✓ Sync predictor")
    
    # Forward pass
    with torch.no_grad():
        sync_probs, audio_feat, video_feat = model(audio, video)
    
    print(f"\nOutput shapes:")
    print(f"  Sync probabilities: {sync_probs.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"(~50% more than basic FCN due to attention)")


def example_3_transfer_learning():
    """Example 3: Transfer Learning with Pre-trained Backbones"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Transfer Learning with Pre-trained Backbones")
    print("="*60)
    
    from SyncNet_TransferLearning import SyncNet_TransferLearning
    
    # Configuration 1: 3D ResNet + VGGish
    print("\n--- Configuration 1: 3D ResNet + VGGish ---")
    try:
        model1 = SyncNet_TransferLearning(
            video_backbone='resnet3d',
            audio_backbone='vggish',
            embedding_dim=512,
            max_offset=15,
            freeze_backbone=False
        )
        print("✓ Model created successfully")
        
        # Test forward pass
        audio = torch.randn(1, 1, 13, 100)
        video = torch.randn(1, 3, 25, 112, 112)
        
        with torch.no_grad():
            sync_probs, _, _ = model1(audio, video)
        print(f"✓ Forward pass successful: {sync_probs.shape}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("  (May need to install: pip install torchvggish)")
    
    # Configuration 2: Simple CNN (no external dependencies)
    print("\n--- Configuration 2: Simple CNN (fallback) ---")
    model2 = SyncNet_TransferLearning(
        video_backbone='simple',
        audio_backbone='simple',
        embedding_dim=512,
        max_offset=15
    )
    print("✓ Fallback model created (no external dependencies)")
    
    with torch.no_grad():
        sync_probs, _, _ = model2(audio, video)
    print(f"✓ Forward pass successful: {sync_probs.shape}")


def example_4_inference():
    """Example 4: Full Inference Pipeline"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Full Inference Pipeline")
    print("="*60)
    
    print("\nTo run inference on a video file:")
    print("\n1. Using FCN SyncNet:")
    print("   python SyncNetInstance_FCN.py \\")
    print("       --videofile video.mp4 \\")
    print("       --tmp_dir data/tmp \\")
    print("       --reference test \\")
    print("       --visualize")
    
    print("\n2. With attention:")
    print("   python SyncNetInstance_FCN.py \\")
    print("       --videofile video.mp4 \\")
    print("       --use_attention \\")
    print("       --visualize")
    
    print("\n3. With custom offset range:")
    print("   python SyncNetInstance_FCN.py \\")
    print("       --videofile video.mp4 \\")
    print("       --max_offset 20")
    
    print("\nOutputs:")
    print("  - Frame-by-frame offset predictions")
    print("  - Confidence scores")
    print("  - Visualization plot (if --visualize)")
    print("  - Console statistics (median offset, mean confidence)")


def example_5_feature_extraction():
    """Example 5: Feature Extraction for Downstream Tasks"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Feature Extraction")
    print("="*60)
    
    from SyncNetModel_FCN import SyncNetFCN
    
    model = SyncNetFCN(embedding_dim=512, max_offset=15)
    model.eval()
    
    # Extract video features only
    video = torch.randn(1, 3, 25, 112, 112)
    with torch.no_grad():
        video_features = model.forward_video(video)
    
    print(f"Video features: {video_features.shape}")
    print(f"  Shape: [Batch, Channels, Time]")
    print(f"  Use for: video classification, action recognition, etc.")
    
    # Extract audio features only
    audio = torch.randn(1, 1, 13, 100)
    with torch.no_grad():
        audio_features = model.forward_audio(audio)
    
    print(f"\nAudio features: {audio_features.shape}")
    print(f"  Shape: [Batch, Channels, Time]")
    print(f"  Use for: speech recognition, emotion detection, etc.")
    
    print("\nApplications:")
    print("  1. Audio-visual speech recognition")
    print("  2. Speaker identification")
    print("  3. Emotion recognition")
    print("  4. Deepfake detection")
    print("  5. Video understanding")


def example_6_comparison():
    """Example 6: Model Comparison"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Model Comparison")
    print("="*60)
    
    from SyncNetModel import S as OriginalSyncNet
    from SyncNetModel_FCN import SyncNetFCN
    from SyncNetModel_FCN import SyncNetFCN_WithAttention
    
    models = {
        'Original SyncNet': OriginalSyncNet(num_layers_in_fc_layers=1024),
        'FCN SyncNet': SyncNetFCN(embedding_dim=512, max_offset=15),
        'FCN + Attention': SyncNetFCN_WithAttention(embedding_dim=512, max_offset=15)
    }
    
    print("\n{:<20} {:>15} {:>20}".format("Model", "Parameters", "Output Type"))
    print("-" * 60)
    
    for name, model in models.items():
        num_params = sum(p.numel() for p in model.parameters())
        
        if 'Original' in name:
            output_type = "Single embedding"
        else:
            output_type = "Temporal map"
        
        print("{:<20} {:>15,} {:>20}".format(name, num_params, output_type))
    
    print("\nKey Differences:")
    print("  Original: FC layers, fixed-length input, single offset")
    print("  FCN: Fully conv, variable-length, frame-by-frame")
    print("  FCN+Attn: + cross-modal attention for better fusion")


def example_7_training_setup():
    """Example 7: Training Setup"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Training Setup (Pseudo-code)")
    print("="*60)
    
    print("""
# 1. Create model
from SyncNet_TransferLearning import SyncNet_TransferLearning

model = SyncNet_TransferLearning(
    video_backbone='resnet3d',
    audio_backbone='vggish',
    freeze_backbone=True  # Freeze for first few epochs
).cuda()

# 2. Setup optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# 3. Training loop
for epoch in range(num_epochs):
    # Phase 1: Frozen backbone (epochs 0-2)
    if epoch == 0:
        model._freeze_backbones()
    
    # Phase 2: Unfreeze (epoch 3+)
    elif epoch == 3:
        for param in model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    for audio, video, labels in train_loader:
        # Forward
        sync_probs, _, _ = model(audio.cuda(), video.cuda())
        
        # Loss (cross-entropy over offset predictions)
        loss = F.cross_entropy(
            sync_probs.view(-1, sync_probs.size(1)),
            labels.cuda().view(-1)
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step()

# 4. Save model
torch.save(model.state_dict(), 'syncnet_finetuned.pth')
    """)
    
    print("Training Tips:")
    print("  1. Start with frozen backbone, train head only (2-3 epochs)")
    print("  2. Unfreeze gradually with lower learning rate")
    print("  3. Use strong data augmentation (temporal jitter, mixup)")
    print("  4. Large batch size helps (32-64 if GPU allows)")
    print("  5. Monitor both loss and frame-level accuracy")


def example_8_deployment():
    """Example 8: Deployment Options"""
    print("\n" + "="*60)
    print("EXAMPLE 8: Deployment Options")
    print("="*60)
    
    print("\n1. ONNX Export (Cross-platform)")
    print("-" * 40)
    print("""
from SyncNetModel_FCN import SyncNetFCN

model = SyncNetFCN(embedding_dim=512, max_offset=15)
model.eval()

# Dummy inputs
audio = torch.randn(1, 1, 13, 100)
video = torch.randn(1, 3, 25, 112, 112)

# Export
torch.onnx.export(
    model,
    (audio, video),
    "syncnet_fcn.onnx",
    input_names=['audio', 'video'],
    output_names=['sync_probs', 'audio_feat', 'video_feat'],
    dynamic_axes={
        'audio': {3: 'audio_time'},
        'video': {2: 'video_time'}
    }
)

# Use with ONNX Runtime
import onnxruntime
session = onnxruntime.InferenceSession("syncnet_fcn.onnx")
outputs = session.run(None, {'audio': audio, 'video': video})
    """)
    
    print("\n2. Quantization (Faster, Smaller)")
    print("-" * 40)
    print("""
import torch.quantization

model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.quantization.prepare(model)

# Calibrate
for data in calibration_data:
    model_prepared(data)

# Convert to INT8
model_int8 = torch.quantization.convert(model_prepared)

# Save
torch.save(model_int8.state_dict(), 'syncnet_int8.pth')

# Result: ~4x smaller, ~2-3x faster
    """)
    
    print("\n3. TorchScript (C++ Deployment)")
    print("-" * 40)
    print("""
# Trace model
model.eval()
traced = torch.jit.trace(model, (audio, video))
traced.save("syncnet_traced.pt")

# Use in C++
// #include <torch/script.h>
// torch::jit::script::Module module = torch::jit::load("syncnet_traced.pt");
    """)
    
    print("\nDeployment Comparison:")
    print("  ONNX: Best for cross-platform (Python, C++, C#, JS)")
    print("  Quantization: Best for speed/size (mobile, edge)")
    print("  TorchScript: Best for C++ integration")


def main():
    parser = argparse.ArgumentParser(description='SyncNet Enhancement Examples')
    parser.add_argument('--example', type=str, default='all',
                       choices=['all', '1', '2', '3', '4', '5', '6', '7', '8'],
                       help='Which example to run (default: all)')
    args = parser.parse_args()
    
    examples = {
        '1': example_1_basic_fcn,
        '2': example_2_attention_fcn,
        '3': example_3_transfer_learning,
        '4': example_4_inference,
        '5': example_5_feature_extraction,
        '6': example_6_comparison,
        '7': example_7_training_setup,
        '8': example_8_deployment,
    }
    
    if args.example == 'all':
        for func in examples.values():
            func()
    else:
        examples[args.example]()
    
    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Review IMPROVEMENTS.md for detailed explanations")
    print("  2. Check ENHANCEMENT_SUMMARY.md for recommendations")
    print("  3. Try inference: python SyncNetInstance_FCN.py --videofile video.mp4")
    print("  4. Integrate pre-trained backbones from SyncNet_TransferLearning.py")
    print("  5. Fine-tune on your dataset")
    print("\nAll code is ready to use! 🚀")


if __name__ == "__main__":
    main()
