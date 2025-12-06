#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FCN-SyncNet CLI Tool - Audio-Video Sync Detection

Detects audio-video synchronization offset in video files using
a Fully Convolutional Neural Network with transfer learning.

Usage:
    python detect_sync.py video.mp4
    python detect_sync.py video.mp4 --verbose
    python detect_sync.py video.mp4 --output results.json

Author: R-V-Abhishek
"""

import argparse
import json
import os
import sys
import time

import torch


def load_model(checkpoint_path='checkpoints/syncnet_fcn_epoch2.pth', max_offset=15):
    """Load the FCN-SyncNet model with trained weights."""
    from SyncNetModel_FCN import StreamSyncFCN
    
    model = StreamSyncFCN(
        max_offset=max_offset,
        pretrained_syncnet_path=None,
        auto_load_pretrained=False
    )
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # Load only encoder weights
        encoder_state = {k: v for k, v in checkpoint['model_state_dict'].items()
                        if 'audio_encoder' in k or 'video_encoder' in k}
        model.load_state_dict(encoder_state, strict=False)
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"✓ Loaded model from {checkpoint_path} (epoch {epoch})")
    else:
        # Fall back to pretrained SyncNet
        print(f"! Checkpoint not found: {checkpoint_path}")
        print("  Loading pretrained SyncNet weights...")
        model = StreamSyncFCN(
            max_offset=max_offset,
            pretrained_syncnet_path='data/syncnet_v2.model',
            auto_load_pretrained=True
        )
    
    model.eval()
    return model


def detect_offset(model, video_path, verbose=False):
    """
    Detect AV offset in a video file.
    
    Returns:
        dict with offset, confidence, raw_offset, and processing time
    """
    start_time = time.time()
    
    offset, confidence, raw_offset = model.detect_offset_correlation(
        video_path,
        calibration_offset=3,
        calibration_scale=-0.5,
        calibration_baseline=-15,
        verbose=verbose
    )
    
    processing_time = time.time() - start_time
    
    return {
        'video': video_path,
        'offset_frames': int(offset),
        'offset_seconds': round(offset / 25.0, 3),  # Assuming 25 fps
        'confidence': round(float(confidence), 6),
        'raw_offset': int(raw_offset),
        'processing_time': round(processing_time, 2)
    }


def print_result(result, verbose=False):
    """Print detection result in a nice format."""
    print()
    print("=" * 50)
    print("  FCN-SyncNet Detection Result")
    print("=" * 50)
    print(f"  Video:      {os.path.basename(result['video'])}")
    print(f"  Offset:     {result['offset_frames']:+d} frames ({result['offset_seconds']:+.3f}s)")
    print(f"  Confidence: {result['confidence']:.6f}")
    print(f"  Time:       {result['processing_time']:.2f}s")
    print("=" * 50)
    
    # Interpretation
    offset = result['offset_frames']
    if abs(offset) <= 1:
        print("  ✓ Audio and video are IN SYNC")
    elif offset > 0:
        print(f"  ! Audio is {abs(offset)} frames BEHIND video")
        print(f"    (delay audio by {abs(result['offset_seconds']):.3f}s to fix)")
    else:
        print(f"  ! Audio is {abs(offset)} frames AHEAD of video")
        print(f"    (advance audio by {abs(result['offset_seconds']):.3f}s to fix)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='FCN-SyncNet: Detect audio-video sync offset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect_sync.py video.mp4
  python detect_sync.py video.mp4 --verbose
  python detect_sync.py video.mp4 --output result.json
  python detect_sync.py video.mp4 --model checkpoints/custom.pth

Output:
  Positive offset = audio behind video (delay audio to fix)
  Negative offset = audio ahead of video (advance audio to fix)
        """
    )
    
    parser.add_argument('video', help='Path to video file (MP4, AVI, MOV, etc.)')
    parser.add_argument('--model', '-m', default='checkpoints/syncnet_fcn_epoch2.pth',
                       help='Path to model checkpoint (default: checkpoints/syncnet_fcn_epoch2.pth)')
    parser.add_argument('--output', '-o', help='Save result to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed processing info')
    parser.add_argument('--json', '-j', action='store_true',
                       help='Output only JSON (for scripting)')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    # Load model
    if not args.json:
        print()
        print("FCN-SyncNet Audio-Video Sync Detector")
        print("-" * 40)
    
    try:
        model = load_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Detect offset
    try:
        result = detect_offset(model, args.video, verbose=args.verbose)
    except Exception as e:
        print(f"Error processing video: {e}")
        sys.exit(1)
    
    # Output result
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_result(result, verbose=args.verbose)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, indent=2, fp=f)
        if not args.json:
            print(f"Result saved to: {args.output}")
    
    return result['offset_frames']


if __name__ == '__main__':
    sys.exit(main())
