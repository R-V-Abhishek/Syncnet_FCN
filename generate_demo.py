#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate Demo Video for FCN-SyncNet

Creates demonstration videos showing sync detection with different offsets.
Outputs a comparison video and terminal recording for presentation.

Usage:
    python generate_demo.py
    python generate_demo.py --output demo_output/

Author: R-V-Abhishek
"""

import argparse
import os
import subprocess
import sys
import time

import torch


def create_offset_videos(source_video, output_dir, offsets=[0, 5, 12]):
    """Create test videos with known audio offsets."""
    os.makedirs(output_dir, exist_ok=True)
    
    created = []
    for offset in offsets:
        if offset == 0:
            # Copy original
            output_path = os.path.join(output_dir, 'test_offset_0.avi')
            cmd = ['ffmpeg', '-y', '-i', source_video, '-c', 'copy', output_path]
        else:
            # Add audio delay (offset in frames, 40ms per frame at 25fps)
            delay_ms = offset * 40
            output_path = os.path.join(output_dir, f'test_offset_{offset}.avi')
            cmd = ['ffmpeg', '-y', '-i', source_video, 
                   '-af', f'adelay={delay_ms}|{delay_ms}',
                   '-c:v', 'copy', output_path]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        created.append((output_path, offset))
        print(f"  Created: test_offset_{offset}.avi (+{offset} frames)")
    
    return created


def run_demo(model, test_videos, baseline_offset=3):
    """Run detection on test videos and print results."""
    results = []
    
    print()
    print("=" * 70)
    print("  FCN-SyncNet Demo - Audio-Video Sync Detection")
    print("=" * 70)
    print()
    
    for video_path, added_offset in test_videos:
        expected = baseline_offset - added_offset  # Original has +3, adding offset shifts it
        
        offset, conf, raw = model.detect_offset_correlation(
            video_path,
            calibration_offset=3,
            calibration_scale=-0.5,
            calibration_baseline=-15,
            verbose=False
        )
        
        error = abs(offset - expected)
        status = "✓" if error <= 3 else "✗"
        
        result = {
            'video': os.path.basename(video_path),
            'added_offset': added_offset,
            'expected': expected,
            'detected': offset,
            'error': error,
            'status': status
        }
        results.append(result)
        
        print(f"  {status} {result['video']}")
        print(f"      Added offset: +{added_offset} frames")
        print(f"      Expected:     {expected:+d} frames")
        print(f"      Detected:     {offset:+d} frames")
        print(f"      Error:        {error} frames")
        print()
    
    # Summary
    total_error = sum(r['error'] for r in results)
    correct = sum(1 for r in results if r['error'] <= 3)
    
    print("-" * 70)
    print(f"  Summary: {correct}/{len(results)} correct (within 3 frames)")
    print(f"  Total error: {total_error} frames")
    print("=" * 70)
    
    return results


def compare_with_original_syncnet(test_videos, baseline_offset=3):
    """Run original SyncNet for comparison."""
    print()
    print("=" * 70)
    print("  Original SyncNet Comparison")
    print("=" * 70)
    print()
    
    original_results = []
    for video_path, added_offset in test_videos:
        expected = baseline_offset - added_offset
        
        # Run original demo_syncnet.py (use same Python interpreter)
        result = subprocess.run(
            [sys.executable, 'demo_syncnet.py', '--videofile', video_path, 
             '--tmp_dir', 'data/work/pytmp'],
            capture_output=True, text=True
        )
        
        # Parse output
        detected = None
        for line in result.stdout.split('\n'):
            if 'AV offset' in line:
                detected = int(line.split(':')[1].strip())
                break
        
        if detected is not None:
            error = abs(detected - expected)
            status = "✓" if error <= 3 else "✗"
            print(f"  {status} {os.path.basename(video_path)}: detected={detected:+d}, expected={expected:+d}, error={error}")
            original_results.append({'error': error})
        else:
            print(f"  ? {os.path.basename(video_path)}: detection failed")
            original_results.append({'error': None})
    
    print("=" * 70)
    return original_results


def main():
    parser = argparse.ArgumentParser(description='Generate FCN-SyncNet demo')
    parser.add_argument('--output', '-o', default='demo_output',
                       help='Output directory for test videos')
    parser.add_argument('--source', '-s', default='data/example.avi',
                       help='Source video file')
    parser.add_argument('--compare', '-c', action='store_true',
                       help='Also run original SyncNet for comparison')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up test videos after demo')
    
    args = parser.parse_args()
    
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║         FCN-SyncNet Demo - Audio-Video Sync Detection            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    
    # Check source video
    if not os.path.exists(args.source):
        print(f"Error: Source video not found: {args.source}")
        sys.exit(1)
    
    # Create test videos
    print("Creating test videos with different offsets...")
    test_videos = create_offset_videos(args.source, args.output, offsets=[0, 5, 12])
    
    # Load FCN model
    print()
    print("Loading FCN-SyncNet model...")
    from SyncNetModel_FCN import StreamSyncFCN
    
    model = StreamSyncFCN(max_offset=15, pretrained_syncnet_path=None, auto_load_pretrained=False)
    checkpoint = torch.load('checkpoints/syncnet_fcn_epoch2.pth', map_location='cpu')
    encoder_state = {k: v for k, v in checkpoint['model_state_dict'].items()
                    if 'audio_encoder' in k or 'video_encoder' in k}
    model.load_state_dict(encoder_state, strict=False)
    model.eval()
    print(f"  ✓ Loaded checkpoint (epoch {checkpoint.get('epoch', '?')})")
    
    # Run FCN demo
    fcn_results = run_demo(model, test_videos, baseline_offset=3)
    
    # Optionally compare with original
    original_results = None
    if args.compare:
        original_results = compare_with_original_syncnet(test_videos, baseline_offset=3)
        
        # Print comparison summary
        fcn_errors = [r['error'] for r in fcn_results]
        orig_errors = [r['error'] for r in original_results if r['error'] is not None]
        
        print()
        print("╔══════════════════════════════════════════════════════════════════╗")
        print("║                     Comparison Summary                            ║")
        print("╠══════════════════════════════════════════════════════════════════╣")
        fcn_total = sum(fcn_errors)
        fcn_correct = sum(1 for e in fcn_errors if e <= 3)
        print(f"║  FCN-SyncNet:      {fcn_correct}/{len(fcn_results)} correct, {fcn_total} frames total error        ║")
        if orig_errors:
            orig_total = sum(orig_errors)
            orig_correct = sum(1 for e in orig_errors if e <= 3)
            print(f"║  Original SyncNet: {orig_correct}/{len(orig_errors)} correct, {orig_total} frames total error        ║")
        print("╠══════════════════════════════════════════════════════════════════╣")
        print("║  FCN-SyncNet: Research prototype with real-time capability       ║")
        print("║  Status: Working but needs more training data/epochs             ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
    
    # Cleanup
    if args.cleanup:
        print()
        print("Cleaning up test videos...")
        for video_path, _ in test_videos:
            if os.path.exists(video_path):
                os.remove(video_path)
        if os.path.exists(args.output) and not os.listdir(args.output):
            os.rmdir(args.output)
        print("  Done.")
    
    print()
    print("Demo complete!")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
