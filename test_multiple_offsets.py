#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test FCN-SyncNet and Original SyncNet with multiple offset videos.

Creates test videos with known offsets and compares detection accuracy.
"""

import subprocess
import os
import sys

# Enable UTF-8 output on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def create_offset_video(source_video, offset_frames, output_path):
    """
    Create a video with audio offset.
    
    Args:
        source_video: Path to source video
        offset_frames: Positive = audio delayed (behind), Negative = audio ahead
        output_path: Output video path
    """
    if os.path.exists(output_path):
        return True
        
    if offset_frames >= 0:
        # Delay audio - add silence at start
        delay_ms = offset_frames * 40  # 40ms per frame at 25fps
        cmd = [
            'ffmpeg', '-y', '-i', source_video,
            '-af', f'adelay={delay_ms}|{delay_ms}',
            '-c:v', 'copy', output_path
        ]
    else:
        # Advance audio - trim start of audio
        trim_sec = abs(offset_frames) * 0.04
        cmd = [
            'ffmpeg', '-y', '-i', source_video,
            '-af', f'atrim=start={trim_sec},asetpts=PTS-STARTPTS',
            '-c:v', 'copy', output_path
        ]
    
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def test_fcn_model(video_path, verbose=False):
    """Test with FCN-SyncNet model."""
    from SyncNetModel_FCN import StreamSyncFCN
    import torch
    
    model = StreamSyncFCN(
        max_offset=15,
        pretrained_syncnet_path=None,
        auto_load_pretrained=False
    )
    
    checkpoint = torch.load('checkpoints/syncnet_fcn_epoch2.pth', map_location='cpu')
    encoder_state = {k: v for k, v in checkpoint['model_state_dict'].items()
                    if 'audio_encoder' in k or 'video_encoder' in k}
    model.load_state_dict(encoder_state, strict=False)
    model.eval()
    
    offset, confidence, raw_offset = model.detect_offset_correlation(
        video_path,
        calibration_offset=3,
        calibration_scale=-0.5,
        calibration_baseline=-15,
        verbose=verbose
    )
    
    return int(round(offset)), confidence


def test_original_model(video_path, verbose=False):
    """Test with Original SyncNet model."""
    import argparse
    from SyncNetInstance import SyncNetInstance
    
    model = SyncNetInstance()
    model.loadParameters('data/syncnet_v2.model')
    
    opt = argparse.Namespace()
    opt.tmp_dir = 'data/work/pytmp'
    opt.reference = 'offset_test'
    opt.batch_size = 20
    opt.vshift = 15
    
    offset, confidence, dist = model.evaluate(opt, video_path)
    return int(offset), confidence


def main():
    print()
    print("=" * 75)
    print("  Multi-Offset Sync Detection Test")
    print("  Comparing FCN-SyncNet vs Original SyncNet")
    print("=" * 75)
    print()
    
    source_video = 'data/example.avi'
    
    # The source video has an inherent offset of +3 frames
    # So when we add offset X, the expected detection is (3 + X) for Original SyncNet
    base_offset = 3  # Known offset in example.avi
    
    # Test offsets to add
    test_offsets = [0, 5, 10, -5, -10]
    
    print("Creating test videos with various offsets...")
    print()
    
    results = []
    
    for added_offset in test_offsets:
        output_path = f'data/test_offset_{added_offset:+d}.avi'
        expected = base_offset + added_offset
        
        print(f"  Creating {output_path} (adding {added_offset:+d} frames)...")
        if not create_offset_video(source_video, added_offset, output_path):
            print(f"    Failed to create video!")
            continue
        
        print(f"    Testing FCN-SyncNet...")
        fcn_offset, fcn_conf = test_fcn_model(output_path)
        
        print(f"    Testing Original SyncNet...")
        orig_offset, orig_conf = test_original_model(output_path)
        
        results.append({
            'added': added_offset,
            'expected': expected,
            'fcn': fcn_offset,
            'original': orig_offset,
            'fcn_error': abs(fcn_offset - expected),
            'orig_error': abs(orig_offset - expected)
        })
        print()
    
    # Print results table
    print()
    print("=" * 75)
    print("  RESULTS")
    print("=" * 75)
    print()
    print(f"  {'Added':<8} {'Expected':<10} {'FCN':<10} {'Original':<10} {'FCN Err':<10} {'Orig Err':<10}")
    print("  " + "-" * 68)
    
    fcn_total_error = 0
    orig_total_error = 0
    
    for r in results:
        fcn_mark = "✓" if r['fcn_error'] <= 2 else "✗"
        orig_mark = "✓" if r['orig_error'] <= 2 else "✗"
        print(f"  {r['added']:+8d} {r['expected']:+10d} {r['fcn']:+10d} {r['original']:+10d} {r['fcn_error']:>6d} {fcn_mark:<3} {r['orig_error']:>6d} {orig_mark}")
        fcn_total_error += r['fcn_error']
        orig_total_error += r['orig_error']
    
    print("  " + "-" * 68)
    print(f"  {'TOTAL ERROR:':<28} {fcn_total_error:>10d}     {orig_total_error:>10d}")
    print()
    
    # Summary
    fcn_correct = sum(1 for r in results if r['fcn_error'] <= 2)
    orig_correct = sum(1 for r in results if r['orig_error'] <= 2)
    
    print(f"  FCN-SyncNet:      {fcn_correct}/{len(results)} correct (within 2 frames)")
    print(f"  Original SyncNet: {orig_correct}/{len(results)} correct (within 2 frames)")
    print()
    
    # Cleanup test videos
    print("Cleaning up test videos...")
    for added_offset in test_offsets:
        output_path = f'data/test_offset_{added_offset:+d}.avi'
        if os.path.exists(output_path):
            os.remove(output_path)
    print("Done!")


if __name__ == "__main__":
    main()
