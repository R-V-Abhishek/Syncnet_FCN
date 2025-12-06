#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Stream/Video Sync Detection with FCN-SyncNet

Detect audio-video sync offset in video files or live HLS streams.
Uses trained FCN model (epoch 2) with calibration for accurate results.

Usage:
    # Video file
    python test_sync_detection.py --video path/to/video.mp4
    
    # HLS stream
    python test_sync_detection.py --hls http://example.com/stream.m3u8 --duration 15
    
    # Compare FCN with Original SyncNet
    python test_sync_detection.py --video video.mp4 --compare
    
    # Original SyncNet only
    python test_sync_detection.py --video video.mp4 --original
    
    # With verbose output
    python test_sync_detection.py --video video.mp4 --verbose
    
    # Custom model
    python test_sync_detection.py --video video.mp4 --model checkpoints/custom.pth
"""

import os
import sys
import argparse
import torch
import time

# Enable UTF-8 output on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def load_model(model_path=None, device='cpu'):
    """Load the FCN-SyncNet model with trained weights."""
    from SyncNetModel_FCN import StreamSyncFCN
    
    # Default to our best trained model
    if model_path is None:
        model_path = 'checkpoints/syncnet_fcn_epoch2.pth'
    
    # Check if it's a checkpoint file (.pth) or original syncnet model
    if model_path.endswith('.pth') and os.path.exists(model_path):
        # Load our trained FCN checkpoint
        model = StreamSyncFCN(
            max_offset=15,
            pretrained_syncnet_path=None,
            auto_load_pretrained=False
        )
        
        checkpoint = torch.load(model_path, map_location=device)
        
        # Load only encoder weights (skip mismatched head)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            encoder_state = {k: v for k, v in state_dict.items()
                           if 'audio_encoder' in k or 'video_encoder' in k}
            model.load_state_dict(encoder_state, strict=False)
            epoch = checkpoint.get('epoch', '?')
            print(f"✓ Loaded trained FCN model (epoch {epoch})")
        else:
            model.load_state_dict(checkpoint, strict=False)
            print(f"✓ Loaded model weights")
            
    elif os.path.exists(model_path):
        # Load original SyncNet pretrained model
        model = StreamSyncFCN(
            pretrained_syncnet_path=model_path,
            auto_load_pretrained=True
        )
        print(f"✓ Loaded pretrained SyncNet from: {model_path}")
    else:
        print(f"⚠ Model not found: {model_path}")
        print("  Using random initialization (results may be unreliable)")
        model = StreamSyncFCN(
            pretrained_syncnet_path=None,
            auto_load_pretrained=False
        )
    
    model.eval()
    return model.to(device)


def load_original_syncnet(model_path='data/syncnet_v2.model', device='cpu'):
    """Load the original SyncNet model for comparison."""
    from SyncNetInstance import SyncNetInstance
    
    model = SyncNetInstance()
    model.loadParameters(model_path)
    print(f"✓ Loaded Original SyncNet from: {model_path}")
    return model


def run_original_syncnet(model, video_path, verbose=False):
    """
    Run original SyncNet on a video file.
    
    Returns:
        dict with offset_frames, offset_seconds, confidence, processing_time
    """
    import argparse
    
    # Create required options object
    opt = argparse.Namespace()
    opt.tmp_dir = 'data/work/pytmp'
    opt.reference = 'original_test'
    opt.batch_size = 20
    opt.vshift = 15
    
    start_time = time.time()
    
    # Run evaluation
    offset, confidence, dist = model.evaluate(opt, video_path)
    
    elapsed = time.time() - start_time
    
    return {
        'offset_frames': offset,
        'offset_seconds': offset / 25.0,
        'confidence': confidence,
        'min_dist': dist,
        'processing_time': elapsed
    }


def apply_calibration(raw_offset, calibration_offset=3, calibration_scale=-0.5, reference_raw=-15):
    """
    Apply linear calibration to raw model output.
    
    Calibration formula: calibrated = offset + scale * (raw - reference)
    Default: calibrated = 3 + (-0.5) * (raw - (-15))
    
    This corrects for systematic bias in the FCN model's predictions.
    """
    return calibration_offset + calibration_scale * (raw_offset - reference_raw)


def detect_sync(video_path=None, hls_url=None, duration=10, model=None, 
                verbose=False, use_calibration=True):
    """
    Detect audio-video sync offset.
    
    Args:
        video_path: Path to video file
        hls_url: HLS stream URL (.m3u8)
        duration: Capture duration for HLS (seconds)
        model: Pre-loaded model (optional)
        verbose: Print detailed output
        use_calibration: Apply calibration correction
        
    Returns:
        dict with offset_frames, offset_seconds, confidence, raw_offset
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model if not provided
    if model is None:
        model = load_model(device=device)
    
    start_time = time.time()
    
    # Process video or HLS
    if video_path:
        # Use the same method as detect_sync.py for consistency
        if use_calibration:
            offset, confidence, raw_offset = model.detect_offset_correlation(
                video_path,
                calibration_offset=3,
                calibration_scale=-0.5,
                calibration_baseline=-15,
                verbose=verbose
            )
        else:
            raw_offset, confidence = model.process_video_file(
                video_path, 
                verbose=verbose
            )
            offset = raw_offset
            
    elif hls_url:
        raw_offset, confidence = model.process_hls_stream(
            hls_url,
            segment_duration=duration,
            verbose=verbose
        )
        if use_calibration:
            offset = apply_calibration(raw_offset)
        else:
            offset = raw_offset
    else:
        raise ValueError("Must provide either video_path or hls_url")
    
    elapsed = time.time() - start_time
    
    return {
        'offset_frames': round(offset),
        'offset_seconds': offset / 25.0,
        'confidence': confidence,
        'raw_offset': raw_offset if 'raw_offset' in dir() else offset,
        'processing_time': elapsed
    }


def print_results(result, source_name, model_name="FCN-SyncNet"):
    """Print formatted results."""
    offset = result['offset_frames']
    offset_sec = result['offset_seconds']
    confidence = result['confidence']
    elapsed = result['processing_time']
    
    print()
    print("=" * 60)
    print(f"  {model_name} Detection Result")
    print("=" * 60)
    print(f"  Source:     {source_name}")
    print(f"  Offset:     {offset:+d} frames ({offset_sec:+.3f}s)")
    print(f"  Confidence: {confidence:.6f}")
    print(f"  Time:       {elapsed:.2f}s")
    print("=" * 60)
    
    # Interpretation
    if offset > 1:
        print(f"  → Audio is {offset} frames AHEAD of video")
        print(f"    (delay audio by {abs(offset_sec):.3f}s to fix)")
    elif offset < -1:
        print(f"  → Audio is {abs(offset)} frames BEHIND video")
        print(f"    (advance audio by {abs(offset_sec):.3f}s to fix)")
    else:
        print("  ✓ Audio and video are IN SYNC")
    print()


def print_comparison(fcn_result, original_result, source_name):
    """Print side-by-side comparison of both models."""
    print()
    print("╔" + "═" * 70 + "╗")
    print("║" + "  Model Comparison Results".center(70) + "║")
    print("╚" + "═" * 70 + "╝")
    print()
    print(f"  Source: {source_name}")
    print()
    print("  " + "-" * 66)
    print(f"  {'Metric':<20} {'FCN-SyncNet':>20} {'Original SyncNet':>20}")
    print("  " + "-" * 66)
    
    fcn_off = fcn_result['offset_frames']
    orig_off = original_result['offset_frames']
    
    print(f"  {'Offset (frames)':<20} {fcn_off:>+20d} {orig_off:>+20d}")
    print(f"  {'Offset (seconds)':<20} {fcn_result['offset_seconds']:>+20.3f} {original_result['offset_seconds']:>+20.3f}")
    print(f"  {'Confidence':<20} {fcn_result['confidence']:>20.4f} {original_result['confidence']:>20.4f}")
    print(f"  {'Time (seconds)':<20} {fcn_result['processing_time']:>20.2f} {original_result['processing_time']:>20.2f}")
    print("  " + "-" * 66)
    
    # Agreement check
    diff = abs(fcn_off - orig_off)
    if diff == 0:
        print("  ✓ Both models AGREE perfectly!")
    elif diff <= 2:
        print(f"  ≈ Models differ by {diff} frame(s) (close agreement)")
    else:
        print(f"  ✗ Models differ by {diff} frames")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='FCN-SyncNet - Audio-Video Sync Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Video file:   python test_sync_detection.py --video video.mp4
  HLS stream:   python test_sync_detection.py --hls http://stream.m3u8 --duration 15
  Compare:      python test_sync_detection.py --video video.mp4 --compare
  Original:     python test_sync_detection.py --video video.mp4 --original
  Verbose:      python test_sync_detection.py --video video.mp4 --verbose
        """
    )
    
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--hls', type=str, help='HLS stream URL (.m3u8)')
    parser.add_argument('--model', type=str, default=None,
                       help='Model checkpoint (default: checkpoints/syncnet_fcn_epoch2.pth)')
    parser.add_argument('--duration', type=int, default=10,
                       help='Duration for HLS capture (seconds, default: 10)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed processing info')
    parser.add_argument('--no-calibration', action='store_true',
                       help='Disable calibration correction')
    parser.add_argument('--json', action='store_true',
                       help='Output results as JSON')
    parser.add_argument('--compare', action='store_true',
                       help='Compare FCN-SyncNet with Original SyncNet')
    parser.add_argument('--original', action='store_true',
                       help='Use Original SyncNet only (not FCN)')
    
    args = parser.parse_args()
    
    # Validate input
    if not args.video and not args.hls:
        print("Error: Please provide either --video or --hls")
        parser.print_help()
        return 1
    
    # Original SyncNet doesn't support HLS
    if args.hls and (args.original or args.compare):
        print("Error: Original SyncNet does not support HLS streams")
        print("       Use --video for comparison mode")
        return 1
    
    if not args.json:
        print()
        if args.original:
            print("╔══════════════════════════════════════════════════════════════╗")
            print("║      Original SyncNet - Audio-Video Sync Detection           ║")
            print("╚══════════════════════════════════════════════════════════════╝")
        elif args.compare:
            print("╔══════════════════════════════════════════════════════════════╗")
            print("║      Sync Detection - FCN vs Original SyncNet                ║")
            print("╚══════════════════════════════════════════════════════════════╝")
        else:
            print("╔══════════════════════════════════════════════════════════════╗")
            print("║      FCN-SyncNet - Audio-Video Sync Detection                ║")
            print("╚══════════════════════════════════════════════════════════════╝")
        print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not args.json:
        print(f"Device: {device}")
    
    try:
        source = os.path.basename(args.video) if args.video else args.hls
        
        # Run Original SyncNet only
        if args.original:
            original_model = load_original_syncnet()
            if not args.json:
                print(f"\nProcessing: {args.video}")
            result = run_original_syncnet(original_model, args.video, args.verbose)
            
            if args.json:
                import json
                result['source'] = source
                result['model'] = 'original_syncnet'
                print(json.dumps(result, indent=2))
            else:
                print_results(result, source, "Original SyncNet")
            return 0
        
        # Run comparison mode
        if args.compare:
            # Load both models
            fcn_model = load_model(args.model, device)
            original_model = load_original_syncnet()
            
            if not args.json:
                print(f"\nProcessing: {args.video}")
                print("\n[1/2] Running FCN-SyncNet...")
            
            fcn_result = detect_sync(
                video_path=args.video,
                model=fcn_model,
                verbose=args.verbose,
                use_calibration=not args.no_calibration
            )
            
            if not args.json:
                print("[2/2] Running Original SyncNet...")
            
            original_result = run_original_syncnet(original_model, args.video, args.verbose)
            
            if args.json:
                import json
                output = {
                    'source': source,
                    'fcn_syncnet': fcn_result,
                    'original_syncnet': original_result
                }
                print(json.dumps(output, indent=2))
            else:
                print_comparison(fcn_result, original_result, source)
            return 0
        
        # Default: FCN-SyncNet only
        model = load_model(args.model, device)
        
        if args.video:
            if not args.json:
                print(f"\nProcessing: {args.video}")
            result = detect_sync(
                video_path=args.video,
                model=model,
                verbose=args.verbose,
                use_calibration=not args.no_calibration
            )
            
        else:  # HLS
            if not args.json:
                print(f"\nProcessing HLS: {args.hls}")
                print(f"Capturing {args.duration} seconds...")
            result = detect_sync(
                hls_url=args.hls,
                duration=args.duration,
                model=model,
                verbose=args.verbose,
                use_calibration=not args.no_calibration
            )
            source = args.hls
        
        # Output results
        if args.json:
            import json
            result['source'] = source
            print(json.dumps(result, indent=2))
        else:
            print_results(result, source)
        
        return 0
        
    except FileNotFoundError:
        print(f"\n✗ Error: File not found - {args.video or args.hls}")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
