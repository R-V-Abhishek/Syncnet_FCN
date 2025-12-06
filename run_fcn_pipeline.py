class Logger:
    def __init__(self, level="INFO", realtime=False):
        self.levels = {"ERROR": 0, "WARNING": 1, "INFO": 2}
        self.realtime = realtime
        self.level = "ERROR" if realtime else level

    def log(self, msg, level="INFO"):
        if self.levels[level] <= self.levels[self.level]:
            print(f"[{level}] {msg}")

    def info(self, msg):
        self.log(msg, "INFO")

    def warning(self, msg):
        self.log(msg, "WARNING")

    def error(self, msg):
        self.log(msg, "ERROR")
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_fcn_pipeline.py

Pipeline for Fully Convolutional SyncNet (FCN-SyncNet) AV Sync Detection
=======================================================================

This script demonstrates how to use the improved StreamSyncFCN model for audio-video synchronization detection on video files or streams.
It handles preprocessing, buffering, and model inference, and outputs sync offset/confidence for each input.

Usage:
    python run_fcn_pipeline.py --video path/to/video.mp4 [--pretrained path/to/weights] [--window_size 25] [--stride 5] [--buffer_size 100] [--use_attention] [--trace]

Requirements:
    - Python 3.x
    - PyTorch
    - OpenCV
    - ffmpeg (installed and in PATH)
    - python_speech_features
    - numpy, scipy
    - SyncNetModel_FCN.py in the same directory or PYTHONPATH

Author: R V Abhishek
"""

import argparse
from SyncNetModel_FCN import StreamSyncFCN
import os


def main():
    parser = argparse.ArgumentParser(description="FCN SyncNet AV Sync Pipeline")
    parser.add_argument('--video', type=str, help='Path to input video file')
    parser.add_argument('--folder', type=str, help='Path to folder containing video files (batch mode)')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained SyncNet weights (optional)')
    parser.add_argument('--window_size', type=int, default=25, help='Frames per window (default: 25)')
    parser.add_argument('--stride', type=int, default=5, help='Window stride (default: 5)')
    parser.add_argument('--buffer_size', type=int, default=100, help='Temporal buffer size (default: 100)')
    parser.add_argument('--use_attention', action='store_true', help='Use attention model (default: False)')
    parser.add_argument('--trace', action='store_true', help='Return per-window trace (default: False)')
    parser.add_argument('--temp_dir', type=str, default='temp', help='Temporary directory for audio extraction')
    parser.add_argument('--target_size', type=int, nargs=2, default=[112, 112], help='Target video frame size (HxW)')
    parser.add_argument('--realtime', action='store_true', help='Enable real-time mode (minimal checks/logging)')
    parser.add_argument('--keep_temp', action='store_true', help='Keep temporary files for debugging (default: False)')
    parser.add_argument('--summary', action='store_true', help='Print summary statistics for batch mode (default: False)')
    args = parser.parse_args()

    logger = Logger(realtime=args.realtime)
    # Buffer/latency awareness and user guidance
    frame_rate = 25  # Default, can be parameterized if needed
    effective_latency_frames = args.window_size + (args.buffer_size - 1) * args.stride
    effective_latency_sec = effective_latency_frames / frame_rate
    if not args.realtime:
        logger.info("")
        logger.info("Buffer/Latency Settings:")
        logger.info(f"  Window size: {args.window_size} frames")
        logger.info(f"  Stride: {args.stride} frames")
        logger.info(f"  Buffer size: {args.buffer_size} windows")
        logger.info(f"  Effective latency: {effective_latency_frames} frames (~{effective_latency_sec:.2f} sec @ {frame_rate} FPS)")
        if effective_latency_sec > 2.0:
            logger.warning("High effective latency. Consider reducing buffer size or stride for real-time applications.")

    import shutil
    import glob
    import csv
    temp_cleanup_needed = not args.keep_temp

    def process_one_video(video_path):
        # Real-time compatible input quality checks (sample only first few frames/samples, or skip if --realtime)
        if not args.realtime:
            import numpy as np
            def check_video_audio_quality_realtime(video_path, temp_dir, target_size):
                # Check first few video frames
                import cv2
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                max_check = 10
                while frame_count < max_check:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                cap.release()
                if frame_count < 3:
                    logger.warning(f"Very few video frames extracted in first {max_check} frames ({frame_count}). Results may be unreliable.")

                # Check short audio segment
                import subprocess, os
                audio_path = os.path.join(temp_dir, 'temp_audio.wav')
                cmd = ['ffmpeg', '-y', '-i', video_path, '-ac', '1', '-ar', '16000', '-vn', '-t', '0.5', '-acodec', 'pcm_s16le', audio_path]
                try:
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    from scipy.io import wavfile
                    sr, audio = wavfile.read(audio_path)
                    if np.abs(audio).mean() < 1e-2:
                        logger.warning("Audio appears to be silent or very low energy in first 0.5s. Results may be unreliable.")
                except Exception:
                    logger.warning("Could not extract audio for quality check.")
                if os.path.exists(audio_path):
                    os.remove(audio_path)

            check_video_audio_quality_realtime(video_path, args.temp_dir, tuple(args.target_size))

        try:
            result = model.process_video_file(
                video_path=video_path,
                return_trace=args.trace,
                temp_dir=args.temp_dir,
                target_size=tuple(args.target_size),
                verbose=not args.realtime
            )
        except Exception as e:
            logger.error(f"Failed to process video file: {e}")
            if os.path.exists(args.temp_dir) and temp_cleanup_needed:
                logger.info(f"Cleaning up temp directory: {args.temp_dir}")
                shutil.rmtree(args.temp_dir, ignore_errors=True)
            return None

        # Check for empty or mismatched audio/video after extraction
        if result is None:
            logger.error("No result returned from model. Possible extraction failure.")
            if os.path.exists(args.temp_dir) and temp_cleanup_needed:
                logger.info(f"Cleaning up temp directory: {args.temp_dir}")
                shutil.rmtree(args.temp_dir, ignore_errors=True)
            return None

        if args.trace:
            offset, conf, trace = result
            logger.info("")
            logger.info(f"Final Offset: {offset:.2f} frames, Confidence: {conf:.3f}")
            logger.info("Trace (per window):")
            for i, (o, c, t) in enumerate(zip(trace['offsets'], trace['confidences'], trace['timestamps'])):
                logger.info(f"  Window {i}: Offset={o:.2f}, Confidence={c:.3f}, StartFrame={t}")
        else:
            offset, conf = result
            logger.info("")
            logger.info(f"Final Offset: {offset:.2f} frames, Confidence: {conf:.3f}")

        # Clean up temp directory unless --keep_temp is set
        if os.path.exists(args.temp_dir) and temp_cleanup_needed:
            if not args.realtime:
                # Print temp dir size before cleanup
                def get_dir_size(path):
                    total = 0
                    for dirpath, dirnames, filenames in os.walk(path):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            if os.path.isfile(fp):
                                total += os.path.getsize(fp)
                    return total
                size_mb = get_dir_size(args.temp_dir) / (1024*1024)
                logger.info(f"Cleaning up temp directory: {args.temp_dir} (size: {size_mb:.2f} MB)")
            shutil.rmtree(args.temp_dir, ignore_errors=True)
        return (offset, conf) if result is not None else None

    # Instantiate the model (once for all videos)
    model = StreamSyncFCN(
        window_size=args.window_size,
        stride=args.stride,
        buffer_size=args.buffer_size,
        use_attention=args.use_attention,
        pretrained_syncnet_path=args.pretrained,
        auto_load_pretrained=bool(args.pretrained)
    )

    # Batch/folder mode
    if args.folder:
        video_files = sorted(glob.glob(os.path.join(args.folder, '*.mp4')) +
                             glob.glob(os.path.join(args.folder, '*.avi')) +
                             glob.glob(os.path.join(args.folder, '*.mov')) +
                             glob.glob(os.path.join(args.folder, '*.mkv')))
        logger.info(f"Found {len(video_files)} video files in {args.folder}")
        results = []
        for video_path in video_files:
            logger.info(f"\nProcessing: {video_path}")
            res = process_one_video(video_path)
            if res is not None:
                offset, conf = res
                results.append({'video': os.path.basename(video_path), 'offset': offset, 'confidence': conf})
            else:
                results.append({'video': os.path.basename(video_path), 'offset': None, 'confidence': None})
        # Save results to CSV
        csv_path = os.path.join(args.folder, 'syncnet_fcn_results.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['video', 'offset', 'confidence'])
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        logger.info(f"\nBatch processing complete. Results saved to {csv_path}")

        # Print summary statistics if requested
        if args.summary:
            valid_offsets = [r['offset'] for r in results if r['offset'] is not None]
            valid_confs = [r['confidence'] for r in results if r['confidence'] is not None]
            if valid_offsets:
                import numpy as np
                logger.info(f"Summary: {len(valid_offsets)} valid results")
                logger.info(f"  Offset: mean={np.mean(valid_offsets):.2f}, std={np.std(valid_offsets):.2f}, min={np.min(valid_offsets):.2f}, max={np.max(valid_offsets):.2f}")
                logger.info(f"  Confidence: mean={np.mean(valid_confs):.3f}, std={np.std(valid_confs):.3f}, min={np.min(valid_confs):.3f}, max={np.max(valid_confs):.3f}")
            else:
                logger.warning("No valid results for summary statistics.")
        return

    # Single video mode
    if not args.video:
        logger.error("You must specify either --video or --folder.")
        return
    logger.info(f"\nProcessing: {args.video}")
    process_one_video(args.video)

if __name__ == "__main__":
    main()
