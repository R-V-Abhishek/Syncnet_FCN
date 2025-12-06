#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_model.py - Comprehensive Evaluation Script for FCN-SyncNet

This script evaluates the trained FCN-SyncNet model and generates metrics
suitable for documentation and README.

Usage:
    # Evaluate on validation set
    python evaluate_model.py --model checkpoints_regression/syncnet_fcn_best.pth --data_dir E:/voxceleb2_dataset/VoxCeleb2/dev --num_samples 500
    
    # Quick test on single video
    python evaluate_model.py --model checkpoints_regression/syncnet_fcn_best.pth --video data/example.avi
    
    # Generate full report
    python evaluate_model.py --model checkpoints_regression/syncnet_fcn_best.pth --data_dir E:/voxceleb2_dataset/VoxCeleb2/dev --full_report

Author: R V Abhishek
Date: 2025
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import sys
import json
import time
from datetime import datetime
import glob
import random
import cv2
import subprocess
from scipy.io import wavfile
import python_speech_features

# Import model
from SyncNetModel_FCN import StreamSyncFCN, SyncNetFCN


class ModelEvaluator:
    """Evaluator for FCN-SyncNet models."""
    
    def __init__(self, model_path, max_offset=125, use_attention=False, device=None):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model checkpoint
            max_offset: Maximum offset in frames (default: 125 = ±5 seconds at 25fps)
            use_attention: Whether model uses attention
            device: Device to use (default: auto-detect)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_offset = max_offset
        
        print(f"Device: {self.device}")
        print(f"Loading model from: {model_path}")
        
        # Load model
        self.model = StreamSyncFCN(
            max_offset=max_offset,
            use_attention=use_attention,
            pretrained_syncnet_path=None,
            auto_load_pretrained=False
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.checkpoint_info = {
                'epoch': checkpoint.get('epoch', 'unknown'),
                'metrics': checkpoint.get('metrics', {})
            }
        else:
            self.model.load_state_dict(checkpoint)
            self.checkpoint_info = {'epoch': 'unknown', 'metrics': {}}
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded (Epoch: {self.checkpoint_info['epoch']})")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def extract_audio_mfcc(self, video_path, temp_dir='temp_eval'):
        """Extract audio and compute MFCC."""
        os.makedirs(temp_dir, exist_ok=True)
        audio_path = os.path.join(temp_dir, 'temp_audio.wav')
        
        cmd = ['ffmpeg', '-y', '-i', video_path, '-ac', '1', '-ar', '16000',
               '-vn', '-acodec', 'pcm_s16le', audio_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        sample_rate, audio = wavfile.read(audio_path)
        
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        mfcc = python_speech_features.mfcc(audio, sample_rate, numcep=13)
        mfcc_tensor = torch.FloatTensor(mfcc.T).unsqueeze(0).unsqueeze(0)
        
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
    
    def evaluate_single_video(self, video_path, ground_truth_offset=0, verbose=True):
        """
        Evaluate a single video.
        
        Args:
            video_path: Path to video file
            ground_truth_offset: Known offset in frames (for computing error)
            verbose: Print progress
            
        Returns:
            dict with prediction and metrics
        """
        if verbose:
            print(f"Evaluating: {video_path}")
        
        try:
            # Extract features
            mfcc = self.extract_audio_mfcc(video_path)
            video = self.extract_video_frames(video_path)
            
            # Ensure minimum length
            min_frames = 25
            if video.shape[2] < min_frames:
                if verbose:
                    print(f"  Warning: Video too short ({video.shape[2]} frames)")
                return None
            
            # Crop to valid length
            audio_frames = mfcc.shape[3] // 4
            video_frames = video.shape[2]
            min_length = min(audio_frames, video_frames)
            
            video = video[:, :, :min_length, :, :]
            mfcc = mfcc[:, :, :, :min_length*4]
            
            # Run inference
            start_time = time.time()
            with torch.no_grad():
                mfcc = mfcc.to(self.device)
                video = video.to(self.device)
                
                predicted_offsets, audio_feat, video_feat = self.model(mfcc, video)
                
                # Get prediction
                pred_offset = predicted_offsets.mean().item()
            
            inference_time = time.time() - start_time
            
            # Compute error
            error = abs(pred_offset - ground_truth_offset)
            
            result = {
                'video': os.path.basename(video_path),
                'predicted_offset': pred_offset,
                'ground_truth_offset': ground_truth_offset,
                'absolute_error': error,
                'error_seconds': error / 25.0,  # Convert to seconds
                'inference_time': inference_time,
                'video_frames': min_length,
            }
            
            if verbose:
                print(f"  Predicted: {pred_offset:.2f} frames ({pred_offset/25:.3f}s)")
                print(f"  Ground Truth: {ground_truth_offset} frames")
                print(f"  Error: {error:.2f} frames ({error/25:.3f}s)")
                print(f"  Inference time: {inference_time*1000:.1f}ms")
            
            return result
            
        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
            return None
    
    def evaluate_dataset(self, data_dir, num_samples=100, offset_range=None, verbose=True):
        """
        Evaluate on a dataset with synthetic offsets.
        
        Args:
            data_dir: Path to dataset directory
            num_samples: Number of samples to evaluate
            offset_range: Tuple (min, max) for synthetic offsets (default: ±max_offset)
            verbose: Print progress
            
        Returns:
            dict with aggregate metrics
        """
        if offset_range is None:
            offset_range = (-self.max_offset, self.max_offset)
        
        # Find video files
        video_files = glob.glob(os.path.join(data_dir, '**', '*.mp4'), recursive=True)
        
        if len(video_files) == 0:
            print(f"No video files found in {data_dir}")
            return None
        
        print(f"Found {len(video_files)} videos")
        
        # Sample videos
        if len(video_files) > num_samples:
            video_files = random.sample(video_files, num_samples)
        
        print(f"Evaluating {len(video_files)} samples...")
        print("="*60)
        
        results = []
        errors = []
        inference_times = []
        
        for i, video_path in enumerate(video_files):
            # Generate random offset (simulating desync)
            ground_truth = random.randint(offset_range[0], offset_range[1])
            
            result = self.evaluate_single_video(
                video_path, 
                ground_truth_offset=ground_truth,
                verbose=(verbose and i % 10 == 0)
            )
            
            if result:
                results.append(result)
                errors.append(result['absolute_error'])
                inference_times.append(result['inference_time'])
            
            # Progress
            if (i + 1) % 50 == 0:
                print(f"Progress: {i+1}/{len(video_files)}")
        
        # Compute aggregate metrics
        errors = np.array(errors)
        inference_times = np.array(inference_times)
        
        metrics = {
            'num_samples': len(results),
            'mae_frames': float(np.mean(errors)),
            'mae_seconds': float(np.mean(errors) / 25.0),
            'rmse_frames': float(np.sqrt(np.mean(errors**2))),
            'std_frames': float(np.std(errors)),
            'median_error_frames': float(np.median(errors)),
            'max_error_frames': float(np.max(errors)),
            'accuracy_1_frame': float(np.mean(errors <= 1) * 100),
            'accuracy_3_frames': float(np.mean(errors <= 3) * 100),
            'accuracy_1_second': float(np.mean(errors <= 25) * 100),
            'avg_inference_time_ms': float(np.mean(inference_times) * 1000),
            'max_offset_range': offset_range,
        }
        
        return metrics, results
    
    def generate_report(self, metrics, output_path='evaluation_report.json'):
        """Generate evaluation report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'epoch': self.checkpoint_info.get('epoch'),
                'training_metrics': self.checkpoint_info.get('metrics', {}),
                'max_offset': self.max_offset,
            },
            'evaluation_metrics': metrics,
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport saved to: {output_path}")
        return report


def print_metrics_summary(metrics):
    """Print formatted metrics summary."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\n📊 Sample Statistics:")
    print(f"   Total samples evaluated: {metrics['num_samples']}")
    
    print(f"\n📏 Error Metrics:")
    print(f"   Mean Absolute Error (MAE): {metrics['mae_frames']:.2f} frames ({metrics['mae_seconds']:.4f} seconds)")
    print(f"   Root Mean Square Error (RMSE): {metrics['rmse_frames']:.2f} frames")
    print(f"   Standard Deviation: {metrics['std_frames']:.2f} frames")
    print(f"   Median Error: {metrics['median_error_frames']:.2f} frames")
    print(f"   Max Error: {metrics['max_error_frames']:.2f} frames")
    
    print(f"\n✅ Accuracy Metrics:")
    print(f"   Within ±1 frame: {metrics['accuracy_1_frame']:.2f}%")
    print(f"   Within ±3 frames: {metrics['accuracy_3_frames']:.2f}%")
    print(f"   Within ±1 second (25 frames): {metrics['accuracy_1_second']:.2f}%")
    
    print(f"\n⚡ Performance:")
    print(f"   Avg Inference Time: {metrics['avg_inference_time_ms']:.1f}ms per video")
    
    print("\n" + "="*60)


def print_readme_metrics(metrics):
    """Print metrics formatted for README.md."""
    print("\n" + "="*60)
    print("METRICS FOR README.md (Copy below)")
    print("="*60)
    
    print("""
## Model Performance

| Metric | Value |
|--------|-------|
| Mean Absolute Error (MAE) | {:.2f} frames ({:.4f}s) |
| Root Mean Square Error (RMSE) | {:.2f} frames |
| Accuracy (±1 frame) | {:.2f}% |
| Accuracy (±3 frames) | {:.2f}% |
| Accuracy (±1 second) | {:.2f}% |
| Average Inference Time | {:.1f}ms |

### Test Configuration
- **Test samples**: {} videos
- **Max offset range**: ±{} frames (±{:.1f} seconds)
- **Device**: CUDA/CPU
""".format(
        metrics['mae_frames'],
        metrics['mae_seconds'],
        metrics['rmse_frames'],
        metrics['accuracy_1_frame'],
        metrics['accuracy_3_frames'],
        metrics['accuracy_1_second'],
        metrics['avg_inference_time_ms'],
        metrics['num_samples'],
        metrics['max_offset_range'][1],
        metrics['max_offset_range'][1] / 25.0
    ))


def main():
    parser = argparse.ArgumentParser(description='Evaluate FCN-SyncNet Model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to dataset directory for batch evaluation')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to single video for quick test')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples for dataset evaluation (default: 100)')
    parser.add_argument('--max_offset', type=int, default=125,
                       help='Max offset in frames (default: 125)')
    parser.add_argument('--use_attention', action='store_true',
                       help='Use attention model')
    parser.add_argument('--full_report', action='store_true',
                       help='Generate full JSON report')
    parser.add_argument('--readme', action='store_true',
                       help='Print metrics formatted for README')
    parser.add_argument('--output', type=str, default='evaluation_report.json',
                       help='Output path for report')
    
    args = parser.parse_args()
    
    # Validate args
    if not args.video and not args.data_dir:
        parser.error("Please specify either --video or --data_dir")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=args.model,
        max_offset=args.max_offset,
        use_attention=args.use_attention
    )
    
    print("\n" + "="*60)
    
    # Single video evaluation
    if args.video:
        print("SINGLE VIDEO EVALUATION")
        print("="*60)
        result = evaluator.evaluate_single_video(args.video, verbose=True)
        
        if result:
            print("\n✓ Evaluation complete")
    
    # Dataset evaluation
    elif args.data_dir:
        print("DATASET EVALUATION")
        print("="*60)
        
        metrics, results = evaluator.evaluate_dataset(
            args.data_dir,
            num_samples=args.num_samples,
            verbose=True
        )
        
        if metrics:
            print_metrics_summary(metrics)
            
            if args.readme:
                print_readme_metrics(metrics)
            
            if args.full_report:
                evaluator.generate_report(metrics, args.output)
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
