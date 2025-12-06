#!/usr/bin/python
#-*- coding: utf-8 -*-
"""
Fully Convolutional SyncNet Instance for Inference

This module provides inference capabilities for the FCN-SyncNet model,
including variable-length input processing and temporal sync prediction.

Key improvements over original:
1. Processes entire sequences at once (no fixed windows)
2. Returns frame-by-frame sync predictions
3. Better temporal smoothing
4. Confidence estimation per frame

Author: Enhanced version
Date: 2025-11-22
"""

import torch
import torch.nn.functional as F
import numpy as np
import time, os, math, glob, subprocess
import cv2
import python_speech_features

from scipy import signal
from scipy.io import wavfile
from SyncNetModel_FCN import SyncNetFCN, SyncNetFCN_WithAttention
from shutil import rmtree


class SyncNetInstance_FCN(torch.nn.Module):
    """
    SyncNet instance for fully convolutional inference.
    Supports variable-length inputs and dense temporal predictions.
    """
    
    def __init__(self, model_type='fcn', embedding_dim=512, max_offset=15, use_attention=False):
        super(SyncNetInstance_FCN, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.max_offset = max_offset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        if use_attention:
            self.model = SyncNetFCN_WithAttention(
                embedding_dim=embedding_dim,
                max_offset=max_offset
            ).to(self.device)
        else:
            self.model = SyncNetFCN(
                embedding_dim=embedding_dim,
                max_offset=max_offset
            ).to(self.device)
    
    def loadParameters(self, path):
        """Load model parameters from checkpoint."""
        loaded_state = torch.load(path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(loaded_state, dict):
            if 'model_state_dict' in loaded_state:
                state_dict = loaded_state['model_state_dict']
            elif 'state_dict' in loaded_state:
                state_dict = loaded_state['state_dict']
            else:
                state_dict = loaded_state
        else:
            state_dict = loaded_state.state_dict()
        
        # Load with strict=False to allow partial loading
        try:
            self.model.load_state_dict(state_dict, strict=True)
            print(f"Model loaded from {path}")
        except:
            print(f"Warning: Could not load all parameters from {path}")
            self.model.load_state_dict(state_dict, strict=False)
    
    def preprocess_audio(self, audio_path, target_length=None):
        """
        Load and preprocess audio file.
        
        Args:
            audio_path: Path to audio WAV file
            target_length: Optional target length in frames
            
        Returns:
            mfcc_tensor: [1, 1, 13, T] - MFCC features
            sample_rate: Audio sample rate
        """
        # Load audio
        sample_rate, audio = wavfile.read(audio_path)
        
        # Compute MFCC
        mfcc = python_speech_features.mfcc(audio, sample_rate)
        mfcc = mfcc.T  # [13, T]
        
        # Truncate or pad to target length
        if target_length is not None:
            if mfcc.shape[1] > target_length:
                mfcc = mfcc[:, :target_length]
            elif mfcc.shape[1] < target_length:
                pad_width = target_length - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='edge')
        
        # Add batch and channel dimensions
        mfcc = np.expand_dims(mfcc, axis=0)  # [1, 13, T]
        mfcc = np.expand_dims(mfcc, axis=0)  # [1, 1, 13, T]
        
        # Convert to tensor
        mfcc_tensor = torch.FloatTensor(mfcc)
        
        return mfcc_tensor, sample_rate
    
    def preprocess_video(self, video_path, target_length=None):
        """
        Load and preprocess video file.
        
        Args:
            video_path: Path to video file or directory of frames
            target_length: Optional target length in frames
            
        Returns:
            video_tensor: [1, 3, T, H, W] - video frames
        """
        # Load video frames
        if os.path.isdir(video_path):
            # Load from directory
            flist = sorted(glob.glob(os.path.join(video_path, '*.jpg')))
            images = [cv2.imread(f) for f in flist]
        else:
            # Load from video file
            cap = cv2.VideoCapture(video_path)
            images = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                images.append(frame)
            cap.release()
        
        if len(images) == 0:
            raise ValueError(f"No frames found in {video_path}")
        
        # Truncate or pad to target length
        if target_length is not None:
            if len(images) > target_length:
                images = images[:target_length]
            elif len(images) < target_length:
                # Pad by repeating last frame
                last_frame = images[-1]
                images.extend([last_frame] * (target_length - len(images)))
        
        # Stack and normalize
        im = np.stack(images, axis=0)  # [T, H, W, 3]
        im = im.astype(float) / 255.0  # Normalize to [0, 1]
        
        # Rearrange to [1, 3, T, H, W]
        im = np.transpose(im, (3, 0, 1, 2))  # [3, T, H, W]
        im = np.expand_dims(im, axis=0)  # [1, 3, T, H, W]
        
        # Convert to tensor
        video_tensor = torch.FloatTensor(im)
        
        return video_tensor
    
    def evaluate(self, opt, videofile):
        """
        Evaluate sync for a video file.
        Returns frame-by-frame sync predictions.
        
        Args:
            opt: Options object with configuration
            videofile: Path to video file
            
        Returns:
            offsets: [T] - predicted offset for each frame
            confidences: [T] - confidence for each frame
            sync_probs: [2K+1, T] - full probability distribution
        """
        self.model.eval()
        
        # Create temporary directory
        if os.path.exists(os.path.join(opt.tmp_dir, opt.reference)):
            rmtree(os.path.join(opt.tmp_dir, opt.reference))
        os.makedirs(os.path.join(opt.tmp_dir, opt.reference))
        
        # Extract frames and audio
        print("Extracting frames and audio...")
        frames_path = os.path.join(opt.tmp_dir, opt.reference)
        audio_path = os.path.join(opt.tmp_dir, opt.reference, 'audio.wav')
        
        # Extract frames
        command = (f"ffmpeg -y -i {videofile} -threads 1 -f image2 "
                  f"{os.path.join(frames_path, '%06d.jpg')}")
        subprocess.call(command, shell=True, stdout=subprocess.DEVNULL, 
                       stderr=subprocess.DEVNULL)
        
        # Extract audio
        command = (f"ffmpeg -y -i {videofile} -async 1 -ac 1 -vn "
                  f"-acodec pcm_s16le -ar 16000 {audio_path}")
        subprocess.call(command, shell=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        
        # Preprocess audio and video
        print("Loading and preprocessing data...")
        audio_tensor, sample_rate = self.preprocess_audio(audio_path)
        video_tensor = self.preprocess_video(frames_path)
        
        # Check length consistency
        audio_duration = audio_tensor.shape[3] / 100.0  # MFCC is 100 fps
        video_duration = video_tensor.shape[2] / 25.0   # Video is 25 fps
        
        if abs(audio_duration - video_duration) > 0.1:
            print(f"WARNING: Audio ({audio_duration:.2f}s) and video "
                  f"({video_duration:.2f}s) lengths differ")
        
        # Align lengths (use shorter)
        min_length = min(
            video_tensor.shape[2],  # video frames
            audio_tensor.shape[3] // 4  # audio frames (4:1 ratio)
        )
        
        video_tensor = video_tensor[:, :, :min_length, :, :]
        audio_tensor = audio_tensor[:, :, :, :min_length*4]
        
        print(f"Processing {min_length} frames...")
        
        # Forward pass
        tS = time.time()
        with torch.no_grad():
            sync_probs, audio_feat, video_feat = self.model(
                audio_tensor.to(self.device),
                video_tensor.to(self.device)
            )
        
        print(f'Compute time: {time.time()-tS:.3f} sec')
        
        # Compute offsets and confidences
        offsets, confidences = self.model.compute_offset(sync_probs)
        
        # Convert to numpy
        offsets = offsets.cpu().numpy()[0]  # [T]
        confidences = confidences.cpu().numpy()[0]  # [T]
        sync_probs = sync_probs.cpu().numpy()[0]  # [2K+1, T]
        
        # Apply temporal smoothing to confidences
        confidences_smooth = signal.medfilt(confidences, kernel_size=9)
        
        # Compute overall statistics
        median_offset = np.median(offsets)
        mean_confidence = np.mean(confidences_smooth)
        
        # Find consensus offset (mode)
        offset_hist, offset_bins = np.histogram(offsets, bins=2*self.max_offset+1)
        consensus_offset = offset_bins[np.argmax(offset_hist)]
        
        # Print results
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print('\nFrame-wise confidence (smoothed):')
        print(confidences_smooth)
        print(f'\nConsensus offset: \t{consensus_offset:.1f} frames')
        print(f'Median offset: \t\t{median_offset:.1f} frames')
        print(f'Mean confidence: \t{mean_confidence:.3f}')
        
        return offsets, confidences_smooth, sync_probs
    
    def evaluate_batch(self, opt, videofile, chunk_size=100, overlap=10):
        """
        Evaluate long videos in chunks with overlap for consistency.
        
        Args:
            opt: Options object
            videofile: Path to video file
            chunk_size: Number of frames per chunk
            overlap: Number of overlapping frames between chunks
            
        Returns:
            offsets: [T] - predicted offset for each frame
            confidences: [T] - confidence for each frame
        """
        self.model.eval()
        
        # Create temporary directory
        if os.path.exists(os.path.join(opt.tmp_dir, opt.reference)):
            rmtree(os.path.join(opt.tmp_dir, opt.reference))
        os.makedirs(os.path.join(opt.tmp_dir, opt.reference))
        
        # Extract frames and audio
        frames_path = os.path.join(opt.tmp_dir, opt.reference)
        audio_path = os.path.join(opt.tmp_dir, opt.reference, 'audio.wav')
        
        # Extract frames
        command = (f"ffmpeg -y -i {videofile} -threads 1 -f image2 "
                  f"{os.path.join(frames_path, '%06d.jpg')}")
        subprocess.call(command, shell=True, stdout=subprocess.DEVNULL, 
                       stderr=subprocess.DEVNULL)
        
        # Extract audio
        command = (f"ffmpeg -y -i {videofile} -async 1 -ac 1 -vn "
                  f"-acodec pcm_s16le -ar 16000 {audio_path}")
        subprocess.call(command, shell=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        
        # Preprocess audio and video
        audio_tensor, sample_rate = self.preprocess_audio(audio_path)
        video_tensor = self.preprocess_video(frames_path)
        
        # Process in chunks
        all_offsets = []
        all_confidences = []
        
        stride = chunk_size - overlap
        num_chunks = (video_tensor.shape[2] - overlap) // stride + 1
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * stride
            end_idx = min(start_idx + chunk_size, video_tensor.shape[2])
            
            # Extract chunk
            video_chunk = video_tensor[:, :, start_idx:end_idx, :, :]
            audio_chunk = audio_tensor[:, :, :, start_idx*4:end_idx*4]
            
            # Forward pass
            with torch.no_grad():
                sync_probs, _, _ = self.model(
                    audio_chunk.to(self.device),
                    video_chunk.to(self.device)
                )
            
            # Compute offsets
            offsets, confidences = self.model.compute_offset(sync_probs)
            
            # Handle overlap (average predictions)
            if chunk_idx > 0:
                # Average overlapping region
                overlap_frames = overlap
                all_offsets[-overlap_frames:] = (
                    all_offsets[-overlap_frames:] + 
                    offsets[:overlap_frames].cpu().numpy()[0]
                ) / 2
                all_confidences[-overlap_frames:] = (
                    all_confidences[-overlap_frames:] + 
                    confidences[:overlap_frames].cpu().numpy()[0]
                ) / 2
                
                # Append non-overlapping part
                all_offsets.extend(offsets[overlap_frames:].cpu().numpy()[0])
                all_confidences.extend(confidences[overlap_frames:].cpu().numpy()[0])
            else:
                all_offsets.extend(offsets.cpu().numpy()[0])
                all_confidences.extend(confidences.cpu().numpy()[0])
        
        offsets = np.array(all_offsets)
        confidences = np.array(all_confidences)
        
        return offsets, confidences
    
    def extract_features(self, opt, videofile, feature_type='both'):
        """
        Extract audio and/or video features for downstream tasks.
        
        Args:
            opt: Options object
            videofile: Path to video file
            feature_type: 'audio', 'video', or 'both'
            
        Returns:
            features: Dictionary with audio_features and/or video_features
        """
        self.model.eval()
        
        # Preprocess
        if feature_type in ['audio', 'both']:
            audio_path = os.path.join(opt.tmp_dir, opt.reference, 'audio.wav')
            audio_tensor, _ = self.preprocess_audio(audio_path)
        
        if feature_type in ['video', 'both']:
            frames_path = os.path.join(opt.tmp_dir, opt.reference)
            video_tensor = self.preprocess_video(frames_path)
        
        features = {}
        
        # Extract features
        with torch.no_grad():
            if feature_type in ['audio', 'both']:
                audio_features = self.model.forward_audio(audio_tensor.to(self.device))
                features['audio'] = audio_features.cpu().numpy()
            
            if feature_type in ['video', 'both']:
                video_features = self.model.forward_video(video_tensor.to(self.device))
                features['video'] = video_features.cpu().numpy()
        
        return features


# ==================== UTILITY FUNCTIONS ====================

def visualize_sync_predictions(offsets, confidences, save_path=None):
    """
    Visualize sync predictions over time.
    
    Args:
        offsets: [T] - predicted offsets
        confidences: [T] - confidence scores
        save_path: Optional path to save plot
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot offsets
        ax1.plot(offsets, linewidth=2)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Offset (frames)')
        ax1.set_title('Audio-Visual Sync Offset Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Plot confidences
        ax2.plot(confidences, linewidth=2, color='green')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Confidence')
        ax2.set_title('Sync Detection Confidence Over Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
            
    except ImportError:
        print("matplotlib not installed. Skipping visualization.")


if __name__ == "__main__":
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='FCN SyncNet Inference')
    parser.add_argument('--videofile', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--model_path', type=str, default='data/syncnet_v2.model',
                       help='Path to model checkpoint')
    parser.add_argument('--tmp_dir', type=str, default='data/tmp',
                       help='Temporary directory for processing')
    parser.add_argument('--reference', type=str, default='test',
                       help='Reference name for this video')
    parser.add_argument('--use_attention', action='store_true',
                       help='Use attention-based model')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize results')
    parser.add_argument('--max_offset', type=int, default=15,
                       help='Maximum offset to consider (frames)')
    
    opt = parser.parse_args()
    
    # Create instance
    print("Initializing FCN SyncNet...")
    syncnet = SyncNetInstance_FCN(
        use_attention=opt.use_attention,
        max_offset=opt.max_offset
    )
    
    # Load model (if available)
    if os.path.exists(opt.model_path):
        print(f"Loading model from {opt.model_path}")
        try:
            syncnet.loadParameters(opt.model_path)
        except:
            print("Warning: Could not load pretrained weights. Using random initialization.")
    
    # Evaluate
    print(f"\nEvaluating video: {opt.videofile}")
    offsets, confidences, sync_probs = syncnet.evaluate(opt, opt.videofile)
    
    # Visualize
    if opt.visualize:
        viz_path = opt.videofile.replace('.mp4', '_sync_analysis.png')
        viz_path = viz_path.replace('.avi', '_sync_analysis.png')
        visualize_sync_predictions(offsets, confidences, save_path=viz_path)
    
    print("\nDone!")
