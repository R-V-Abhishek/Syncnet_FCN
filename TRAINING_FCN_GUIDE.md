# Training SyncNet FCN Model - Complete Guide

**Purpose:** Train the improved Fully Convolutional SyncNet model on VoxCeleb dataset  
**Prerequisites:** Tested pretrained model, decided improvement is needed

---

## 📋 TABLE OF CONTENTS

1. [Dataset Overview](#1-dataset-overview)
2. [Download VoxCeleb](#2-download-voxceleb)
3. [Data Preprocessing](#3-data-preprocessing)
4. [Training Data Format](#4-training-data-format)
5. [Training the FCN Model](#5-training-the-fcn-model)
6. [Evaluation](#6-evaluation)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. DATASET OVERVIEW

### VoxCeleb Datasets

| Dataset | Speakers | Videos | Size | Use |
|---------|----------|--------|------|-----|
| **VoxCeleb1** | 1,251 | 22,496 | ~30 GB | Testing/Small training |
| **VoxCeleb2** | 6,112 | 150,480 | ~300 GB | Full training |
| **VoxCeleb1-Test** | 40 | 4,874 | ~5 GB | Evaluation |

### What's in the Dataset?
- YouTube interview videos of celebrities
- Audio-visual data with faces and speech
- Pre-cropped face tracks
- Naturally synchronized (ground truth = 0 offset)

### Recommendation
- **Start with VoxCeleb1** (smaller, faster to download)
- **Scale to VoxCeleb2** if you need more data

---

## 2. DOWNLOAD VOXCELEB

### 2.1 Register for Access

1. Go to: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/
2. Click "Download" 
3. Fill the form (use academic email)
4. You'll receive credentials via email

### 2.2 Download Options

#### Option A: Direct Download (if links work)
```powershell
# Create dataset directory
mkdir D:\Datasets\VoxCeleb
cd D:\Datasets\VoxCeleb

# VoxCeleb1 dev set (video)
# Use browser or download manager - files are large
```

#### Option B: Using Official Scripts
```powershell
# Clone VoxCeleb tools
git clone https://github.com/clovaai/voxceleb_trainer.git
cd voxceleb_trainer

# Install dependencies
pip install soundfile

# Download (requires credentials)
python download.py --save_path D:\Datasets\VoxCeleb --download --user USERNAME --password PASSWORD
```

#### Option C: Using Academic Torrent (Fastest)
- Search "VoxCeleb" on https://academictorrents.com/
- Usually faster than direct HTTP download

### 2.3 Dataset Structure After Download

```
D:\Datasets\VoxCeleb\
├── voxceleb1/
│   ├── dev/
│   │   └── aac/           # Audio files (.m4a)
│   │       ├── id10001/
│   │       │   ├── video1/
│   │       │   │   ├── 00001.m4a
│   │       │   │   └── ...
│   │       │   └── video2/
│   │       └── id10002/
│   └── test/
│       └── aac/
├── voxceleb2/
│   ├── dev/
│   │   └── aac/
│   └── test/
│       └── aac/
└── voxceleb1_videos/      # Video files (if downloaded separately)
    └── dev/
        └── mp4/
            ├── id10001/
            │   └── video1.mp4
            └── ...
```

### 2.4 Download Checklist

- [ ] Register at VoxCeleb website
- [ ] Receive credentials via email
- [ ] Download VoxCeleb1 dev set (audio) - ~10 GB
- [ ] Download VoxCeleb1 video files - ~20 GB
- [ ] Verify file counts match expected
- [ ] (Optional) Download VoxCeleb2 for more data

---

## 3. DATA PREPROCESSING

### 3.1 Overview

```
Raw Videos → Face Detection → Face Cropping → Extract MFCC → Training Pairs
```

### 3.2 Preprocessing Script

Create `preprocess_voxceleb.py`:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess VoxCeleb dataset for SyncNet FCN training
"""

import os
import cv2
import glob
import argparse
import subprocess
import numpy as np
import python_speech_features
from scipy.io import wavfile
from tqdm import tqdm
import json

def extract_audio(video_path, audio_path):
    """Extract audio from video using FFmpeg"""
    command = f'ffmpeg -y -i "{video_path}" -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 "{audio_path}"'
    subprocess.call(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return os.path.exists(audio_path)

def extract_frames(video_path, frames_dir, fps=25):
    """Extract frames from video"""
    os.makedirs(frames_dir, exist_ok=True)
    command = f'ffmpeg -y -i "{video_path}" -vf fps={fps} "{frames_dir}/%06d.jpg"'
    subprocess.call(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return len(glob.glob(os.path.join(frames_dir, '*.jpg')))

def compute_mfcc(audio_path):
    """Compute MFCC features from audio"""
    try:
        sample_rate, audio = wavfile.read(audio_path)
        mfcc = python_speech_features.mfcc(audio, sample_rate)
        return mfcc
    except Exception as e:
        print(f"Error computing MFCC: {e}")
        return None

def process_video(video_path, output_dir, min_frames=50):
    """Process a single video file"""
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_id)
    
    # Skip if already processed
    if os.path.exists(os.path.join(video_output_dir, 'info.json')):
        return True
    
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Extract audio
    audio_path = os.path.join(video_output_dir, 'audio.wav')
    if not extract_audio(video_path, audio_path):
        return False
    
    # Extract frames
    frames_dir = os.path.join(video_output_dir, 'frames')
    num_frames = extract_frames(video_path, frames_dir)
    
    if num_frames < min_frames:
        return False
    
    # Compute MFCC
    mfcc = compute_mfcc(audio_path)
    if mfcc is None:
        return False
    
    # Save MFCC
    np.save(os.path.join(video_output_dir, 'mfcc.npy'), mfcc)
    
    # Save info
    info = {
        'video_path': video_path,
        'num_frames': num_frames,
        'mfcc_shape': list(mfcc.shape),
        'fps': 25,
        'sample_rate': 16000
    }
    with open(os.path.join(video_output_dir, 'info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Preprocess VoxCeleb for SyncNet')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to VoxCeleb videos')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    args = parser.parse_args()
    
    # Find all video files
    video_files = glob.glob(os.path.join(args.input_dir, '**', '*.mp4'), recursive=True)
    video_files += glob.glob(os.path.join(args.input_dir, '**', '*.avi'), recursive=True)
    
    print(f"Found {len(video_files)} videos")
    
    # Process videos
    success_count = 0
    for video_path in tqdm(video_files, desc="Processing"):
        if process_video(video_path, args.output_dir):
            success_count += 1
    
    print(f"Successfully processed {success_count}/{len(video_files)} videos")

if __name__ == '__main__':
    main()
```

### 3.3 Run Preprocessing

```powershell
# Activate your venv first!
.\venv\Scripts\Activate.ps1

# Run preprocessing
python preprocess_voxceleb.py --input_dir D:\Datasets\VoxCeleb\voxceleb1_videos --output_dir D:\Datasets\VoxCeleb_Processed
```

### 3.4 Preprocessing Output Structure

```
D:\Datasets\VoxCeleb_Processed\
├── video_001/
│   ├── frames/
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   ├── audio.wav
│   ├── mfcc.npy
│   └── info.json
├── video_002/
│   └── ...
└── ...
```

---

## 4. TRAINING DATA FORMAT

### 4.1 Input Format for SyncNet FCN

| Input | Shape | Description |
|-------|-------|-------------|
| **Video** | `[B, 3, T, H, W]` | RGB frames, T=variable length |
| **Audio** | `[B, 1, 13, M]` | MFCC features, M=T*4 |
| **Label** | `[B]` | Offset in frames (0=synced) |

Where:
- `B` = Batch size
- `T` = Number of video frames
- `H, W` = Height, Width (typically 224x224 or 112x112)
- `M` = Number of MFCC frames (4 per video frame at 25fps video, 100fps MFCC)

### 4.2 Creating Training Pairs

Create `create_training_pairs.py`:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create training pairs with positive (synced) and negative (offset) samples
"""

import os
import json
import random
import argparse
from tqdm import tqdm

def create_pairs(processed_dir, output_file, max_offset=15):
    """
    Create training pairs:
    - Positive: original synced pairs (label=0)
    - Negative: shifted pairs (label=offset)
    """
    
    # Find all processed videos
    videos = []
    for video_dir in os.listdir(processed_dir):
        info_path = os.path.join(processed_dir, video_dir, 'info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
            info['video_dir'] = os.path.join(processed_dir, video_dir)
            videos.append(info)
    
    print(f"Found {len(videos)} processed videos")
    
    pairs = []
    
    for video in tqdm(videos, desc="Creating pairs"):
        num_frames = video['num_frames']
        video_dir = video['video_dir']
        
        # Skip short videos
        if num_frames < 50:
            continue
        
        # Create synced pair (positive)
        pairs.append({
            'video_dir': video_dir,
            'offset': 0,
            'label': 'synced'
        })
        
        # Create offset pairs (negatives)
        for offset in [-10, -5, 5, 10]:
            if abs(offset) < num_frames - 20:  # Ensure enough frames
                pairs.append({
                    'video_dir': video_dir,
                    'offset': offset,
                    'label': 'offset'
                })
    
    # Shuffle pairs
    random.shuffle(pairs)
    
    # Split into train/val
    split_idx = int(len(pairs) * 0.9)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    
    # Save
    with open(output_file.replace('.json', '_train.json'), 'w') as f:
        json.dump(train_pairs, f, indent=2)
    
    with open(output_file.replace('.json', '_val.json'), 'w') as f:
        json.dump(val_pairs, f, indent=2)
    
    print(f"Created {len(train_pairs)} training pairs, {len(val_pairs)} validation pairs")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, default='training_pairs.json')
    args = parser.parse_args()
    
    create_pairs(args.processed_dir, args.output_file)
```

### 4.3 Run Pair Creation

```powershell
python create_training_pairs.py --processed_dir D:\Datasets\VoxCeleb_Processed --output_file training_pairs.json
```

---

## 5. TRAINING THE FCN MODEL

### 5.1 Training Script

Create `train_syncnet_fcn.py`:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train SyncNet FCN model
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2

from SyncNetModel_FCN import SyncNetFCN  # Import your FCN model

class SyncNetDataset(Dataset):
    """Dataset for SyncNet training"""
    
    def __init__(self, pairs_file, frame_size=(112, 112), num_frames=25):
        with open(pairs_file, 'r') as f:
            self.pairs = json.load(f)
        self.frame_size = frame_size
        self.num_frames = num_frames
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        video_dir = pair['video_dir']
        offset = pair['offset']
        
        # Load frames
        frames_dir = os.path.join(video_dir, 'frames')
        frame_files = sorted(os.listdir(frames_dir))
        
        # Select frame range
        start_frame = max(0, -offset)
        end_frame = min(len(frame_files), len(frame_files) - offset)
        
        # Sample frames
        if end_frame - start_frame < self.num_frames:
            return self.__getitem__((idx + 1) % len(self))
        
        frame_indices = np.linspace(start_frame, end_frame - 1, self.num_frames, dtype=int)
        
        frames = []
        for i in frame_indices:
            frame_path = os.path.join(frames_dir, frame_files[i])
            frame = cv2.imread(frame_path)
            frame = cv2.resize(frame, self.frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        frames = np.stack(frames, axis=0)  # [T, H, W, C]
        frames = frames.transpose(3, 0, 1, 2)  # [C, T, H, W]
        frames = frames.astype(np.float32) / 255.0
        
        # Load MFCC
        mfcc = np.load(os.path.join(video_dir, 'mfcc.npy'))
        
        # Align MFCC with offset
        mfcc_start = max(0, offset * 4)
        mfcc_end = mfcc_start + self.num_frames * 4
        
        if mfcc_end > len(mfcc):
            return self.__getitem__((idx + 1) % len(self))
        
        mfcc = mfcc[mfcc_start:mfcc_end, :]
        mfcc = mfcc.T  # [13, M]
        mfcc = np.expand_dims(mfcc, 0)  # [1, 13, M]
        
        # Label: 1 if synced, 0 if offset
        label = 1.0 if offset == 0 else 0.0
        
        return {
            'video': torch.FloatTensor(frames),
            'audio': torch.FloatTensor(mfcc),
            'label': torch.FloatTensor([label]),
            'offset': offset
        }

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        video = batch['video'].to(device)
        audio = batch['audio'].to(device)
        label = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        video_feat = model.forward_video(video)
        audio_feat = model.forward_audio(audio)
        
        # Compute similarity (cosine)
        similarity = nn.functional.cosine_similarity(video_feat, audio_feat, dim=1)
        
        # Binary cross entropy loss
        loss = criterion(similarity.unsqueeze(1), label)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Accuracy
        predicted = (similarity > 0.5).float()
        correct += (predicted == label.squeeze()).sum().item()
        total += label.size(0)
    
    return total_loss / len(dataloader), correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            label = batch['label'].to(device)
            
            video_feat = model.forward_video(video)
            audio_feat = model.forward_audio(audio)
            
            similarity = nn.functional.cosine_similarity(video_feat, audio_feat, dim=1)
            loss = criterion(similarity.unsqueeze(1), label)
            
            total_loss += loss.item()
            
            predicted = (similarity > 0.5).float()
            correct += (predicted == label.squeeze()).sum().item()
            total += label.size(0)
    
    return total_loss / len(dataloader), correct / total

def main():
    parser = argparse.ArgumentParser(description='Train SyncNet FCN')
    parser.add_argument('--train_pairs', type=str, required=True)
    parser.add_argument('--val_pairs', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset and DataLoader
    train_dataset = SyncNetDataset(args.train_pairs)
    val_dataset = SyncNetDataset(args.val_pairs)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model
    model = SyncNetFCN().to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    best_val_acc = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            print(f"Saved best model with val_acc: {val_acc:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")

if __name__ == '__main__':
    main()
```

### 5.2 Run Training

```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# Run training
python train_syncnet_fcn.py `
    --train_pairs training_pairs_train.json `
    --val_pairs training_pairs_val.json `
    --output_dir checkpoints `
    --batch_size 16 `
    --epochs 50 `
    --lr 0.0001
```

### 5.3 Training Parameters Guide

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 16 | Reduce if GPU OOM |
| `epochs` | 50 | Increase for better convergence |
| `lr` | 1e-4 | Learning rate |
| `device` | cuda | Use 'cpu' if no GPU |

### 5.4 Monitoring Training

```powershell
# View training progress
type checkpoints\training_log.txt

# Check GPU usage
nvidia-smi -l 1
```

---

## 6. EVALUATION

### 6.1 Test the Trained Model

Create `evaluate_fcn.py`:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate trained SyncNet FCN model
"""

import os
import torch
import argparse
from SyncNetModel_FCN import SyncNetFCN
from SyncNetInstance_FCN import SyncNetFCNInstance

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--tmp_dir', type=str, default='data/work/pytmp')
    args = parser.parse_args()
    
    # Load model
    model = SyncNetFCNInstance()
    model.loadParameters(args.model_path)
    
    # Run evaluation
    offset, conf, dists = model.evaluate(args, videofile=args.video)
    
    print(f"\n{'='*40}")
    print(f"Video: {args.video}")
    print(f"Detected Offset: {offset} frames")
    print(f"Confidence: {conf:.3f}")
    print(f"{'='*40}")

if __name__ == '__main__':
    main()
```

### 6.2 Run Evaluation

```powershell
python evaluate_fcn.py --model_path checkpoints/best_model.pth --video test_videos/my_test.mp4
```

### 6.3 Compare with Original Model

```powershell
# Original SyncNet
python demo_syncnet.py --videofile test_videos/my_test.mp4 --tmp_dir data/work/pytmp

# Your trained FCN
python evaluate_fcn.py --model_path checkpoints/best_model.pth --video test_videos/my_test.mp4
```

---

## 7. TROUBLESHOOTING

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| CUDA OOM | Batch too large | Reduce `--batch_size` to 8 or 4 |
| Slow training | CPU mode | Install CUDA PyTorch |
| NaN loss | Learning rate too high | Reduce `--lr` to 1e-5 |
| Low accuracy | Not enough data | Use more VoxCeleb data |
| Overfitting | Not enough augmentation | Add data augmentation |

### GPU Memory Requirements

| Batch Size | GPU Memory Needed |
|------------|-------------------|
| 4 | ~4 GB |
| 8 | ~6 GB |
| 16 | ~10 GB |
| 32 | ~16 GB |

### Training Time Estimates

| Dataset Size | GPU | Time per Epoch |
|--------------|-----|----------------|
| 1,000 videos | GTX 1080 | ~30 min |
| 5,000 videos | GTX 1080 | ~2.5 hours |
| 20,000 videos | RTX 3080 | ~4 hours |

---

## 📋 COMPLETE TRAINING CHECKLIST

### Phase 1: Download Data
- [ ] Register at VoxCeleb website
- [ ] Receive credentials
- [ ] Download VoxCeleb1 videos (~30 GB)
- [ ] Verify download complete

### Phase 2: Preprocess
- [ ] Create `preprocess_voxceleb.py`
- [ ] Run preprocessing
- [ ] Verify output structure
- [ ] Check processed video count

### Phase 3: Create Pairs
- [ ] Create `create_training_pairs.py`
- [ ] Run pair creation
- [ ] Verify train/val split

### Phase 4: Train
- [ ] Create `train_syncnet_fcn.py`
- [ ] Start training
- [ ] Monitor loss/accuracy
- [ ] Save best model

### Phase 5: Evaluate
- [ ] Test on held-out videos
- [ ] Compare with original SyncNet
- [ ] Document improvements

---

## 📊 EXPECTED RESULTS

After training on VoxCeleb1:

| Metric | Original SyncNet | FCN Model (Expected) |
|--------|------------------|---------------------|
| Sync Accuracy | ~85% | ~88-92% |
| Offset Detection (±1 frame) | ~75% | ~80-85% |
| Inference Speed | 1x | 1.2x faster |

---

*Guide created: November 27, 2025*
