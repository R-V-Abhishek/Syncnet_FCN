# Dataset Setup Guide

## Problem: VoxCeleb2 Dataset Not Found

The training script expects VoxCeleb2 dataset at:
```
E:/voxceleb2_dataset/VoxCeleb2/dev
```

But this path doesn't exist on your system.

## Solutions:

### Option 1: Download VoxCeleb2 (Recommended for Real Training)

VoxCeleb2 is a large dataset (~270GB). Download from:
- **Official**: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html
- Requires registration and agreement to terms

**Download Steps**:
1. Register at VoxCeleb website
2. Download `vox2_dev_mp4.zip` (development set)
3. Extract to `E:\voxceleb2_dataset\VoxCeleb2\dev`

### Option 2: Use Small Sample Dataset (Quick Testing)

Create a small test dataset with your own videos:

```powershell
# Create directory structure
New-Item -Path "E:\voxceleb_sample\dev\id00001\video001" -ItemType Directory -Force

# Copy some test videos (you need to provide these)
# Videos should have both audio and video, at least 1-2 seconds long
Copy-Item "path\to\your\video1.mp4" "E:\voxceleb_sample\dev\id00001\video001\00001.mp4"
Copy-Item "path\to\your\video2.mp4" "E:\voxceleb_sample\dev\id00001\video001\00002.mp4"
# Add more videos...

# Then train with:
.\\venv\\Scripts\\python.exe train_syncnet_fcn_improved.py `
    --data_dir E:/voxceleb_sample/dev `
    --pretrained_model data/syncnet_v2.model `
    --batch_size 2 `
    --epochs 5 `
    --lr 0.00001 `
    --output_dir checkpoints_test
```

### Option 3: Use Existing Videos in Your Project

If you have test videos in your project:

```powershell
# Check for existing videos
Get-ChildItem "c:\Users\admin\Syncnet_FCN" -Recurse -Filter "*.mp4"

# Create dataset from them
New-Item -Path "c:\Users\admin\Syncnet_FCN\sample_dataset\dev\speaker1\video1" -ItemType Directory -Force

# Copy your test videos there
# Then train with local path:
.\\venv\\Scripts\\python.exe train_syncnet_fcn_improved.py `
    --data_dir c:/Users/admin/Syncnet_FCN/sample_dataset/dev `
    --pretrained_model data/syncnet_v2.model `
    --batch_size 2 `
    --epochs 5
```

### Option 4: Download Small Sample from YouTube

Use `yt-dlp` to download sample videos:

```powershell
# Install yt-dlp
pip install yt-dlp

# Download a few talking head videos
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]" `
    -o "E:/voxceleb_sample/dev/speaker1/video1/%(autonumber)s.mp4" `
    --max-downloads 10 `
    "https://www.youtube.com/watch?v=SOME_VIDEO_ID"
```

## Minimum Dataset Requirements

For training to work, you need:
- **Minimum**: 50-100 videos (for testing the pipeline)
- **Recommended**: 1000+ videos (for meaningful training)
- **Full training**: VoxCeleb2 dev set (~150,000 videos)

Each video should:
- Have both audio and video tracks
- Be at least 1-2 seconds long
- Contain speech (talking heads work best)
- Be in MP4 format (or convertible by FFmpeg)

## Quick Test with Minimal Data

If you just want to test the training pipeline works:

```powershell
# Create minimal test dataset
$testDir = "c:\Users\admin\Syncnet_FCN\minimal_test\dev\speaker1\video1"
New-Item -Path $testDir -ItemType Directory -Force

# You need to manually add at least 10-20 MP4 videos to this directory
# Then run:
.\\venv\\Scripts\\python.exe train_syncnet_fcn_improved.py `
    --data_dir c:/Users/admin/Syncnet_FCN/minimal_test/dev `
    --pretrained_model data/syncnet_v2.model `
    --batch_size 2 `
    --epochs 2 `
    --lr 0.00001 `
    --output_dir checkpoints_minimal_test `
    --num_workers 0
```

## Current Status

- ✅ Model code is ready
- ✅ Training script is ready
- ❌ **Dataset is missing**
- ❌ Need to download or create dataset

**Next step**: Choose one of the options above to get training data.
