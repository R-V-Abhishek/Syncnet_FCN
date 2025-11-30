# Quick Setup and Run Guide for VoxCeleb2 Training

## Prerequisites on the Other PC

### 1. Install Python Packages
```bash
pip install torch torchvision
pip install opencv-python
pip install scipy
pip install python_speech_features
pip install numpy
```

### 2. Install FFmpeg
- Download FFmpeg: https://ffmpeg.org/download.html
- Add FFmpeg to system PATH
- Test: `ffmpeg -version`

### 3. Extract VoxCeleb2 Dataset

You have these files:
- `vox2_dev_aac_partaa`
- `vox2_dev_mp4_partaa`

**To extract:**

1. Check if you have ALL parts (partaa, partab, partac, etc.)
2. Use 7-Zip or WinRAR:
   - Right-click on the FIRST part (`vox2_dev_mp4_partaa`)
   - Choose "Extract Here" or "Extract to..."
   - It will automatically combine all parts

**After extraction, you should have:**
```
E:\voxceleb2 dataset\VoxCeleb2\
└── dev\
    ├── id00012\
    ├── id00013\
    └── ... (many speaker IDs)
        └── (video folders with .mp4 files)
```

## Copy Files to Other PC

Copy these files to your other PC:
1. `SyncNetModel_FCN.py`
2. `SyncNetModel.py` (if you have it)
3. `train_syncnet_fcn_complete.py` (the new complete script)
4. `data/syncnet_v2.model` (pretrained weights)
5. Any detector files in the `detectors/` folder

## Run Training

### Basic Command
```bash
python train_syncnet_fcn_complete.py --data_dir "E:\voxceleb2 dataset\VoxCeleb2\dev" --pretrained_model data\syncnet_v2.model
```

### With Custom Settings
```bash
python train_syncnet_fcn_complete.py --data_dir "E:\voxceleb2 dataset\VoxCeleb2\dev" --pretrained_model data\syncnet_v2.model --batch_size 4 --epochs 20 --lr 0.0001 --output_dir my_checkpoints
```

### All Available Options
```bash
python train_syncnet_fcn_complete.py \
    --data_dir "E:\voxceleb2 dataset\VoxCeleb2\dev"     # REQUIRED: Dataset path
    --pretrained_model data\syncnet_v2.model            # Pretrained weights
    --batch_size 4                                      # Batch size (lower if GPU memory issues)
    --epochs 10                                         # Number of epochs
    --lr 0.001                                          # Learning rate
    --output_dir checkpoints                            # Where to save checkpoints
    --use_attention                                     # Enable attention (optional)
    --num_workers 2                                     # DataLoader workers
```

## Important Notes

### GPU vs CPU
- Training will automatically use GPU if available
- On CPU, training will be VERY slow
- Recommended: Use a machine with NVIDIA GPU

### Batch Size
- If you get "Out of Memory" errors, reduce batch size:
  - Try `--batch_size 2` or `--batch_size 1`
- Larger batch = faster but needs more GPU memory

### First Run Test
Start with a small test to verify everything works:
```bash
# Test on just a few videos first
python train_syncnet_fcn_complete.py --data_dir "E:\voxceleb2 dataset\VoxCeleb2\dev" --pretrained_model data\syncnet_v2.model --epochs 1 --batch_size 2
```

## Expected Output

You should see:
```
Using device: cuda
Creating model...
✓ Loaded 245 pretrained conv parameters
✓ Froze pretrained conv layers
Loading dataset...
Found 1234 videos in dataset
Trainable parameters: 2,345,678
Frozen parameters: 12,345,678

Starting training...
================================================================================

Epoch 1/10
--------------------------------------------------------------------------------
  Batch 0/308, Loss: 0.6931, Acc: 50.00%
  Batch 10/308, Loss: 0.6234, Acc: 62.50%
  ...
```

## Troubleshooting

### "FFmpeg not found"
- Install FFmpeg and add to PATH
- Restart terminal

### "No videos found"
- Check dataset path is correct
- Make sure you extracted the archives
- Verify .mp4 files exist in subdirectories

### "Out of memory"
- Reduce batch_size: `--batch_size 1`
- Close other programs
- Use a machine with more GPU memory

### Very slow processing
- First batch is always slowest
- Use GPU instead of CPU
- Reduce num_workers if disk is slow

## After Training

Checkpoints will be saved in the `checkpoints/` folder:
- `syncnet_fcn_epoch1.pth`
- `syncnet_fcn_epoch2.pth`
- etc.

### Load and Use Trained Model
```python
from SyncNetModel_FCN import StreamSyncFCN
import torch

model = StreamSyncFCN()
checkpoint = torch.load('checkpoints/syncnet_fcn_epoch10.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use it!
offset, confidence = model.process_video_file('test.mp4')
print(f"Offset: {offset:.2f} frames")
```
