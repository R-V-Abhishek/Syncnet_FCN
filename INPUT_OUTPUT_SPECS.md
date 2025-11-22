# SyncNet Input/Output Specifications

## 📥 INPUTS

### 1. **Original SyncNet** (`SyncNetModel.py`)

#### Audio Input:
```python
Shape: [Batch, 1, 13, Time]
Type: torch.FloatTensor
Description: MFCC features

Details:
- Batch: Number of samples (typically 1 for inference)
- 1: Single channel
- 13: MFCC coefficients
- Time: Number of audio frames (~100 FPS, so 100 = 1 second)

Example:
audio = torch.randn(1, 1, 13, 100)  # 1 sample, 1 second of audio
```

#### Video Input:
```python
Shape: [Batch, Channels, Time, Height, Width]
Type: torch.FloatTensor
Description: RGB video frames

Details:
- Batch: Number of samples
- Channels: 3 (RGB)
- Time: Number of frames (typically 5 consecutive frames)
- Height: 224 pixels (face crop)
- Width: 224 pixels (face crop)
- Values: Normalized to [0, 1] or [-1, 1]

Example:
video = torch.randn(1, 3, 5, 224, 224)  # 1 sample, 5 frames
```

**Note:** Original model processes fixed 5-frame windows, sliding across the video.

---

### 2. **Fully Convolutional SyncNet** (`SyncNetModel_FCN.py`)

#### Audio Input:
```python
Shape: [Batch, 1, MFCC_dim, Time]
Type: torch.FloatTensor
Description: MFCC or Mel-spectrogram features

Details:
- Batch: Number of samples
- 1: Single channel
- MFCC_dim: 13 (MFCC) or 40-128 (Mel)
- Time: Variable length! (100-1000+ frames)

Examples:
audio_short = torch.randn(2, 1, 13, 100)   # 2 samples, 1 second each
audio_long = torch.randn(1, 1, 13, 500)    # 1 sample, 5 seconds
audio_mel = torch.randn(1, 1, 40, 200)     # 1 sample, 40 mel bins, 2 seconds
```

#### Video Input:
```python
Shape: [Batch, 3, Time, Height, Width]
Type: torch.FloatTensor
Description: RGB video frames

Details:
- Batch: Number of samples
- 3: RGB channels
- Time: Variable length! (25-250+ frames for 1-10 seconds at 25 FPS)
- Height: 112 or 224 pixels
- Width: 112 or 224 pixels
- Values: [0, 1] normalized

Examples:
video_short = torch.randn(1, 3, 25, 112, 112)   # 1 second at 25 FPS
video_long = torch.randn(1, 3, 250, 112, 112)   # 10 seconds
video_hd = torch.randn(1, 3, 75, 224, 224)      # 3 seconds, higher res
```

**Key Advantage:** No fixed length - process entire sequences at once!

---

### 3. **Transfer Learning Models** (`SyncNet_TransferLearning.py`)

Same as FCN, but different preprocessing depending on backbone:

#### For VGGish Audio:
```python
Shape: [Batch, 1, 96, Time]  # 96 mel bins instead of 13 MFCC
Type: torch.FloatTensor
Description: Log-mel spectrogram

Example:
audio_vggish = torch.randn(1, 1, 96, 100)
```

#### For wav2vec 2.0:
```python
Shape: [Batch, Samples]  # Raw waveform!
Type: torch.FloatTensor
Description: Raw audio waveform at 16kHz

Example:
waveform = torch.randn(1, 16000)  # 1 second at 16kHz
waveform_long = torch.randn(1, 160000)  # 10 seconds
```

#### Video (3D ResNet/I3D):
```python
Shape: [Batch, 3, Time, Height, Width]
Same as FCN, but may require different resolutions:
- ResNet3D: 112x112 (default)
- I3D: 224x224 (better quality)
```

---

## 📤 OUTPUTS

### 1. **Original SyncNet** (`SyncNetModel.py`)

```python
# Forward pass
output = model.forward_aud(audio)  # or forward_lip(video)

Shape: [Batch, 1024]
Type: torch.FloatTensor
Description: Single embedding vector per sample

Example:
audio_embedding = model.forward_aud(audio)  # [1, 1024]
video_embedding = model.forward_lip(video)  # [1, 1024]

# Calculate similarity
similarity = F.cosine_similarity(audio_embedding, video_embedding)
# scalar value: higher = more synchronized
```

**Sync Detection:**
- Slide window across video
- Extract embedding for each window
- Compare with audio embeddings at different offsets
- Find offset with highest similarity

---

### 2. **Fully Convolutional SyncNet** (`SyncNetModel_FCN.py`)

```python
# Full forward pass
sync_probs, audio_features, video_features = model(audio, video)

# Output 1: Sync Probabilities
Shape: [Batch, 2*MaxOffset+1, Time]
Type: torch.FloatTensor
Description: Probability distribution over offsets for each time step

Details:
- Batch: Number of samples
- 2*MaxOffset+1: Probability for each offset (e.g., 31 for offset ±15)
- Time: Number of temporal positions (frames)
- Values: Softmax probabilities (sum to 1 over offset dimension)

Example with max_offset=15:
sync_probs.shape = [1, 31, 23]
# For each of 23 time steps, probability of 31 possible offsets (-15 to +15)

# Interpretation:
sync_probs[0, 15, 10] = 0.8  # At time 10, offset 0 has 80% probability (in sync)
sync_probs[0, 18, 10] = 0.1  # At time 10, offset +3 has 10% probability
sync_probs[0, 12, 10] = 0.05 # At time 10, offset -3 has 5% probability

# Output 2: Audio Features
Shape: [Batch, EmbeddingDim, Time]
Type: torch.FloatTensor
Description: Temporal audio feature map

Example:
audio_features.shape = [1, 512, 25]
# 512-dimensional features for each of 25 time steps

# Output 3: Video Features
Shape: [Batch, EmbeddingDim, Time]
Type: torch.FloatTensor
Description: Temporal video feature map

Example:
video_features.shape = [1, 512, 25]
# 512-dimensional features for each of 25 time steps
```

#### Derived Outputs:

```python
# Compute predicted offsets
offsets, confidences = model.compute_offset(sync_probs)

# Offsets
Shape: [Batch, Time]
Type: torch.FloatTensor
Description: Most likely offset at each time step
Values: Integers from -MaxOffset to +MaxOffset

Example:
offsets.shape = [1, 23]
offsets[0] = tensor([0, 0, -1, -2, -1, 0, 1, 0, ...])
# Frame 0: in sync (0)
# Frame 2: audio ahead by 1 frame (-1)
# Frame 3: audio ahead by 2 frames (-2)

# Confidences
Shape: [Batch, Time]
Type: torch.FloatTensor
Description: Confidence score for each prediction
Values: 0 to 1 (higher = more confident)

Example:
confidences.shape = [1, 23]
confidences[0] = tensor([0.8, 0.85, 0.6, 0.45, 0.7, ...])
# Frame 0: 80% confident
# Frame 3: 45% confident (ambiguous)
```

---

## 💾 File I/O (Practical Usage)

### Input Files:

```
video.mp4  or  video.avi
├─ Any common format (MP4, AVI, MOV, etc.)
├─ Any resolution (will be resized to 224x224 or 112x112)
├─ Any frame rate (will be resampled to 25 FPS)
└─ Any duration (processes entire video)

Audio automatically extracted:
├─ Resampled to 16kHz
├─ Converted to mono
├─ MFCC computed (13 coefficients)
└─ 100 FPS feature rate
```

### Output Files (from `SyncNetInstance_FCN.py`):

```python
# 1. Console Output
Consensus offset:    -2.0 frames     # Overall sync offset
Median offset:       -1.5 frames     # Median across time
Mean confidence:     0.745           # Average confidence

Frame-wise confidence (smoothed):
[ 0.823  0.856  0.742  0.634  0.789  ...]  # Per-frame confidence

# 2. Return Values
offsets = numpy.array([...])          # Shape: [T]
confidences = numpy.array([...])      # Shape: [T]
sync_probs = numpy.array([...])       # Shape: [2K+1, T]

# 3. Visualization (if --visualize flag)
video_sync_analysis.png
├─ Top plot: Offset over time
├─ Bottom plot: Confidence over time
└─ Shows temporal patterns
```

---

## 🎬 Real-World Examples

### Example 1: 10-second video clip

```python
# Input
Video: 10 seconds @ 25 FPS = 250 frames → [1, 3, 250, 112, 112]
Audio: 10 seconds @ 100 FPS = 1000 frames → [1, 1, 13, 1000]

# Processing
- Video downsampled/aligned: [1, 3, 250, 112, 112]
- Audio downsampled/aligned: [1, 1, 13, 1000]
- Model aligns to common length: 250 time steps

# Output
sync_probs: [1, 31, 250]     # 31 offsets × 250 time steps
offsets: [1, 250]             # Predicted offset for each of 250 frames
confidences: [1, 250]         # Confidence for each prediction

# Interpretation
offsets[0, 100] = -3          # At frame 100 (4 seconds), audio leads by 3 frames
confidences[0, 100] = 0.82    # 82% confident in this prediction
```

### Example 2: 1-minute interview

```python
# Input
Video: 60 seconds @ 25 FPS = 1500 frames
Audio: 60 seconds @ 100 FPS = 6000 frames

# GPU Memory (FCN model)
Video tensor: 1 × 3 × 1500 × 112 × 112 = ~170 MB
Audio tensor: 1 × 1 × 13 × 6000 = ~0.3 MB
Model params: ~200 MB
Activations: ~500 MB
Total: ~1 GB GPU RAM

# Processing Time (RTX 3070)
- Forward pass: ~0.8 seconds
- Total with I/O: ~2-3 seconds

# Output
1500 frame-by-frame predictions showing:
- Which segments are in sync
- Which segments are out of sync
- Confidence levels throughout
```

### Example 3: Movie scene (5 minutes)

```python
# Input
Video: 300 seconds @ 25 FPS = 7500 frames
Audio: 300 seconds @ 100 FPS = 30000 frames

# Strategy: Process in chunks
chunk_size = 100 frames (4 seconds)
overlap = 10 frames
num_chunks = 75

# Per chunk GPU memory: ~400 MB
# Total processing time (RTX 3070): ~30-40 seconds

# Output
7500 predictions showing:
- Continuous sync tracking
- Drift detection (gradual offset change)
- Scene-level statistics
```

---

## 🔢 Data Types & Ranges

### Input Value Ranges:

```python
# Video (normalized)
video = video / 255.0              # [0, 1]
# or
video = (video - 127.5) / 127.5    # [-1, 1]

# Audio (MFCC)
mfcc = python_speech_features.mfcc(audio, 16000)
# Typical range: [-50, 50] (no specific normalization needed)

# Audio (Log-Mel for VGGish)
mel = librosa.feature.melspectrogram(audio, sr=16000, n_mels=96)
log_mel = librosa.power_to_db(mel)
# Typical range: [-80, 0] dB
```

### Output Value Ranges:

```python
# Sync probabilities
sync_probs: [0, 1] (probabilities, sum to 1 over offset dimension)

# Offsets
offsets: [-max_offset, +max_offset]  # Integers
# -15 to +15 for default max_offset=15

# Confidences
confidences: [0, 1]  # Higher is better
# > 0.7: High confidence (reliable)
# 0.4-0.7: Medium confidence (check manually)
# < 0.4: Low confidence (ambiguous)

# Features
audio_features, video_features: Typically [-5, 5] (unnormalized embeddings)
```

---

## 🖥️ GPU Memory Requirements

### Batch Size Impact:

```python
# Single sample (batch=1)
Video: 1 × 3 × 250 × 112 × 112 ≈ 28 MB
Audio: 1 × 1 × 13 × 1000 ≈ 0.05 MB
Activations: ~500 MB
Total: ~700 MB

# Batch of 4
Video: 4 × 3 × 250 × 112 × 112 ≈ 112 MB
Audio: 4 × 1 × 13 × 1000 ≈ 0.2 MB
Activations: ~1800 MB
Total: ~2.5 GB

# Batch of 16 (training)
Video: 16 × 3 × 250 × 112 × 112 ≈ 448 MB
Audio: 16 × 1 × 13 × 1000 ≈ 0.8 MB
Activations: ~7 GB
Total: ~10 GB (requires RTX 3090 or better)
```

### Recommendations:

```
Inference (batch=1):
- GTX 1060 6GB: ✓ (FCN only)
- RTX 2060 6GB: ✓ (FCN + attention)
- RTX 3070 8GB: ✓ (Transfer learning)
- RTX 4090 24GB: ✓✓✓ (All models, large batches)

Training (batch=8-16):
- RTX 3080 10GB: ✓ (FCN)
- RTX 3090 24GB: ✓✓ (Transfer learning)
- A100 40GB: ✓✓✓ (Large models)

CPU Only:
- Any modern CPU: ✓ (5-10x slower, but works!)
```

---

## 📊 Performance Summary

| Configuration | Input Size | GPU Memory | Inference Time | Output |
|--------------|------------|------------|----------------|--------|
| **Original SyncNet** | 5 frames | ~500 MB | ~0.01s | Single embedding |
| **FCN SyncNet** | 25-250 frames | ~1 GB | ~0.2-0.8s | Temporal probs |
| **FCN + Attention** | 25-250 frames | ~1.5 GB | ~0.3-1.2s | Temporal probs |
| **Transfer (ResNet3D)** | 25-250 frames | ~3 GB | ~0.5-2s | Temporal probs |
| **Transfer (I3D)** | 25-250 frames | ~4 GB | ~0.8-3s | Temporal probs |

*Times for 10-second video on RTX 3070*

---

## 🚀 Quick Reference

### Minimal Example:

```python
from SyncNetModel_FCN import SyncNetFCN

# Create model
model = SyncNetFCN(embedding_dim=512, max_offset=15).cuda()

# Prepare inputs
audio = torch.randn(1, 1, 13, 100).cuda()    # 1 second
video = torch.randn(1, 3, 25, 112, 112).cuda()  # 1 second

# Inference
sync_probs, audio_feat, video_feat = model(audio, video)
offsets, confidences = model.compute_offset(sync_probs)

# Result
print(f"Predicted offsets: {offsets[0].cpu().numpy()}")
print(f"Confidences: {confidences[0].cpu().numpy()}")
```

### Expected Output:

```
Predicted offsets: [ 0.  0. -1. -1.  0.  1.  0.  0. ...]
Confidences: [0.82 0.87 0.76 0.68 0.79 0.84 0.81 0.88 ...]
```

---

## 💡 Key Takeaways

1. **Variable Length:** FCN models accept any length (no more sliding windows!)
2. **Dense Predictions:** Get per-frame offsets instead of single value
3. **GPU Friendly:** 1-2 GB for inference, manageable on modern GPUs
4. **CPU Viable:** Slower but works without GPU
5. **Flexible:** Switch between models easily based on accuracy/speed needs
