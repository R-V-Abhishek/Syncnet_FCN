# SyncNet Lab Setup - Complete Checklist

**Date:** November 27, 2025  
**For:** College Project Lab (Shared Environment)  
**Project:** SyncNet Audio-Visual Synchronization

---

## 🚨 BEFORE YOU GO TO LAB - Quick Prep

- [ ] **Push your code to GitHub** (so you can clone in lab)
  ```powershell
  git add .
  git commit -m "Ready for lab testing"
  git push
  ```

- [ ] **Find 2-3 test videos** (10-30 sec talking head videos, save links or filenames)

- [ ] **Keep this checklist open** (or print it)

That's it! Everything else downloads faster directly in lab.

---

## ✅ PHASE 1: Initial Lab Setup (10 mins)

### 1.1 Check What's Available
- [ ] Open command prompt/PowerShell
- [ ] Check Python version:
  ```powershell
  python --version
  ```
  - ✅ **Supported: Python 3.9, 3.10, 3.11, 3.12, 3.13** (all work with PyTorch!)

- [ ] Check if pip works:
  ```powershell
  pip --version
  ```

- [ ] Check if CUDA is available (for GPU):
  ```powershell
  nvidia-smi
  ```
  - Note the CUDA version if available
  - ⚠️ **If GPU is old/unsupported, see Section 1.4 below**

- [ ] Check FFmpeg:
  ```powershell
  ffmpeg -version
  ```
  - ⚠️ If not found, you need to install it (see 1.3)

### 1.2 Install Python (if missing or wrong version)

**If Python not found or version < 3.9:**

- [ ] **Option A - Chocolatey (if admin available):**
  ```powershell
  choco install python
  ```

- [ ] **Option B - From python.org (no admin needed):**
  1. Go to: https://www.python.org/downloads/
  2. Download Python 3.11 or 3.12 (recommended)
  3. Run installer
  4. ✅ **IMPORTANT:** Check "Add Python to PATH" during install
  5. Check "Install for all users" if you have admin

- [ ] Verify after install:
  ```powershell
  # Close and reopen PowerShell first!
  python --version
  ```

### 1.3 Install FFmpeg (if missing)
- [ ] **Option A - Chocolatey (if admin):**
  ```powershell
  choco install ffmpeg
  ```

- [ ] **Option B - Manual (no admin):**
  1. Download from: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
  2. Extract to: `C:\Users\<YourUser>\ffmpeg`
  3. Add to PATH for this session:
     ```powershell
     $env:PATH += ";C:\Users\$env:USERNAME\ffmpeg\bin"
     ```

- [ ] Verify FFmpeg works:
  ```powershell
  ffmpeg -version
  ```

### 1.4 Handling Unsupported/Old GPU or No CUDA

**Check if your GPU supports CUDA:**
```powershell
nvidia-smi
```

**Common scenarios:**

| Scenario | What You'll See | Solution |
|----------|-----------------|----------|
| **No NVIDIA GPU** | `nvidia-smi` not found | Use CPU-only PyTorch |
| **Old GPU (pre-2016)** | GPU shows but CUDA fails | Use CPU-only PyTorch |
| **CUDA version mismatch** | CUDA errors at runtime | Install matching PyTorch |

**Install CPU-only PyTorch (works on ANY machine):**
```powershell
# Instead of the CUDA version, use this:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Check GPU Compute Capability:**
- GTX 700 series or older → ❌ Not supported, use CPU
- GTX 900 series → ⚠️ May work with older CUDA
- GTX 1000+ series → ✅ Supported
- RTX series → ✅ Fully supported

**If you're unsure, just use CPU version** - it's slower but always works!

**Performance comparison:**
| Hardware | Time for 30sec video |
|----------|---------------------|
| CPU (i5/i7) | ~30-60 seconds |
| Old GPU | ~15-30 seconds |
| Modern GPU | ~5-10 seconds |

For testing purposes, CPU is perfectly fine!

---

## ✅ PHASE 2: Project Setup (10 mins)

### 2.1 Clone Your Project
- [ ] Create working directory:
  ```powershell
  cd C:\Users\$env:USERNAME
  ```

- [ ] Clone from GitHub:
  ```powershell
  git clone https://github.com/R-V-Abhishek/Syncnet_FCN.git
  cd Syncnet_FCN
  ```

### 2.2 Create Virtual Environment
- [ ] Create venv:
  ```powershell
  python -m venv venv
  ```

- [ ] Activate venv:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  
  - ⚠️ **If you get "execution policy" error:**
    ```powershell
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    .\venv\Scripts\Activate.ps1
    ```
  
  - ⚠️ **Alternative (cmd.exe):**
    ```cmd
    venv\Scripts\activate.bat
    ```

- [ ] Verify venv is active (you should see `(venv)` in prompt)

### 2.3 Install Dependencies
- [ ] **Check GPU situation first:**
  ```powershell
  nvidia-smi
  ```

- [ ] **If CUDA available (modern NVIDIA GPU):**
  ```powershell
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

- [ ] **If NO CUDA / Old GPU / Unsure:**
  ```powershell
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```

- [ ] Install other requirements:
  ```powershell
  pip install -r requirements.txt
  ```

- [ ] Verify installation:
  ```powershell
  python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
  python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
  ```
  - If CUDA shows `False` but you installed CPU version, that's expected and fine!

---

## ✅ PHASE 3: Download Models (5 mins)

### 3.1 Create Directories & Download
- [ ] Create folders and download everything:
  ```powershell
  mkdir data
  mkdir data\work\pytmp -Force
  mkdir detectors\s3fd\weights -Force
  
  # Download pretrained model (~47 MB)
  Invoke-WebRequest -Uri "http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model" -OutFile "data/syncnet_v2.model"
  
  # Download example video (~2 MB)
  Invoke-WebRequest -Uri "http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/example.avi" -OutFile "data/example.avi"
  
  # Download face detector (~89 MB)
  Invoke-WebRequest -Uri "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth" -OutFile "detectors/s3fd/weights/sfd_face.pth"
  ```

### 3.2 Verify Downloads
- [ ] Check files exist:
  ```powershell
  dir data\
  dir detectors\s3fd\weights\
  ```

Expected:
```
data\syncnet_v2.model  (~47 MB)
data\example.avi       (~2 MB)
detectors\s3fd\weights\sfd_face.pth  (~89 MB)
```

---

## ✅ PHASE 4: First Test Run (10 mins)

### 4.1 Quick Sanity Test
- [ ] Make sure venv is activated!
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```

- [ ] Run demo:
  ```powershell
  python demo_syncnet.py --videofile data/example.avi --tmp_dir data/work/pytmp
  ```

### 4.2 Expected Output
```
Model data/syncnet_v2.model loaded.
Compute time X.XXX sec.
Framewise conf: 
[...]
AV offset:      3 
Min dist:       5.353
Confidence:     10.021
```

### 4.3 Troubleshooting First Run

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'torch'` | Activate venv! `.\venv\Scripts\Activate.ps1` |
| `FileNotFoundError: syncnet_v2.model` | Copy model to `data/` folder |
| `ffmpeg not found` | Install FFmpeg (see 1.2) |
| `CUDA out of memory` | Add `--batch_size 5` to command |
| `CUDA not available` | Will run on CPU (slower but works) |

---

## 🎬 WHERE TO GET TEST VIDEOS

### Quick Test (Included with model)
- `data/example.avi` - Downloaded with the model, ready to use

### Free Test Videos Online

**Option 1: YouTube (Download with yt-dlp)**
```powershell
# Install yt-dlp
pip install yt-dlp

# Create test_videos folder first
mkdir test_videos

# Download a short interview clip (30 sec)
# Replace VIDEO_ID with actual YouTube video ID
yt-dlp -f "best[height<=480]" --download-sections "*0:00-0:30" -o ".\test_videos\youtube_test.mp4" "https://www.youtube.com/watch?v=VIDEO_ID"
```

**Note:** Run this from your project folder (`Syncnet_FCN`), or use full path:
```powershell
yt-dlp -f "best[height<=480]" --download-sections "*0:00-0:30" -o "D:\Coding\syncnet_python\test_videos\youtube_test.mp4" "https://www.youtube.com/watch?v=VIDEO_ID"
```

**Good YouTube search terms:**
- "interview face close up"
- "talking head video"
- "speech presentation"

**Option 2: Pexels (Free stock videos)**
- https://www.pexels.com/search/videos/talking/
- https://www.pexels.com/search/videos/interview/
- No login required, free to download

**Option 3: VoxCeleb Test Samples**
- https://www.robots.ox.ac.uk/~vgg/data/voxceleb/
- Small test samples available without full registration

**Option 4: Record yourself!**
- Use phone camera, 10-30 seconds
- Clear face, speaking to camera
- Transfer via email/cloud to lab PC

### Test Video Requirements
| Property | Requirement |
|----------|-------------|
| Duration | 5-60 seconds ideal |
| Face | Clearly visible, facing camera |
| Audio | Clear speech (not music) |
| Format | MP4, AVI, MOV all work |
| Resolution | 480p or higher |

---

## ✅ PHASE 5: Testing with Controlled Offsets (20 mins)

### 5.1 Create Test Videos Folder
- [ ] Create test folder:
  ```powershell
  mkdir test_videos
  ```

- [ ] Copy your test video to this folder:
  ```powershell
  copy E:\my_test_video.mp4 test_videos\original.mp4
  ```

### 5.2 Create Offset Test Videos
- [ ] **Test 1: Original (should be offset ~0)**
  ```powershell
  python demo_syncnet.py --videofile test_videos/original.mp4 --tmp_dir data/work/pytmp
  ```
  - [ ] Record result: Offset = ___, Confidence = ___

- [ ] **Create +5 frame delay (200ms):**
  ```powershell
  ffmpeg -i test_videos/original.mp4 -itsoffset 0.2 -i test_videos/original.mp4 -map 0:v -map 1:a -c copy test_videos/delay_5frames.mp4
  ```

- [ ] **Test +5 frame delay:**
  ```powershell
  python demo_syncnet.py --videofile test_videos/delay_5frames.mp4 --tmp_dir data/work/pytmp
  ```
  - [ ] Record result: Offset = ___, Confidence = ___
  - [ ] Expected: Offset around +5

- [ ] **Create +10 frame delay (400ms):**
  ```powershell
  ffmpeg -i test_videos/original.mp4 -itsoffset 0.4 -i test_videos/original.mp4 -map 0:v -map 1:a -c copy test_videos/delay_10frames.mp4
  ```

- [ ] **Test +10 frame delay:**
  ```powershell
  python demo_syncnet.py --videofile test_videos/delay_10frames.mp4 --tmp_dir data/work/pytmp
  ```
  - [ ] Record result: Offset = ___, Confidence = ___
  - [ ] Expected: Offset around +10

- [ ] **Create +15 frame delay (600ms):**
  ```powershell
  ffmpeg -i test_videos/original.mp4 -itsoffset 0.6 -i test_videos/original.mp4 -map 0:v -map 1:a -c copy test_videos/delay_15frames.mp4
  ```

- [ ] **Test +15 frame delay:**
  ```powershell
  python demo_syncnet.py --videofile test_videos/delay_15frames.mp4 --tmp_dir data/work/pytmp
  ```
  - [ ] Record result: Offset = ___, Confidence = ___
  - [ ] Expected: Offset around +15

### 5.3 Results Recording Table

| Video | Introduced Offset | Detected Offset | Confidence | Pass? |
|-------|-------------------|-----------------|------------|-------|
| original.mp4 | 0 | ___ | ___ | ___ |
| delay_5frames.mp4 | +5 | ___ | ___ | ___ |
| delay_10frames.mp4 | +10 | ___ | ___ | ___ |
| delay_15frames.mp4 | +15 | ___ | ___ | ___ |

**Pass Criteria:** Detected offset within ±2 frames of introduced offset

---

## ✅ PHASE 6: Document Results (15 mins)

### 6.1 Create Results Summary
- [ ] Screenshot each test output
- [ ] Fill in the results table above
- [ ] Note any errors or issues

### 6.2 Performance Assessment

**Based on your results, check one:**

- [ ] **✅ MODEL WORKS WELL** - All offsets detected within ±2 frames
  - → Move to Phase 7A (Data Creation Planning)

- [ ] **⚠️ MODEL NEEDS IMPROVEMENT** - Offsets incorrect by >2 frames
  - → Move to Phase 7B (Test Improved Versions)

- [ ] **❌ MODEL FAILS** - Completely wrong results or crashes
  - → Debug and document issues for supervisor

---

## ✅ PHASE 7A: If Model Works - Data Planning

### Data Creation Questions to Answer

- [ ] **Use Case:** What will this be used for?
  - [ ] Deepfake detection
  - [ ] Dubbing quality check
  - [ ] Live streaming sync
  - [ ] Other: _______________

- [ ] **Data Type Needed:**
  - [ ] Same domain as VoxCeleb (celebrity interviews) → Minimal data needed
  - [ ] Different domain → Need domain-specific data

- [ ] **How to Create Data:**
  - Take synced videos from target domain
  - Use FFmpeg to create offset versions (like we did in testing)
  - Create positive (synced) and negative (offset) pairs

- [ ] **How Much Data:**
  - For testing: 50-100 clips
  - For fine-tuning: 1,000+ clips

---

## ✅ PHASE 7B: If Model Needs Improvement

### Test FCN Variant
- [ ] Check if FCN files exist:
  ```powershell
  dir SyncNetModel_FCN.py
  dir SyncNetInstance_FCN.py
  ```

- [ ] Review example usage:
  ```powershell
  type example_usage.py
  ```

- [ ] Note: FCN model may need training first (no pretrained weights)

### Document Issues for Supervisor
- [ ] What specific offsets failed?
- [ ] What was the confidence level?
- [ ] Any patterns in failures?

---

## 🆘 EMERGENCY TROUBLESHOOTING

### "Python not found"
```powershell
# Check if Python is in a different location
where.exe python
# Or use full path
C:\Python39\python.exe --version
```

### "Permission denied" errors
```powershell
# Work in your user folder instead
cd C:\Users\%USERNAME%\syncnet_project
```

### "pip install fails"
```powershell
# Try with --user flag
pip install --user -r requirements.txt
```

### "CUDA out of memory"
```powershell
# Reduce batch size
python demo_syncnet.py --videofile data/example.avi --tmp_dir data/work/pytmp --batch_size 5
```

### "Cannot activate venv"
```powershell
# Use cmd.exe instead of PowerShell
cmd
venv\Scripts\activate.bat
```

### "Module not found" after installing
```powershell
# Make sure you're in the right venv
where.exe python
# Should show: C:\...\syncnet_project\venv\Scripts\python.exe
```

---

## 📋 QUICK REFERENCE COMMANDS

```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# Deactivate venv
deactivate

# Run demo test
python demo_syncnet.py --videofile data/example.avi --tmp_dir data/work/pytmp

# Check PyTorch/CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Create delayed video
ffmpeg -i input.mp4 -itsoffset 0.2 -i input.mp4 -map 0:v -map 1:a -c copy output.mp4
```

---

## 📝 RESULTS TO SHOW SUPERVISOR

1. Screenshot of successful model load
2. Results table with offset detection accuracy  
3. Assessment: Does model need improvement?
4. If yes → What specific improvements to try
5. If no → Data creation plan for your use case

---

*Checklist created: November 27, 2025*  
*Total time: ~60 minutes* 🚀
