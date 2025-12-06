# 🚀 Deployment Guide

This guide covers deploying FCN-SyncNet to various platforms.

---

## 🤗 Hugging Face Spaces (Recommended)

**Pros:**
- ✅ Free GPU/CPU instances
- ✅ Good RAM allocation
- ✅ Easy sharing and embedding
- ✅ Automatic Git LFS for large models
- ✅ Public or private spaces

**Cons:**
- ⚠️ Cold start time
- ⚠️ Public by default

### Step-by-Step Deployment

#### 1. Prepare Your Repository

```bash
# Navigate to project directory
cd c:\Users\admin\Syncnet_FCN

# Copy README for Hugging Face
copy README_HF.md README.md

# Ensure all files are committed
git add .
git commit -m "Prepare for Hugging Face deployment"
```

#### 2. Create Hugging Face Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Fill in details:
   - **Space name**: `fcn-syncnet`
   - **License**: MIT
   - **SDK**: Gradio
   - **Hardware**: CPU (upgrade to GPU if needed)

#### 3. Initialize Git LFS (for large model files)

```bash
# Install Git LFS if not already installed
git lfs install

# Track model files
git lfs track "*.pth"
git lfs track "*.model"

# Add .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS for model files"
```

#### 4. Push to Hugging Face

```bash
# Add Hugging Face remote
git remote add hf https://huggingface.co/spaces/<your-username>/fcn-syncnet

# Push to Hugging Face
git push hf main
```

#### 5. Files Needed on Hugging Face

Ensure these files are in your repository:
- ✅ `app_gradio.py` (main application)
- ✅ `requirements_hf.txt` → rename to `requirements.txt`
- ✅ `README_HF.md` → rename to `README.md`
- ✅ `checkpoints/syncnet_fcn_epoch2.pth` (Git LFS)
- ✅ `data/syncnet_v2.model` (Git LFS)
- ✅ `detectors/s3fd/weights/sfd_face.pth` (Git LFS)
- ✅ All `.py` files (models, instances, detect_sync, etc.)

#### 6. Configure Space Settings

In your Hugging Face Space settings:
- **SDK**: Gradio
- **Python version**: 3.10
- **Hardware**: Start with CPU, upgrade to GPU if needed

---

## 🎓 Google Colab

**Pros:**
- ✅ Free GPU access (Tesla T4)
- ✅ Good for demos and testing
- ✅ Easy to share notebooks

**Cons:**
- ⚠️ Session timeouts
- ⚠️ Not suitable for production

### Deployment Steps

1. Create a new Colab notebook
2. Install dependencies:

```python
!git clone https://github.com/R-V-Abhishek/Syncnet_FCN.git
%cd Syncnet_FCN
!pip install -r requirements.txt
```

3. Run the app:

```python
!python app_gradio.py
```

4. Use Colab's public URL feature to share

---

## 🚂 Railway.app

**Pros:**
- ✅ Easy deployment from GitHub
- ✅ Automatic HTTPS
- ✅ Good performance

**Cons:**
- ⚠️ Paid service ($5-20/month)
- ⚠️ Sleep after inactivity on free tier

### Deployment Steps

1. Go to [railway.app](https://railway.app)
2. Connect GitHub repository
3. Add `railway.json`:

```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python app.py",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

4. Set environment variables (if needed)
5. Deploy!

---

## 🎨 Render

**Pros:**
- ✅ Free tier available
- ✅ Easy setup
- ✅ Good for small projects

**Cons:**
- ⚠️ Slow cold starts
- ⚠️ Limited free tier resources

### Deployment Steps

1. Create `render.yaml`:

```yaml
services:
  - type: web
    name: fcn-syncnet
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python app.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.10.0
```

2. Connect GitHub repo to Render
3. Deploy!

---

## ☁️ Cloud Platforms (AWS/GCP/Azure)

**Pros:**
- ✅ Full control
- ✅ Scalable
- ✅ Production-ready

**Cons:**
- ⚠️ Requires payment
- ⚠️ More complex setup

### Recommended Services

**AWS:**
- EC2 (GPU instances: g4dn.xlarge)
- Lambda (serverless, but cold start issues)
- Elastic Beanstalk (easy deployment)

**Google Cloud:**
- Compute Engine (GPU VMs)
- Cloud Run (serverless containers)

**Azure:**
- VM with GPU
- App Service

---

## 📊 Resource Requirements

| Platform | RAM | GPU | Storage | Cost |
|----------|-----|-----|---------|------|
| Hugging Face | 16GB | Optional | 5GB | Free |
| Colab | 12GB | Tesla T4 | 15GB | Free |
| Railway | 8GB | No | 10GB | $5-20/mo |
| Render | 512MB-4GB | No | 1GB | Free-$7/mo |
| AWS EC2 g4dn | 16GB | NVIDIA T4 | 125GB | ~$0.50/hr |

---

## 🎯 Recommended Deployment Path

### For Testing/Demos:
1. **Google Colab** - Quickest for testing
2. **Hugging Face Spaces** - Best for sharing

### For Production:
1. **Hugging Face Spaces** (if traffic is low-medium)
2. **Railway/Render** (if you need custom domain)
3. **AWS/GCP** (if you need high performance/scale)

---

## 🔧 Environment Variables (if needed)

```bash
# Model paths (if not using default)
FCN_MODEL_PATH=checkpoints/syncnet_fcn_epoch2.pth
ORIGINAL_MODEL_PATH=data/syncnet_v2.model
FACE_DETECTOR_PATH=detectors/s3fd/weights/sfd_face.pth

# Calibration parameters
CALIBRATION_OFFSET=3
CALIBRATION_SCALE=-0.5
CALIBRATION_BASELINE=-15
```

---

## 📝 Post-Deployment Checklist

- [ ] Test video upload functionality
- [ ] Verify model loads correctly
- [ ] Check offset detection accuracy
- [ ] Test with various video formats
- [ ] Monitor resource usage
- [ ] Set up error logging
- [ ] Add rate limiting (if public)

---

## 🐛 Troubleshooting

### Issue: Model file too large for Git
**Solution:** Use Git LFS (Large File Storage)

```bash
git lfs install
git lfs track "*.pth"
git lfs track "*.model"
```

### Issue: Out of memory on Hugging Face
**Solution:** Upgrade to GPU space or optimize model loading

### Issue: Cold start too slow
**Solution:** Use Railway/Render with always-on instances (paid)

### Issue: Video processing timeout
**Solution:** 
- Increase timeout limits
- Process videos asynchronously
- Use smaller video chunks

---

## 📞 Support

For deployment issues:
1. Check logs on the platform
2. Review [GitHub Issues](https://github.com/R-V-Abhishek/Syncnet_FCN/issues)
3. Consult platform documentation

---

*Happy Deploying! 🚀*
