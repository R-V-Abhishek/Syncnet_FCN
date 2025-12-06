# How to Deploy SyncNet FCN as a Command-Line Tool

This guide explains how to make your SyncNet FCN project available as system-wide command-line tools (like `syncnet-detect`, `syncnet-train`, etc.).

---

## 📋 Prerequisites

Before deployment, ensure you have:
- Python 3.8 or higher installed
- pip package manager
- FFmpeg installed and in your system PATH

---

## 🚀 Quick Deployment (3 Steps)

### Step 1: Create `setup.py`

Create a file named `setup.py` in your project root with this content:

```python
from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='syncnet-fcn',
    version='1.0.0',
    author='R-V-Abhishek',
    description='Fully Convolutional Audio-Video Synchronization Network',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.8',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'syncnet-detect=detect_sync:main',
            'syncnet-generate-demo=generate_demo:main',
            'syncnet-train-fcn=train_syncnet_fcn_complete:main',
            'syncnet-train-classification=train_syncnet_fcn_classification:main',
            'syncnet-evaluate=evaluate_model:main',
            'syncnet-fcn-pipeline=run_fcn_pipeline:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
)
```

### Step 2: Install the Package

Open PowerShell/Command Prompt in your project directory and run:

```bash
# For development (changes to code are immediately reflected)
pip install -e .

# OR for standard installation
pip install .
```

### Step 3: Verify Installation

Test that commands are available:

```bash
syncnet-detect --help
```

---

## 🎯 Available Commands After Installation

Once installed, you can use these commands from anywhere:

| Command | Purpose | Example Usage |
|---------|---------|---------------|
| `syncnet-detect` | Detect AV sync offset | `syncnet-detect video.mp4` |
| `syncnet-generate-demo` | Generate comparison demos | `syncnet-generate-demo --compare` |
| `syncnet-train-fcn` | Train FCN model | `syncnet-train-fcn --data_dir /path/to/data` |
| `syncnet-train-classification` | Train classification model | `syncnet-train-classification --epochs 10` |
| `syncnet-evaluate` | Evaluate model | `syncnet-evaluate --model model.pth` |
| `syncnet-fcn-pipeline` | Run FCN pipeline | `syncnet-fcn-pipeline --video video.mp4` |

---

## 📖 Usage Examples

### Example 1: Detect sync in a video
```bash
syncnet-detect Test_video.mp4 --verbose
```

### Example 2: Save results to JSON
```bash
syncnet-detect video.mp4 --output results.json
```

### Example 3: Batch process multiple videos (PowerShell)
```powershell
Get-ChildItem *.mp4 | ForEach-Object {
    syncnet-detect $_.FullName --output "$($_.BaseName)_sync.json"
}
```

### Example 4: Train classification model
```bash
syncnet-train-classification --data_dir C:\Datasets\VoxCeleb2 --epochs 10 --batch_size 32
```

---

## 🔧 Troubleshooting

### Problem: Command not found

**Solution 1:** Ensure Python Scripts directory is in PATH
- Windows: `C:\Users\<username>\AppData\Local\Programs\Python\Python3x\Scripts`
- Close and reopen your terminal after installation

**Solution 2:** Use Python module syntax
```bash
python -m detect_sync video.mp4
```

### Problem: Import errors

**Solution:** Reinstall dependencies
```bash
pip install --upgrade --force-reinstall -e .
```

---

## 🗑️ Uninstalling

To remove the command-line tools:

```bash
pip uninstall syncnet-fcn
```

---

## 🌐 Sharing Your Tool

### Option 1: Share as Wheel
```bash
pip install build
python -m build
# Share the .whl file from dist/ folder
```

### Option 2: Install from Git
```bash
pip install git+https://github.com/YOUR_USERNAME/Syncnet_FCN.git
```

### Option 3: Upload to PyPI
```bash
pip install twine
python -m build
twine upload dist/*
# Others can install: pip install syncnet-fcn
```

---

## ⚡ Development Workflow

1. **Make changes to your code**
2. **Test immediately** (if installed with `-e` flag)
3. **No reinstall needed** for editable installations

If you need to update the installed version:
```bash
pip install --upgrade .
```

---

## 💡 Key Points

- ✅ Use `pip install -e .` for development (editable mode)
- ✅ Use `pip install .` for production deployment
- ✅ All your Python scripts become system-wide commands
- ✅ Works on Windows, Mac, and Linux
- ✅ No need to specify full paths to scripts anymore

---

## 📝 Next Steps

1. Create `setup.py` in your project root
2. Run `pip install -e .`
3. Test with `syncnet-detect --help`
4. Start using your commands from anywhere!
