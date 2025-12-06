import gradio as gr
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detect_sync import detect_offset_correlation
from SyncNetInstance_FCN import SyncNetInstance as SyncNetInstanceFCN

# Initialize model
print("Loading FCN-SyncNet model...")
fcn_model = SyncNetInstanceFCN()
fcn_model.loadParameters("checkpoints/syncnet_fcn_epoch2.pth")
print("Model loaded successfully!")

def analyze_video(video_file):
    """
    Analyze a video file for audio-video synchronization
    
    Args:
        video_file: Uploaded video file path
        
    Returns:
        str: Analysis results
    """
    try:
        if video_file is None:
            return "❌ Please upload a video file"
        
        print(f"Processing video: {video_file}")
        
        # Detect offset using correlation method with calibration
        offset, conf, min_dist = detect_offset_correlation(
            video_file, 
            fcn_model,
            calibration_offset=3,
            calibration_scale=-0.5,
            calibration_baseline=-15
        )
        
        # Interpret results
        if offset > 0:
            sync_status = f"🔊 Audio leads video by {offset} frames"
            description = "Audio is playing before the corresponding video frames"
        elif offset < 0:
            sync_status = f"🎬 Video leads audio by {abs(offset)} frames"
            description = "Video is playing before the corresponding audio"
        else:
            sync_status = "✅ Audio and video are synchronized"
            description = "Perfect synchronization detected"
        
        # Confidence interpretation
        if conf > 0.8:
            conf_text = "Very High"
            conf_emoji = "🟢"
        elif conf > 0.6:
            conf_text = "High"
            conf_emoji = "🟡"
        elif conf > 0.4:
            conf_text = "Medium"
            conf_emoji = "🟠"
        else:
            conf_text = "Low"
            conf_emoji = "🔴"
        
        result = f"""
## 📊 Sync Detection Results

### {sync_status}

**Description:** {description}

---

### 📈 Detailed Metrics

- **Offset:** {offset} frames
- **Confidence:** {conf_emoji} {conf:.2%} ({conf_text})
- **Min Distance:** {min_dist:.4f}

---

### 💡 Interpretation

- **Positive offset:** Audio is ahead of video (delayed video sync)
- **Negative offset:** Video is ahead of audio (delayed audio sync)
- **Zero offset:** Perfect synchronization

---

### ⚡ Model Info

- **Model:** FCN-SyncNet (Calibrated)
- **Processing:** ~3x faster than original SyncNet
- **Calibration:** Applied (offset=3, scale=-0.5, baseline=-15)
        """
        
        return result
        
    except Exception as e:
        return f"❌ Error processing video: {str(e)}\n\nPlease ensure the video has both audio and video tracks."

# Create Gradio interface
with gr.Blocks(title="FCN-SyncNet: Audio-Video Sync Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎬 FCN-SyncNet: Real-Time Audio-Visual Synchronization Detection
    
    Upload a video to detect audio-video synchronization offset. This model uses a Fully Convolutional Network (FCN) 
    for fast and accurate sync detection.
    
    ### How it works:
    1. Upload a video file (MP4, AVI, MOV, etc.)
    2. The model extracts audio-visual features
    3. Correlation analysis detects the offset
    4. Calibration ensures accurate results
    
    ### Performance:
    - **Speed:** ~3x faster than original SyncNet
    - **Accuracy:** Matches original SyncNet performance
    - **Real-time capable:** Can process HLS streams
    """)
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video")
            analyze_btn = gr.Button("🔍 Analyze Sync", variant="primary", size="lg")
        
        with gr.Column():
            output_text = gr.Markdown(label="Results")
    
    analyze_btn.click(
        fn=analyze_video,
        inputs=video_input,
        outputs=output_text
    )
    
    gr.Markdown("""
    ---
    
    ## 📚 About
    
    This project implements a **Fully Convolutional Network (FCN)** approach to audio-visual synchronization detection,
    built upon the original SyncNet architecture.
    
    ### Key Features:
    - ✅ **3x faster** than original SyncNet
    - ✅ **Calibrated output** corrects regression-to-mean bias
    - ✅ **Real-time capable** for HLS streams
    - ✅ **High accuracy** matches original SyncNet
    
    ### Research Journey:
    - Tried regression (regression-to-mean problem)
    - Tried classification (loss of precision)
    - **Solution:** Correlation method + calibration formula
    
    ### GitHub:
    [github.com/R-V-Abhishek/Syncnet_FCN](https://github.com/R-V-Abhishek/Syncnet_FCN)
    
    ---
    
    *Built with ❤️ using Gradio and PyTorch*
    """)

if __name__ == "__main__":
    demo.launch()
