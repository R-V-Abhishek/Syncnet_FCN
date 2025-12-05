#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SyncNet FCN - Flask Backend API

Provides a web API for the SyncNet FCN audio-video sync detection.
Serves the frontend and handles video analysis requests.

Usage:
    python app.py

Then open http://localhost:5000 in your browser.

Author: R-V-Abhishek
"""

import os
import sys
import json
import time
import shutil
import tempfile
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, static_folder='frontend', static_url_path='')

# Configuration
UPLOAD_FOLDER = tempfile.mkdtemp(prefix='syncnet_')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB max

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Global model instance (lazy loaded)
_model = None


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_model(window_size=25, stride=5, buffer_size=100, use_attention=False):
    """Get or create model instance."""
    global _model
    
    # Load FCN model with trained checkpoint
    from SyncNetModel_FCN import StreamSyncFCN
    import torch
    
    checkpoint_path = 'checkpoints/syncnet_fcn_epoch2.pth'
    
    model = StreamSyncFCN(
        max_offset=15,
        pretrained_syncnet_path=None,
        auto_load_pretrained=False
    )
    
    # Load trained weights
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        encoder_state = {k: v for k, v in checkpoint['model_state_dict'].items()
                        if 'audio_encoder' in k or 'video_encoder' in k}
        model.load_state_dict(encoder_state, strict=False)
        print(f"✓ Loaded FCN model (epoch {checkpoint.get('epoch', '?')})")
    
    model.eval()
    return model


# ========================================
# Routes
# ========================================

@app.route('/')
def index():
    """Serve the frontend."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def static_files(path):
    """Serve static files."""
    return send_from_directory(app.static_folder, path)


@app.route('/api/status')
def api_status():
    """Check API and model status."""
    try:
        # Check if model can be loaded
        pretrained_exists = os.path.exists('data/syncnet_v2.model')
        
        return jsonify({
            'status': 'Model Ready' if pretrained_exists else 'No Pretrained Model',
            'pretrained_available': pretrained_exists,
            'version': '1.0.0'
        })
    except Exception as e:
        return jsonify({
            'status': 'Error',
            'error': str(e)
        }), 500


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """Analyze a video for audio-video sync."""
    start_time = time.time()
    temp_video_path = None
    temp_dir = None
    
    try:
        # Check if video file is present
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        
        if video_file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        
        if not allowed_file(video_file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: MP4, AVI, MOV, MKV'}), 400
        
        # Get settings from form data
        window_size = int(request.form.get('window_size', 25))
        stride = int(request.form.get('stride', 5))
        buffer_size = int(request.form.get('buffer_size', 100))
        
        # Validate settings
        window_size = max(5, min(100, window_size))
        stride = max(1, min(50, stride))
        buffer_size = max(10, min(500, buffer_size))
        
        # Save uploaded file
        filename = secure_filename(video_file.filename)
        temp_video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(temp_video_path)
        
        # Create temp directory for processing
        temp_dir = tempfile.mkdtemp(prefix='syncnet_proc_')
        
        # Get model
        model = get_model(
            window_size=window_size,
            stride=stride,
            buffer_size=buffer_size
        )
        
        # Process video using calibrated method
        offset, confidence, raw_offset = model.detect_offset_correlation(
            video_path=temp_video_path,
            calibration_offset=3,
            calibration_scale=-0.5,
            calibration_baseline=-15,
            temp_dir=temp_dir,
            verbose=False
        )
        
        processing_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'video_name': filename,
            'offset_frames': int(offset),
            'offset_seconds': float(offset / 25.0),
            'confidence': float(confidence),
            'raw_offset': int(raw_offset),
            'processing_time': float(processing_time),
            'settings': {
                'window_size': window_size,
                'stride': stride,
                'buffer_size': buffer_size
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
        
    finally:
        # Cleanup
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except:
                pass
        
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass


@app.route('/api/analyze-stream', methods=['POST'])
def api_analyze_stream():
    """Analyze a HLS stream URL for audio-video sync."""
    start_time = time.time()
    temp_video_path = None
    temp_dir = None
    
    try:
        # Get JSON data
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'No stream URL provided'}), 400
        
        stream_url = data['url']
        
        # Validate URL
        if not stream_url.startswith(('http://', 'https://')):
            return jsonify({'error': 'Invalid URL. Must start with http:// or https://'}), 400
        
        # Get settings
        window_size = int(data.get('window_size', 25))
        stride = int(data.get('stride', 5))
        buffer_size = int(data.get('buffer_size', 100))
        
        # Validate settings
        window_size = max(5, min(100, window_size))
        stride = max(1, min(50, stride))
        buffer_size = max(10, min(500, buffer_size))
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix='syncnet_stream_')
        temp_video_path = os.path.join(temp_dir, 'stream_sample.mp4')
        
        # Download a segment of the stream using ffmpeg (10 seconds)
        import subprocess
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', stream_url,
            '-t', '10',  # 10 seconds
            '-c', 'copy',
            '-bsf:a', 'aac_adtstoasc',
            temp_video_path
        ]
        
        print(f"Downloading stream: {stream_url}")
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )
        
        if result.returncode != 0 or not os.path.exists(temp_video_path):
            # Try alternative approach without codec copy
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', stream_url,
                '-t', '10',
                '-c:v', 'libx264',
                '-c:a', 'aac',
                temp_video_path
            ]
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0 or not os.path.exists(temp_video_path):
                return jsonify({'error': f'Failed to download stream. FFmpeg error: {result.stderr[:500]}'}), 400
        
        # Get model
        model = get_model(
            window_size=window_size,
            stride=stride,
            buffer_size=buffer_size
        )
        
        # Process video
        proc_result = model.process_video_file(
            video_path=temp_video_path,
            return_trace=False,
            temp_dir=temp_dir,
            target_size=(112, 112),
            verbose=False
        )
        
        if proc_result is None:
            return jsonify({'error': 'Failed to process stream. Check if stream has audio track.'}), 400
        
        offset, confidence = proc_result
        processing_time = time.time() - start_time
        
        # Extract stream name from URL
        stream_name = stream_url.split('/')[-1][:50] if '/' in stream_url else stream_url[:50]
        
        return jsonify({
            'success': True,
            'video_name': stream_name,
            'source_url': stream_url,
            'offset_frames': float(offset),
            'offset_seconds': float(offset / 25.0),
            'confidence': float(confidence),
            'processing_time': float(processing_time),
            'settings': {
                'window_size': window_size,
                'stride': stride,
                'buffer_size': buffer_size
            }
        })
        
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Stream download timed out. The stream may be slow or unavailable.'}), 408
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
        
    finally:
        # Cleanup
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass


# ========================================
# Main
# ========================================

if __name__ == '__main__':
    print()
    print("=" * 50)
    print("  SyncNet FCN - Web Interface")
    print("=" * 50)
    print()
    print("  Starting server...")
    print("  Open http://localhost:5000 in your browser")
    print()
    print("  Press Ctrl+C to stop")
    print("=" * 50)
    print()
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )
