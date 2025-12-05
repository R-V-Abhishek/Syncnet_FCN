/**
 * SyncNet FCN - Frontend JavaScript
 * Handles video upload, API communication, and results display
 */

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const removeFile = document.getElementById('removeFile');
const settingsToggle = document.getElementById('settingsToggle');
const settingsContent = document.getElementById('settingsContent');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');
const errorModal = document.getElementById('errorModal');
const closeModal = document.getElementById('closeModal');

// State
let selectedFile = null;

// Settings Elements
const windowSizeInput = document.getElementById('windowSize');
const strideInput = document.getElementById('stride');
const bufferSizeInput = document.getElementById('bufferSize');

// Results Elements
const syncBadge = document.getElementById('syncBadge');
const offsetFrames = document.getElementById('offsetFrames');
const offsetSeconds = document.getElementById('offsetSeconds');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceFill = document.getElementById('confidenceFill');
const processingTime = document.getElementById('processingTime');
const videoName = document.getElementById('videoName');
const interpretationBox = document.getElementById('interpretationBox');
const interpretationIcon = document.getElementById('interpretationIcon');
const interpretationText = document.getElementById('interpretationText');
const fixSuggestion = document.getElementById('fixSuggestion');
const fixText = document.getElementById('fixText');

// API Base URL
const API_BASE = window.location.origin;

// ========================================
// Event Listeners
// ========================================

// Upload Area Click
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// File Input Change
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// Drag and Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    
    if (e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
});

// Remove File
removeFile.addEventListener('click', () => {
    clearFileSelection();
});

// Settings Toggle
settingsToggle.addEventListener('click', () => {
    settingsToggle.classList.toggle('open');
    settingsContent.classList.toggle('hidden');
});

// Analyze Button
analyzeBtn.addEventListener('click', () => {
    if (selectedFile) {
        analyzeVideo();
    }
});

// Close Modal
closeModal.addEventListener('click', () => {
    errorModal.classList.add('hidden');
});

// Close modal on backdrop click
errorModal.addEventListener('click', (e) => {
    if (e.target === errorModal) {
        errorModal.classList.add('hidden');
    }
});

// ========================================
// Functions
// ========================================

/**
 * Handle file selection
 */
function handleFileSelect(file) {
    // Validate file type
    const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/x-msvideo'];
    const validExtensions = ['.mp4', '.avi', '.mov', '.mkv'];
    
    const extension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!validTypes.includes(file.type) && !validExtensions.includes(extension)) {
        showError('Invalid File Type', 'Please upload a video file (MP4, AVI, MOV, or MKV)');
        return;
    }
    
    selectedFile = file;
    
    // Update UI
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    
    uploadArea.classList.add('hidden');
    fileInfo.classList.remove('hidden');
    analyzeBtn.disabled = false;
    
    // Hide previous results
    resultsSection.classList.add('hidden');
}

/**
 * Clear file selection
 */
function clearFileSelection() {
    selectedFile = null;
    fileInput.value = '';
    
    uploadArea.classList.remove('hidden');
    fileInfo.classList.add('hidden');
    analyzeBtn.disabled = true;
}

/**
 * Format file size
 */
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    return (bytes / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
}

/**
 * Show error modal
 */
function showError(title, message) {
    document.getElementById('errorTitle').textContent = title;
    document.getElementById('errorMessage').textContent = message;
    errorModal.classList.remove('hidden');
}

/**
 * Set button loading state
 */
function setLoading(loading) {
    const btnText = analyzeBtn.querySelector('.btn-text');
    const btnLoader = analyzeBtn.querySelector('.btn-loader');
    
    if (loading) {
        btnText.classList.add('hidden');
        btnLoader.classList.remove('hidden');
        analyzeBtn.disabled = true;
    } else {
        btnText.classList.remove('hidden');
        btnLoader.classList.add('hidden');
        analyzeBtn.disabled = false;
    }
}

/**
 * Analyze video via API
 */
async function analyzeVideo() {
    if (!selectedFile) return;
    
    setLoading(true);
    
    // Prepare form data
    const formData = new FormData();
    formData.append('video', selectedFile);
    formData.append('window_size', windowSizeInput.value);
    formData.append('stride', strideInput.value);
    formData.append('buffer_size', bufferSizeInput.value);
    
    try {
        const response = await fetch(`${API_BASE}/api/analyze`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `Server error: ${response.status}`);
        }
        
        const result = await response.json();
        displayResults(result);
        
    } catch (error) {
        console.error('Analysis error:', error);
        showError('Analysis Failed', error.message || 'Failed to analyze video. Please try again.');
    } finally {
        setLoading(false);
    }
}

/**
 * Display analysis results
 */
function displayResults(result) {
    resultsSection.classList.remove('hidden');
    
    const offset = result.offset_frames;
    const confidence = result.confidence;
    const time = result.processing_time;
    const video = result.video_name || selectedFile.name;
    
    // Update offset
    offsetFrames.textContent = offset >= 0 ? `+${offset.toFixed(1)}` : offset.toFixed(1);
    offsetSeconds.textContent = `${(offset / 25).toFixed(3)} seconds`;
    
    // Update confidence
    const confidencePercent = Math.round(confidence * 100);
    confidenceValue.textContent = confidencePercent;
    confidenceFill.style.width = `${confidencePercent}%`;
    
    // Update processing time
    processingTime.textContent = time.toFixed(2);
    videoName.textContent = video;
    
    // Determine sync status
    const isSynced = Math.abs(offset) <= 2; // Within 2 frames is considered synced
    
    // Update sync badge
    syncBadge.className = 'sync-badge ' + (isSynced ? 'synced' : 'out-of-sync');
    syncBadge.querySelector('.badge-text').textContent = isSynced ? 'In Sync' : 'Out of Sync';
    
    // Update interpretation
    interpretationBox.className = 'interpretation-box ' + (isSynced ? 'synced' : 'out-of-sync');
    
    if (isSynced) {
        interpretationIcon.textContent = '✓';
        interpretationText.textContent = 'Audio and video are synchronized. No adjustment needed.';
        fixSuggestion.classList.add('hidden');
    } else if (offset > 0) {
        interpretationIcon.textContent = '⚠';
        interpretationText.textContent = `Audio is ${Math.abs(offset).toFixed(1)} frames behind the video.`;
        fixText.textContent = `Delay audio by ${Math.abs(offset / 25).toFixed(3)} seconds`;
        fixSuggestion.classList.remove('hidden');
    } else {
        interpretationIcon.textContent = '⚠';
        interpretationText.textContent = `Audio is ${Math.abs(offset).toFixed(1)} frames ahead of the video.`;
        fixText.textContent = `Advance audio by ${Math.abs(offset / 25).toFixed(3)} seconds`;
        fixSuggestion.classList.remove('hidden');
    }
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ========================================
// Initialize
// ========================================

// Check API status on load
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_BASE}/api/status`);
        if (response.ok) {
            const data = await response.json();
            document.getElementById('modelStatus').textContent = data.status || 'Model Ready';
        }
    } catch (error) {
        console.warn('Could not check API status:', error);
        document.getElementById('modelStatus').textContent = 'Connecting...';
    }
});
