# SyncNet FCN Project Progress

## Project Overview
Audio-Visual Sync Detection using Fully Convolutional Networks (FCN) based on SyncNet architecture.

---

## Progress Log

### November 28, 2025

#### ✅ Completed Tasks

1. **Fixed CUDA Compatibility Issues**
   - Modified `SyncNetInstance.py` to auto-detect CUDA availability
   - Modified `SyncNetInstance_FCN.py` to auto-detect CUDA availability
   - Code now runs on both GPU (CUDA) and CPU seamlessly
   - Changed all `.cuda()` calls to `.to(self.device)` pattern
   - Fixed `map_location` in model loading to use detected device

2. **Tested Original SyncNet Demo**
   - Successfully ran `demo_syncnet.py` with example video
   - Results:
     - **AV offset**: 3 frames (~120ms audio ahead)
     - **Min dist**: 5.358 (excellent lip-sync quality)
     - **Confidence**: 10.081 (very high confidence)
   - Framewise confidence analysis shows consistent sync quality

3. **Updated `.gitignore`**
   - Fixed `__pycache__/` pattern (was missing underscores)
   - Changed `data/` to `data/work/` to allow example files
   - Added Python bytecode patterns
   - Added virtual environment variations
   - Added IDE/editor files
   - Added Jupyter notebook checkpoints
   - Added model file extensions
   - Added temporary file patterns

4. **Created Test Videos with Controlled Offsets**
   - Created `test_videos/` folder with 6 test videos
   - Used FFmpeg `adelay` filter to introduce precise audio delays
   - Used FFmpeg `atrim` filter to create audio-ahead scenarios
   - Test videos cover range from -400ms to +600ms offset

5. **Validated SyncNet Model Accuracy**
   - Ran comprehensive tests on all offset videos
   - Model correctly detects relative offsets within ±1 frame accuracy
   - All tests passed with high confidence scores

#### Test Results Summary

| Video | Introduced Offset | Detected Offset | Min Dist | Confidence | Status |
|-------|-------------------|-----------------|----------|------------|--------|
| `original.avi` | 0 (baseline ~3) | **3** | 5.358 | 10.081 | ✅ Baseline |
| `audio_delay_200ms.avi` | +5 frames | **-2** | 6.790 | 8.272 | ✅ Pass |
| `audio_delay_400ms.avi` | +10 frames | **-7** | 6.852 | 8.082 | ✅ Pass |
| `audio_delay_600ms.avi` | +15 frames | **-12** | 6.936 | 8.936 | ✅ Pass |
| `audio_ahead_200ms.avi` | -5 frames | **8** | 6.810 | 8.200 | ✅ Pass |
| `audio_ahead_400ms.avi` | -10 frames | **13** | 6.889 | 8.947 | ✅ Pass |

**Note:** Detected offsets are relative to baseline (3). Model correctly identifies:
- Audio delay → negative offset shift
- Audio ahead → positive offset shift

---

## Next Steps

- [ ] Test FCN model variant (`SyncNetInstance_FCN.py`)
- [ ] Compare FCN vs original model performance
- [ ] Test with different video types (interviews, presentations)
- [ ] Document findings for supervisor meeting

