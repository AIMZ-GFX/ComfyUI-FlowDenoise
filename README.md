# ComfyUI-FlowDenoise

Professional motion-compensated temporal video denoising for ComfyUI.

FlowDenoise leverages state-of-the-art optical flow estimation (MEMFOF / RAFT) to align neighboring frames, then separates and removes chroma and luma noise with independent per-channel control. Purpose-built for suppressing AI-generated video artifacts including color flicker, chroma spikes, and temporal noise patterns.
https://github.com/AIMZ-GFX/ComfyUI-FlowDenoise/issues/1#issue-4193001304
## Features

- **Optical flow alignment** via MEMFOF (state-of-the-art 2025) or RAFT models
- **Batched inference pipeline** for high-throughput processing on high-VRAM GPUs
- **Chroma / Luma separation** with independent denoising strength per channel
- **Multiple color spaces**: YCbCr, HSV, LAB
- **Noise visualization**: heatmap, signed (red/blue), and grayscale preview modes
- **Scene-aware processing**: automatic scene change detection prevents cross-scene blending artifacts

## Installation

### Via ComfyUI-Manager (Recommended)

Search for `ComfyUI-FlowDenoise` in ComfyUI-Manager and install.

### Manual Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/AIMZ-GFX/ComfyUI-FlowDenoise.git
```

### Dependencies

```bash
pip install memfof
```

The MEMFOF optical flow model (`egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH`) is automatically downloaded from HuggingFace on first use.

RAFT models (`raft_small`, `raft_large`) are provided via `torchvision` and require no additional installation.

## Nodes

### Temporal Flow Average

Motion-compensated temporal averaging using optical flow.

Aligns neighboring frames to the current frame using dense optical flow estimation, then computes a weighted average to produce a clean temporal reference. Frames further in time receive lower weight via exponential decay. Scene boundaries are automatically detected to prevent cross-cut blending.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 2 | Number of frames to average on each side (total window = 2n+1) |
| `weight_decay` | 0.8 | Exponential decay for temporal weights (lower = more aggressive averaging) |
| `flow_model` | memfof | Optical flow model: `memfof`, `raft_small`, `raft_large` |
| `flow_iterations` | 8 | Number of flow refinement iterations |
| `color_threshold` | 0.04 | Per-pixel color difference threshold for outlier rejection |
| `scene_threshold` | 0.06 | Scene change detection threshold (mean frame difference) |
| `batch_size` | 1 | MEMFOF batch size (higher = faster but more VRAM) |

**Outputs:**
- `clean` -- Temporally averaged (denoised) frames
- `weight_map` -- Per-pixel confidence weights used in the averaging

---

### Extract Noise (Chroma/Luma)

Extracts and visualizes the noise difference between original and clean frames, separated into chroma and luma components. Useful for diagnostics and tuning denoising parameters.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `noise_amplify` | 5.0 | Amplification factor for noise visualization |
| `color_space` | YCbCr | Color space for chroma/luma separation: `YCbCr`, `HSV`, `LAB` |
| `noise_preview` | heatmap | Visualization mode: `heatmap` (turbo colormap), `signed` (red=positive, blue=negative), `gray` (classic grayscale) |

**Outputs:**
- `noise_total` -- Combined noise visualization (all channels)
- `noise_chroma` -- Chroma noise only
- `noise_luma` -- Luminance noise only

---

### Selective Denoise

Selectively blends original and clean frames with independent chroma/luma control in the chosen color space. This is where the final denoising balance is set.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chroma_strength` | 0.8 | Chroma denoising strength (0=keep original, 1=fully clean) |
| `luma_strength` | 0.3 | Luma denoising strength (0=keep original, 1=fully clean) |
| `color_space` | YCbCr | Color space for separation: `YCbCr`, `HSV`, `LAB` |
| `clamp_output` | true | Clamp output values to [0, 1] |

## Workflow

An example workflow is included in `workflow_example.json`. The standard pipeline:

```
LoadVideo -> Temporal Flow Average -> Extract Noise (preview)
                                   -> Selective Denoise -> Save Video
```

1. **Load Video** -- Import your video with VHS_LoadVideo
2. **Temporal Flow Average** -- Align and average neighboring frames to create a clean reference
3. **Extract Noise** -- (Optional) Visualize what noise is being removed
4. **Selective Denoise** -- Blend original and clean with independent chroma/luma control
5. **Save Video** -- Export with VHS_VideoCombine

## Recommended Settings

### AI-Generated Video (Seedance, Kling, etc.)

Chroma flicker and color spikes common in AI video generators:

```
Temporal Flow Average:
  window_size: 2-3
  weight_decay: 0.7
  flow_model: memfof
  batch_size: 8-16 (RTX 5090 32GB, 720p)

Selective Denoise:
  chroma_strength: 0.7-0.9
  luma_strength: 0.1-0.3
  color_space: YCbCr
```

### Film Grain Removal

Subtle grain in live-action footage:

```
Temporal Flow Average:
  window_size: 3-5
  weight_decay: 0.6
  flow_model: memfof

Selective Denoise:
  chroma_strength: 0.5-0.7
  luma_strength: 0.3-0.5
  color_space: LAB
```

### Chroma-Only Cleanup

Remove color noise while preserving all luminance detail:

```
Selective Denoise:
  chroma_strength: 0.9
  luma_strength: 0.0
```

## How It Works

1. **Optical Flow Estimation**: MEMFOF computes dense motion vectors between adjacent frames
2. **Frame Alignment**: Neighboring frames are warped to match the current frame's viewpoint using the estimated flow
3. **Weighted Averaging**: Aligned frames are averaged with exponential temporal decay and per-pixel outlier rejection
4. **Color Space Separation**: The noise (original - clean) is decomposed into chroma and luma components
5. **Selective Blending**: Original and clean frames are blended with independent control per component

This purely mathematical approach requires no training, works on any video content, and produces deterministic results.

## License

MIT License. See [LICENSE](LICENSE) for details.

**Note**: This project depends on the MEMFOF optical flow model. Please verify the [MEMFOF model license](https://huggingface.co/egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH) for your intended use case.

## Acknowledgments

- [MEMFOF](https://huggingface.co/egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH) -- State-of-the-art optical flow model
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) -- Node-based Stable Diffusion GUI
- [VHS (Video Helper Suite)](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) -- Video I/O nodes for ComfyUI

---

*Developed by AIMZ GFX Division*
