import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nodes import _warp_with_flow


# ---------------------------------------------------------------------------
# TemporalUNet: multi-frame U-Net with temporal context (~2M params)
# ---------------------------------------------------------------------------

class _DoubleConv(nn.Module):
    """Conv3x3 + ReLU + Conv3x3 + ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TemporalUNet(nn.Module):
    """Multi-frame U-Net for temporal video denoising.

    Takes 9-channel input (aligned_prev + current + aligned_next) and outputs
    3-channel corrected frame. The temporal context enables the network to
    compare the center frame with its neighbors and selectively correct
    spike frames while preserving normal frames.

    Architecture (~2M parameters):
        Encoder: 9→32→64→128
        Bottleneck: 128→256
        Decoder: 256→128→64→32→3
    """

    def __init__(self, in_channels: int = 9):
        super().__init__()
        # Encoder
        self.enc1 = _DoubleConv(in_channels, 32)
        self.enc2 = _DoubleConv(32, 64)
        self.enc3 = _DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = _DoubleConv(128, 256)

        # Decoder (concat skip → double channels in)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = _DoubleConv(256 + 128, 128)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = _DoubleConv(128 + 64, 64)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = _DoubleConv(64 + 32, 32)

        # Final 1x1 → 3ch output
        self.final = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)              # [B, 32, H, W]
        e2 = self.enc2(self.pool(e1))   # [B, 64, H/2, W/2]
        e3 = self.enc3(self.pool(e2))   # [B, 128, H/4, W/4]

        # Bottleneck
        b = self.bottleneck(self.pool(e3))  # [B, 256, H/8, W/8]

        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = self._pad_and_cat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._pad_and_cat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._pad_and_cat(d1, e1)
        d1 = self.dec1(d1)

        return self.final(d1)  # [B, 3, H, W]

    @staticmethod
    def _pad_and_cat(upsampled, skip):
        """Handle odd-sized feature maps by padding before concat."""
        dh = skip.shape[2] - upsampled.shape[2]
        dw = skip.shape[3] - upsampled.shape[3]
        if dh != 0 or dw != 0:
            upsampled = F.pad(upsampled, [0, dw, 0, dh])
        return torch.cat([upsampled, skip], dim=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_checkpoint_dir() -> str:
    """Return ComfyUI/models/video_denoise/ path, creating it if needed."""
    here = os.path.dirname(os.path.abspath(__file__))
    comfyui_root = os.path.abspath(os.path.join(here, "..", ".."))
    ckpt_dir = os.path.join(comfyui_root, "models", "video_denoise")
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir


def _load_memfof(compute_device: torch.device):
    """Load MEMFOF optical flow model."""
    from memfof import MEMFOF
    return MEMFOF.from_pretrained(
        "egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH"
    ).eval().to(compute_device)


def _compute_aligned_neighbors(frames: torch.Tensor, i: int,
                                flow_model, compute_device: torch.device):
    """Compute flow-aligned prev/next neighbors for frame i.

    Args:
        frames: BCHW tensor on CPU
        i: frame index
        flow_model: MEMFOF model on GPU
        compute_device: GPU device

    Returns:
        aligned_prev, aligned_next: both [1, 3, H, W] on GPU
    """
    B = frames.shape[0]
    prev_idx = max(i - 1, 0)
    next_idx = min(i + 1, B - 1)

    triplet = torch.stack([
        frames[prev_idx], frames[i], frames[next_idx],
    ], dim=0).unsqueeze(0).to(compute_device)
    triplet_255 = (triplet * 255.0).clamp(0, 255)

    with torch.no_grad():
        out = flow_model(triplet_255, iters=8)

    flow_field = out["flow"][-1]  # [1, 2, 2, H, W]
    bwd_flow = flow_field[:, 0]   # warps prev to current viewpoint
    fwd_flow = flow_field[:, 1]   # warps next to current viewpoint
    del triplet, triplet_255, out, flow_field

    # Align prev neighbor
    if prev_idx != i:
        prev_gpu = frames[prev_idx:prev_idx + 1].to(compute_device)
        aligned_prev = _warp_with_flow(prev_gpu, bwd_flow)
        del prev_gpu
    else:
        aligned_prev = frames[i:i + 1].to(compute_device)

    # Align next neighbor
    if next_idx != i:
        next_gpu = frames[next_idx:next_idx + 1].to(compute_device)
        aligned_next = _warp_with_flow(next_gpu, fwd_flow)
        del next_gpu
    else:
        aligned_next = frames[i:i + 1].to(compute_device)

    del bwd_flow, fwd_flow
    return aligned_prev, aligned_next


def _generate_training_pairs(frames: torch.Tensor, compute_device: torch.device):
    """Generate multi-frame training pairs using MEMFOF optical flow.

    For each frame_i:
    - Input (9ch): concat(aligned_prev, frame_i, aligned_next)
    - Target (3ch): average(aligned_prev, aligned_next)

    The 9ch input gives the network temporal context to distinguish spike
    frames (center ≠ neighbors) from normal frames (center ≈ neighbors).

    Returns:
        Tuple of (pairs, weights) where:
        - pairs: List of (input_9ch, target_3ch) tuples on CPU
        - weights: per-frame L1 distances for weighted sampling
    """
    B = frames.shape[0]
    if B < 2:
        return [], torch.tensor([])

    flow_model = _load_memfof(compute_device)

    pairs = []
    pair_diffs = []

    for i in range(B):
        aligned_prev, aligned_next = _compute_aligned_neighbors(
            frames, i, flow_model, compute_device)

        frame_i_gpu = frames[i:i + 1].to(compute_device)

        # 9ch input: (aligned_prev, current, aligned_next)
        input_9ch = torch.cat([aligned_prev, frame_i_gpu, aligned_next],
                              dim=1).cpu()  # [1, 9, H, W]

        # Target: neighbor average (clean reference)
        target = ((aligned_prev + aligned_next) / 2.0).cpu()  # [1, 3, H, W]

        diff = (frame_i_gpu.cpu() - target).abs().mean().item()
        pair_diffs.append(diff)

        pairs.append((input_9ch, target))
        del aligned_prev, aligned_next, frame_i_gpu

    del flow_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    weights = torch.tensor(pair_diffs)
    print(f"[TrainVideoDenoiser] Per-frame L1 stats: "
          f"min={weights.min():.6f}, max={weights.max():.6f}, "
          f"mean={weights.mean():.6f}, std={weights.std():.6f}")
    top_k = min(10, len(weights))
    top_vals, top_idx = weights.topk(top_k)
    print(f"[TrainVideoDenoiser] Top-{top_k} highest-diff frames (likely spikes): "
          f"indices={top_idx.tolist()}, diffs={[f'{v:.4f}' for v in top_vals.tolist()]}")

    return pairs, weights


def _train_unet(unet: TemporalUNet, pairs: list, training_steps: int,
                learning_rate: float, batch_size: int, patch_size: int,
                compute_device: torch.device, pair_weights: torch.Tensor = None):
    """Train TemporalUNet on multi-frame pairs with weighted sampling.

    Args:
        unet: TemporalUNet model (already on compute_device)
        pairs: list of (input_9ch, target_3ch), CPU tensors
        training_steps: number of training iterations
        learning_rate: Adam learning rate
        batch_size: crops per step
        patch_size: random crop spatial size
        compute_device: GPU device
        pair_weights: per-pair L1 distances for weighted sampling (None=uniform)
    """
    import random
    import numpy as np

    unet.train()
    optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)

    n_pairs = len(pairs)
    if n_pairs == 0:
        return

    # Build sampling probabilities.
    # With multi-frame input, the network CAN distinguish spikes from normal
    # frames. Linear weighting gives spikes moderate emphasis (~20-30% of batches).
    if pair_weights is not None and len(pair_weights) == n_pairs:
        w = pair_weights.clone()
        w = w + w.mean() * 0.05  # floor for normal frames
        sampling_probs = (w / w.sum()).numpy()
        print(f"  [TrainVideoDenoiser] Weighted sampling: "
              f"top pair prob={sampling_probs.max():.4f}, "
              f"bottom pair prob={sampling_probs.min():.4f}, "
              f"ratio={sampling_probs.max() / max(sampling_probs.min(), 1e-10):.1f}x")
    else:
        sampling_probs = None

    with torch.inference_mode(False), torch.enable_grad():
        for step in range(training_steps):
            crops_input = []
            crops_target = []

            if sampling_probs is not None:
                indices = np.random.choice(n_pairs, size=batch_size,
                                           replace=True, p=sampling_probs)
            else:
                indices = [random.randint(0, n_pairs - 1) for _ in range(batch_size)]

            for idx in indices:
                inp, tgt = pairs[idx]  # inp: [1,9,H,W], tgt: [1,3,H,W]
                _, _, H, W = inp.shape

                if H > patch_size and W > patch_size:
                    top = random.randint(0, H - patch_size)
                    left = random.randint(0, W - patch_size)
                else:
                    top, left = 0, 0
                ps_h = min(patch_size, H)
                ps_w = min(patch_size, W)

                crops_input.append(inp[:, :, top:top + ps_h, left:left + ps_w])
                crops_target.append(tgt[:, :, top:top + ps_h, left:left + ps_w])

            batch_inp = torch.cat(crops_input, dim=0).to(compute_device).clone()
            batch_tgt = torch.cat(crops_target, dim=0).to(compute_device).clone()

            clean_estimate = unet(batch_inp)
            loss = F.l1_loss(clean_estimate, batch_tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 500 == 0 or step == training_steps - 1:
                print(f"  [TrainVideoDenoiser] step {step}/{training_steps}  "
                      f"loss={loss.item():.6f}")

    unet.eval()


# ---------------------------------------------------------------------------
# ComfyUI Node: TrainVideoDenoiser
# ---------------------------------------------------------------------------

class TrainVideoDenoiser:
    """Train a multi-frame U-Net denoiser on the input video.
    Uses MEMFOF optical flow to align neighboring frames, giving the network
    temporal context to detect and correct spike frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "training_steps": ("INT", {
                    "default": 3000, "min": 100, "max": 50000, "step": 100,
                    "tooltip": "Number of training iterations"
                }),
                "learning_rate": ("FLOAT", {
                    "default": 0.0001, "min": 0.000001, "max": 0.01,
                    "step": 0.00001,
                    "tooltip": "Adam optimizer learning rate"
                }),
                "batch_size": ("INT", {
                    "default": 4, "min": 1, "max": 32, "step": 1,
                    "tooltip": "Number of random crops per training step"
                }),
                "patch_size": ("INT", {
                    "default": 128, "min": 32, "max": 512, "step": 32,
                    "tooltip": "Random crop size for training patches"
                }),
                "checkpoint_name": ("STRING", {
                    "default": "denoise_model",
                    "tooltip": "Checkpoint filename (saved to models/video_denoise/)"
                }),
            },
            "optional": {
                "base_checkpoint": ("STRING", {
                    "default": "",
                    "tooltip": "Existing checkpoint path for fine-tuning (leave empty to train from scratch)"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("checkpoint_path",)
    FUNCTION = "train"
    CATEGORY = "FlowDenoise"
    DESCRIPTION = ("Train a multi-frame video denoiser with temporal context. "
                   "The network learns to detect and correct spike frames by "
                   "comparing each frame to its flow-aligned neighbors.")

    def train(self, images: torch.Tensor, training_steps: int,
              learning_rate: float, batch_size: int, patch_size: int,
              checkpoint_name: str, base_checkpoint: str = ""):

        compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        B, H, W, C = images.shape

        with torch.inference_mode(False):
            frames = images.clone().permute(0, 3, 1, 2)  # BHWC → BCHW

            # --- Phase 1: Generate multi-frame training pairs ---
            print(f"[TrainVideoDenoiser] Phase 1: Generating {B} training pairs "
                  f"(9ch multi-frame) with MEMFOF flow...")
            pairs, pair_weights = _generate_training_pairs(frames, compute_device)
            print(f"[TrainVideoDenoiser] Phase 1 complete: {len(pairs)} pairs generated")

            # --- Phase 2: Train TemporalUNet ---
            unet = TemporalUNet()
            if base_checkpoint and os.path.isfile(base_checkpoint):
                print(f"[TrainVideoDenoiser] Loading base checkpoint: {base_checkpoint}")
                state = torch.load(base_checkpoint, map_location="cpu",
                                   weights_only=True)
                unet.load_state_dict(state)

            unet = unet.to(compute_device)
            print(f"[TrainVideoDenoiser] Phase 2: Training for {training_steps} steps "
                  f"(batch={batch_size}, patch={patch_size}, lr={learning_rate})...")
            _train_unet(unet, pairs, training_steps, learning_rate,
                        batch_size, patch_size, compute_device,
                        pair_weights=pair_weights)

        # --- Save checkpoint ---
        ckpt_dir = _get_checkpoint_dir()
        save_name = checkpoint_name if checkpoint_name.endswith(".pth") \
            else checkpoint_name + ".pth"
        save_path = os.path.join(ckpt_dir, save_name)
        torch.save(unet.cpu().state_dict(), save_path)
        print(f"[TrainVideoDenoiser] Checkpoint saved: {save_path}")

        del unet, pairs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (save_path,)


# ---------------------------------------------------------------------------
# ComfyUI Node: ApplyVideoDenoiser
# ---------------------------------------------------------------------------

class ApplyVideoDenoiser:
    """Apply a trained multi-frame denoiser to video frames.
    Uses MEMFOF optical flow at inference to provide temporal context,
    enabling selective correction of spike frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "checkpoint_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to trained denoiser checkpoint (.pth)"
                }),
            },
            "optional": {
                "fine_tune_steps": ("INT", {
                    "default": 0, "min": 0, "max": 10000, "step": 100,
                    "tooltip": "Additional fine-tuning steps on current video (0=skip)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("denoised",)
    FUNCTION = "apply"
    CATEGORY = "FlowDenoise"
    DESCRIPTION = ("Apply a trained multi-frame video denoiser. Uses MEMFOF "
                   "optical flow to align neighbors, giving temporal context "
                   "for selective spike correction.")

    def apply(self, images: torch.Tensor, checkpoint_path: str,
              fine_tune_steps: int = 0):

        compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        B, H, W, C = images.shape
        frames = images.permute(0, 3, 1, 2)  # BHWC → BCHW, CPU

        # Load checkpoint
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}")
        unet = TemporalUNet()
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        unet.load_state_dict(state)
        unet = unet.to(compute_device)

        # Optional fine-tune
        if fine_tune_steps > 0:
            with torch.inference_mode(False):
                frames_clone = frames.clone()
                print(f"[ApplyVideoDenoiser] Fine-tuning for {fine_tune_steps} steps...")
                pairs, pair_weights = _generate_training_pairs(
                    frames_clone, compute_device)
                _train_unet(unet, pairs, fine_tune_steps,
                            learning_rate=0.00005, batch_size=4, patch_size=128,
                            compute_device=compute_device,
                            pair_weights=pair_weights)
                del pairs, pair_weights, frames_clone

        # --- Inference: compute aligned neighbors + denoise per frame ---
        unet.eval()
        output = torch.zeros_like(frames)  # CPU
        print(f"[ApplyVideoDenoiser] Denoising {B} frames (with MEMFOF context)...")

        flow_model = _load_memfof(compute_device)

        frame_diffs = []
        with torch.no_grad():
            for i in range(B):
                aligned_prev, aligned_next = _compute_aligned_neighbors(
                    frames, i, flow_model, compute_device)
                frame_i_gpu = frames[i:i + 1].to(compute_device)

                # Build 9ch input: (aligned_prev, current, aligned_next)
                input_9ch = torch.cat([aligned_prev, frame_i_gpu, aligned_next],
                                      dim=1)

                clean = unet(input_9ch).clamp(0, 1)  # [1, 3, H, W]

                diff = (frame_i_gpu - clean).abs().mean().item()
                frame_diffs.append(diff)
                output[i:i + 1] = clean.cpu()
                del aligned_prev, aligned_next, frame_i_gpu, input_9ch, clean

        del flow_model, unet
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Log results
        diffs_t = torch.tensor(frame_diffs)
        avg_diff = diffs_t.mean().item()
        top_k = min(10, B)
        top_vals, top_idx = diffs_t.topk(top_k)
        print(f"[ApplyVideoDenoiser] Avg diff: {avg_diff:.6f}")
        print(f"[ApplyVideoDenoiser] Top-{top_k} most corrected frames: "
              f"indices={top_idx.tolist()}, "
              f"diffs={[f'{v:.4f}' for v in top_vals.tolist()]}")
        print(f"[ApplyVideoDenoiser] Done.")

        result = output.permute(0, 2, 3, 1)  # BCHW → BHWC
        return (result,)
