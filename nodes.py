import torch
import torch.nn.functional as F
import comfy.utils
from tqdm import tqdm

# Module-level model cache (stays in CPU memory between runs)
_memfof_model = None


def _get_memfof_model(device):
    """Load MEMFOF model once, cache in CPU memory, move to device on demand."""
    global _memfof_model
    if _memfof_model is None:
        from memfof import MEMFOF
        _memfof_model = MEMFOF.from_pretrained(
            "egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH"
        ).eval()
    return _memfof_model.to(device)


def _rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """Convert BCHW RGB [0,1] to HSV [0,1]."""
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    maxc, _ = rgb.max(dim=1, keepdim=True)
    minc, _ = rgb.min(dim=1, keepdim=True)
    diff = maxc - minc

    s = torch.where(maxc > 0, diff / (maxc + 1e-8), torch.zeros_like(maxc))
    v = maxc

    h = torch.zeros_like(maxc)
    mask_r = (maxc == r) & (diff > 0)
    mask_g = (maxc == g) & (diff > 0)
    mask_b = (maxc == b) & (diff > 0)
    h[mask_r] = (((g - b) / (diff + 1e-8)) % 6)[mask_r]
    h[mask_g] = (((b - r) / (diff + 1e-8)) + 2)[mask_g]
    h[mask_b] = (((r - g) / (diff + 1e-8)) + 4)[mask_b]
    h = h / 6.0

    return torch.cat([h, s, v], dim=1)


def _rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    """Convert BCHW RGB [0,1] to YCbCr [0,1]."""
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.5 + (-0.168736 * r - 0.331264 * g + 0.5 * b)
    cr = 0.5 + (0.5 * r - 0.418688 * g - 0.081312 * b)
    return torch.cat([y, cb, cr], dim=1)


def _rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """Convert BCHW RGB [0,1] to approximate LAB, normalized to [0,1] range."""
    # Linearize sRGB
    linear = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

    r, g, b = linear[:, 0:1], linear[:, 1:2], linear[:, 2:3]
    # RGB to XYZ (D65)
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    # Normalize by D65 white point
    x = x / 0.95047
    z = z / 1.08883

    epsilon = 0.008856
    kappa = 903.3

    def f(t):
        return torch.where(t > epsilon, t ** (1.0 / 3.0), (kappa * t + 16.0) / 116.0)

    fx, fy, fz = f(x), f(y), f(z)

    l_star = 116.0 * fy - 16.0    # [0, 100]
    a_star = 500.0 * (fx - fy)    # ~ [-128, 128]
    b_star = 200.0 * (fy - fz)    # ~ [-128, 128]

    # Normalize to [0,1] for IMAGE compatibility
    l_norm = l_star / 100.0
    a_norm = (a_star + 128.0) / 256.0
    b_norm = (b_star + 128.0) / 256.0

    return torch.cat([l_norm, a_norm, b_norm], dim=1)


def _ycbcr_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    """Convert BCHW YCbCr [0,1] back to RGB [0,1]."""
    y, cb, cr = ycbcr[:, 0:1], ycbcr[:, 1:2], ycbcr[:, 2:3]
    r = y + 1.402 * (cr - 0.5)
    g = y - 0.344136 * (cb - 0.5) - 0.714136 * (cr - 0.5)
    b = y + 1.772 * (cb - 0.5)
    return torch.cat([r, g, b], dim=1)


def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """Convert BCHW HSV [0,1] back to RGB [0,1]."""
    h, s, v = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    h6 = h * 6.0
    i = torch.floor(h6)
    f = h6 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i_mod = i.long() % 6

    r = torch.where(i_mod == 0, v, torch.where(i_mod == 1, q, torch.where(
        i_mod == 2, p, torch.where(i_mod == 3, p, torch.where(i_mod == 4, t, v)))))
    g = torch.where(i_mod == 0, t, torch.where(i_mod == 1, v, torch.where(
        i_mod == 2, v, torch.where(i_mod == 3, q, torch.where(i_mod == 4, p, p)))))
    b = torch.where(i_mod == 0, p, torch.where(i_mod == 1, p, torch.where(
        i_mod == 2, t, torch.where(i_mod == 3, v, torch.where(i_mod == 4, v, q)))))
    return torch.cat([r, g, b], dim=1)


def _lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    """Convert BCHW normalized LAB back to RGB [0,1]."""
    l_star = lab[:, 0:1] * 100.0
    a_star = lab[:, 1:2] * 256.0 - 128.0
    b_star = lab[:, 2:3] * 256.0 - 128.0

    fy = (l_star + 16.0) / 116.0
    fx = a_star / 500.0 + fy
    fz = fy - b_star / 200.0

    epsilon = 0.008856
    kappa = 903.3

    def f_inv(t):
        # Use t*t*t instead of t**3 to safely handle negative values
        t3 = t * t * t
        return torch.where(t3 > epsilon, t3, (116.0 * t - 16.0) / kappa)

    x = f_inv(fx) * 0.95047
    y = f_inv(fy)
    z = f_inv(fz) * 1.08883

    r_lin = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g_lin = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b_lin = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z

    linear = torch.cat([r_lin, g_lin, b_lin], dim=1).clamp(min=0)
    srgb = torch.where(linear > 0.0031308,
                       1.055 * linear.clamp(min=1e-10) ** (1.0 / 2.4) - 0.055,
                       12.92 * linear)
    return srgb.clamp(0, 1)


def _convert_color_space(rgb_bchw: torch.Tensor, space: str) -> torch.Tensor:
    """Convert BCHW RGB to the chosen color space."""
    if space == "HSV":
        return _rgb_to_hsv(rgb_bchw)
    elif space == "YCbCr":
        return _rgb_to_ycbcr(rgb_bchw)
    elif space == "LAB":
        return _rgb_to_lab(rgb_bchw)
    else:
        raise ValueError(f"Unknown color space: {space}")


def _convert_back_to_rgb(cs_bchw: torch.Tensor, space: str) -> torch.Tensor:
    """Convert BCHW color space back to RGB."""
    if space == "HSV":
        return _hsv_to_rgb(cs_bchw)
    elif space == "YCbCr":
        return _ycbcr_to_rgb(cs_bchw)
    elif space == "LAB":
        return _lab_to_rgb(cs_bchw)
    else:
        raise ValueError(f"Unknown color space: {space}")


def _warp_with_flow(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp img (B,C,H,W) using flow (B,2,H,W). flow[b,0]=dx, flow[b,1]=dy in pixels."""
    B, C, H, W = img.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=img.device, dtype=img.dtype),
        torch.arange(W, device=img.device, dtype=img.dtype),
        indexing='ij'
    )
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

    new_x = grid_x + flow[:, 0]
    new_y = grid_y + flow[:, 1]

    # Normalize to [-1, 1]
    new_x = 2.0 * new_x / (W - 1) - 1.0
    new_y = 2.0 * new_y / (H - 1) - 1.0

    grid = torch.stack([new_x, new_y], dim=-1)
    warped = F.grid_sample(img, grid, mode='bilinear', padding_mode='border',
                           align_corners=True)
    return warped


def _color_similarity_mask(warped: torch.Tensor, ref: torch.Tensor,
                           threshold: float) -> torch.Tensor:
    """Soft color similarity mask. Returns (B,1,H,W) weights in [0,1]."""
    color_diff = torch.mean((warped - ref) ** 2, dim=1, keepdim=True)
    mask = torch.exp(-color_diff / (2.0 * threshold ** 2))
    # Soften edges with 5x5 blur
    mask = F.avg_pool2d(mask, kernel_size=5, stride=1, padding=2)
    return mask


def _forward_backward_occlusion_mask(flow_fwd: torch.Tensor, flow_bwd: torch.Tensor,
                                      threshold: float = 1.0) -> torch.Tensor:
    """Forward-backward consistency check for occlusion detection.
    Returns (B,1,H,W) mask: 1=consistent (not occluded), 0=occluded.
    flow_fwd: ref->neighbor, flow_bwd: neighbor->ref, both (B,2,H,W)."""
    # Warp backward flow using forward flow to check consistency
    bwd_at_fwd = _warp_with_flow(flow_bwd, flow_fwd)
    # If flow is consistent, fwd + warped_bwd ≈ 0
    fb_diff = torch.sum((flow_fwd + bwd_at_fwd) ** 2, dim=1, keepdim=True)
    fb_norm = torch.sum(flow_fwd ** 2 + bwd_at_fwd ** 2, dim=1, keepdim=True)
    # Relative threshold: inconsistency relative to flow magnitude
    mask = (fb_diff < threshold ** 2 * (fb_norm + 0.5)).float()
    mask = F.avg_pool2d(mask, kernel_size=5, stride=1, padding=2)
    return mask


class TemporalFlowAverage:
    """Motion-compensated temporal averaging using MEMFOF or RAFT optical flow."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "window_size": ("INT", {
                    "default": 2, "min": 1, "max": 15, "step": 1,
                    "tooltip": "Number of neighboring frames in each direction to average"
                }),
                "weight_decay": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Exponential decay per frame distance (1.0 = equal weight)"
                }),
                "flow_model": (["memfof", "raft_small", "raft_large"], {
                    "default": "memfof",
                    "tooltip": "Optical flow model (memfof=SOTA 2025, most accurate)"
                }),
                "flow_iterations": ("INT", {
                    "default": 8, "min": 1, "max": 32, "step": 1,
                    "tooltip": "Refinement iterations (memfof: 8 is good, raft: 20 recommended)"
                }),
                "color_threshold": ("FLOAT", {
                    "default": 0.04, "min": 0.005, "max": 0.5, "step": 0.005,
                    "tooltip": "Color similarity threshold (lower=stricter, rejects ghosting artifacts)"
                }),
                "scene_threshold": ("FLOAT", {
                    "default": 0.06, "min": 0.001, "max": 0.5, "step": 0.001,
                    "tooltip": "Scene change detection threshold (frame MSE above this = cut boundary, skip averaging)"
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 64, "step": 1,
                    "tooltip": "MEMFOF batch size (higher=faster but more VRAM. RTX 5090 32GB 720p: 8~16 recommended)"
                }),
                "precision": (["bf16", "fp32"], {
                    "default": "bf16",
                    "tooltip": "Inference precision. bf16 is ~1.5-2x faster on RTX 30/40/50 series with negligible quality difference. Use fp32 for strict reproducibility."
                }),
                "flow_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.3, "max": 1.0, "step": 0.1,
                    "tooltip": "Compute optical flow at reduced resolution for speed. 1.0=full resolution (best quality), 0.5~3x faster. Warping still uses full resolution."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("clean", "weight_map")
    FUNCTION = "denoise"
    CATEGORY = "FlowDenoise"
    DESCRIPTION = "Motion-compensated temporal averaging using MEMFOF (SOTA) or RAFT optical flow."

    def _accumulate_warped(self, warped, ref_frame, w, color_threshold,
                           weighted_sum, weight_sum):
        """Add warped neighbor to the accumulator with color similarity masking."""
        mask = _color_similarity_mask(warped, ref_frame, color_threshold)
        weighted_sum += w * mask * warped
        weight_sum += w * mask

    def _detect_scene_changes(self, frames: torch.Tensor, threshold: float):
        """Detect scene cuts by inter-frame MSE. Returns set of boundary indices.
        If frame i is in the set, frames i-1 and i belong to different scenes."""
        B = frames.shape[0]
        boundaries = set()
        for i in range(1, B):
            mse = torch.mean((frames[i] - frames[i - 1]) ** 2).item()
            if mse > threshold:
                boundaries.add(i)
        return boundaries

    def _same_scene(self, i: int, j: int, boundaries: set) -> bool:
        """Check if frames i and j belong to the same scene (no cut between them)."""
        lo, hi = min(i, j), max(i, j)
        for b in range(lo + 1, hi + 1):
            if b in boundaries:
                return False
        return True

    def denoise(self, images: torch.Tensor, window_size: int, weight_decay: float,
                flow_model: str, flow_iterations: int, color_threshold: float,
                scene_threshold: float, batch_size: int = 1,
                precision: str = "bf16", flow_scale: float = 1.0):

        B, H, W, C = images.shape
        device = images.device
        compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        flow_scale_f = max(0.3, min(1.0, float(flow_scale)))

        # BHWC -> BCHW, keep on CPU to save GPU memory
        frames = images.permute(0, 3, 1, 2)

        # Detect scene cuts on CPU
        boundaries = self._detect_scene_changes(frames, scene_threshold)
        if boundaries:
            print(f"[TemporalFlowAverage] Scene cuts detected at frames: "
                  f"{sorted(boundaries)}")

        # Output stays on CPU, results written per-frame
        output = torch.zeros_like(frames)
        weight_maps = torch.zeros(B, 1, H, W)

        pbar = comfy.utils.ProgressBar(B)

        if flow_model == "memfof":
            self._denoise_memfof(frames, B, H, W, window_size,
                                 weight_decay, flow_iterations,
                                 color_threshold, boundaries,
                                 output, weight_maps, compute_device,
                                 batch_size, pbar,
                                 precision=precision,
                                 flow_scale=flow_scale_f)
        else:
            self._denoise_raft(frames, B, H, W, window_size, weight_decay,
                               flow_model, flow_iterations,
                               color_threshold, boundaries,
                               output, weight_maps, compute_device, pbar,
                               precision=precision)

        result = output.permute(0, 2, 3, 1).clamp(0, 1).to(device)

        # Normalize weight map for visualization
        w_min = weight_maps.min()
        w_max = weight_maps.max()
        weight_vis = (weight_maps - w_min) / (w_max - w_min + 1e-8)
        weight_vis = weight_vis.expand(-1, 3, -1, -1).permute(0, 2, 3, 1).to(device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (result, weight_vis)

    def _denoise_memfof(self, frames, B, H, W, window_size,
                        weight_decay, flow_iterations, color_threshold,
                        boundaries, output, weight_maps, compute_device,
                        batch_size=1, pbar=None,
                        precision="bf16", flow_scale=1.0):
        """MEMFOF: 3-frame triplet with batched inference.
        frames/output/weight_maps are on CPU. Only working tensors go to GPU."""
        if pbar is not None:
            pbar.update_absolute(0, B)

        model = _get_memfof_model(compute_device)

        amp_enabled = (precision == "bf16" and compute_device.type == "cuda")

        # MEMFOF expects spatial dims to be multiples of 32
        if flow_scale < 1.0:
            fh = max(32, (int(H * flow_scale) // 32) * 32)
            fw = max(32, (int(W * flow_scale) // 32) * 32)
        else:
            fh, fw = H, W

        # Process frames in chunks
        tbar = tqdm(range(0, B, batch_size), desc="TemporalFlowAverage",
                    total=(B + batch_size - 1) // batch_size, unit="chunk")
        for chunk_start in tbar:
            chunk_end = min(chunk_start + batch_size, B)
            chunk_size = chunk_end - chunk_start
            tbar.set_postfix(frame=f"{chunk_end}/{B}")

            # GPU sliding-window buffer: upload chunk + window padding once per
            # chunk so triplet construction and neighbor warping can slice from
            # GPU memory instead of doing redundant H2D transfers.
            buf_start = max(0, chunk_start - window_size)
            buf_end = min(B, chunk_end + window_size)
            gpu_buf = frames[buf_start:buf_end].to(compute_device)

            # chunk_frames is a view into gpu_buf (zero copy)
            chunk_frames = gpu_buf[chunk_start - buf_start:chunk_end - buf_start]
            weighted_sums = chunk_frames.clone()
            weight_sums = torch.ones(chunk_size, 1, H, W, device=compute_device)

            for offset in range(1, window_size + 1):
                w = weight_decay ** offset

                # Collect valid triplets for this offset
                triplet_list = []
                meta = []  # (chunk_idx, has_prev, has_next, prev_idx, next_idx)

                for k, i in enumerate(range(chunk_start, chunk_end)):
                    has_prev = (i - offset) >= 0 and self._same_scene(i, i - offset, boundaries)
                    has_next = (i + offset) < B and self._same_scene(i, i + offset, boundaries)

                    if not has_prev and not has_next:
                        continue

                    prev_idx = i - offset if has_prev else i
                    next_idx = i + offset if has_next else i

                    triplet_list.append(torch.stack([
                        gpu_buf[prev_idx - buf_start],
                        gpu_buf[i - buf_start],
                        gpu_buf[next_idx - buf_start],
                    ], dim=0))
                    meta.append((k, has_prev, has_next, prev_idx, next_idx))

                if not triplet_list:
                    continue

                # Batched MEMFOF inference (triplets already on GPU via gpu_buf)
                batch_triplets = torch.stack(triplet_list, dim=0)
                batch_triplets = (batch_triplets * 255.0).clamp(0, 255)

                # Optional: downsample triplet for low-resolution flow
                if flow_scale < 1.0:
                    Nb, Tb = batch_triplets.shape[:2]
                    triplet_in = F.interpolate(
                        batch_triplets.flatten(0, 1),
                        size=(fh, fw), mode='bilinear', align_corners=False
                    ).unflatten(0, (Nb, Tb))
                else:
                    triplet_in = batch_triplets

                with torch.no_grad(), torch.autocast(
                        device_type=compute_device.type,
                        dtype=torch.bfloat16,
                        enabled=amp_enabled):
                    out_dict = model(triplet_in, iters=flow_iterations)

                flow_lr = out_dict["flow"][-1].float()  # [N, 2, 2, fh, fw]
                del batch_triplets, triplet_in, out_dict

                # Optional: upsample flow back to full resolution and rescale magnitudes
                if flow_scale < 1.0:
                    Nf = flow_lr.shape[0]
                    flow_flat = flow_lr.view(Nf * 2, 2, fh, fw)
                    flow_full = F.interpolate(
                        flow_flat, size=(H, W),
                        mode='bilinear', align_corners=False)
                    flow_full[:, 0] = flow_full[:, 0] * (float(W) / float(fw))
                    flow_full[:, 1] = flow_full[:, 1] * (float(H) / float(fh))
                    flow_fields = flow_full.view(Nf, 2, 2, H, W)
                    del flow_lr, flow_flat, flow_full
                else:
                    flow_fields = flow_lr

                # Distribute flows and accumulate per frame
                for j, (k, has_prev, has_next, prev_idx, next_idx) in enumerate(meta):
                    bwd_flow = flow_fields[j:j + 1, 0]
                    fwd_flow = flow_fields[j:j + 1, 1]
                    ref_frame = chunk_frames[k:k + 1]

                    if has_prev:
                        # Slice neighbor from GPU buffer (no H2D)
                        neighbor = gpu_buf[prev_idx - buf_start:prev_idx - buf_start + 1]
                        warped = _warp_with_flow(neighbor, bwd_flow)
                        self._accumulate_warped(warped, ref_frame, w,
                                                color_threshold,
                                                weighted_sums[k:k + 1],
                                                weight_sums[k:k + 1])
                        del warped

                    if has_next:
                        neighbor = gpu_buf[next_idx - buf_start:next_idx - buf_start + 1]
                        warped = _warp_with_flow(neighbor, fwd_flow)
                        self._accumulate_warped(warped, ref_frame, w,
                                                color_threshold,
                                                weighted_sums[k:k + 1],
                                                weight_sums[k:k + 1])
                        del warped

                del flow_fields

            # Write chunk results to CPU
            output[chunk_start:chunk_end] = (weighted_sums / (weight_sums + 1e-8)).cpu()
            weight_maps[chunk_start:chunk_end] = weight_sums.cpu()
            del gpu_buf, chunk_frames, weighted_sums, weight_sums
            if pbar is not None:
                pbar.update(chunk_size)

        # Model stays on GPU (cached via _get_memfof_model)

    def _denoise_raft(self, frames, B, H, W, window_size, weight_decay,
                      flow_model, flow_iterations, color_threshold,
                      boundaries, output, weight_maps, compute_device, pbar=None,
                      precision="bf16"):
        """RAFT fallback: pair-based flow with forward-backward occlusion check.
        frames/output/weight_maps are on CPU. Only working tensors go to GPU."""
        from torchvision.models.optical_flow import (raft_large, raft_small,
            Raft_Large_Weights, Raft_Small_Weights)

        amp_enabled = (precision == "bf16" and compute_device.type == "cuda")

        # Pad to multiple of 8 for RAFT
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8

        if flow_model == "raft_large":
            model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(compute_device)
        else:
            model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(compute_device)
        model.eval()

        for i in tqdm(range(B), desc="TemporalFlowAverage (RAFT)", unit="frame"):
            ref_cpu = frames[i:i + 1]
            if pad_h > 0 or pad_w > 0:
                ref_padded = F.pad(ref_cpu, [0, pad_w, 0, pad_h], mode='reflect')
            else:
                ref_padded = ref_cpu

            ref_gpu = ref_padded.to(compute_device)
            ref_orig = ref_cpu.to(compute_device)
            weighted_sum = ref_orig.clone()
            weight_sum = torch.ones(1, 1, H, W, device=compute_device)

            for offset in range(1, window_size + 1):
                w = weight_decay ** offset

                for j in [i - offset, i + offset]:
                    if j < 0 or j >= B:
                        continue
                    if not self._same_scene(i, j, boundaries):
                        continue

                    nb_cpu = frames[j:j + 1]
                    if pad_h > 0 or pad_w > 0:
                        nb_padded = F.pad(nb_cpu, [0, pad_w, 0, pad_h], mode='reflect')
                    else:
                        nb_padded = nb_cpu
                    nb_gpu = nb_padded.to(compute_device)

                    ref_255 = (ref_gpu * 255.0).clamp(0, 255)
                    nb_255 = (nb_gpu * 255.0).clamp(0, 255)

                    with torch.no_grad(), torch.autocast(
                            device_type=compute_device.type,
                            dtype=torch.bfloat16,
                            enabled=amp_enabled):
                        flow_fwd = model(ref_255, nb_255,
                                         num_flow_updates=flow_iterations)[-1]
                        flow_bwd = model(nb_255, ref_255,
                                         num_flow_updates=flow_iterations)[-1]

                    flow_fwd = flow_fwd[:, :, :H, :W].float()
                    flow_bwd = flow_bwd[:, :, :H, :W].float()

                    nb_orig = nb_cpu.to(compute_device)
                    warped = _warp_with_flow(nb_orig, flow_fwd)

                    color_mask = _color_similarity_mask(warped, ref_orig,
                                                        color_threshold)
                    occ_mask = _forward_backward_occlusion_mask(flow_fwd, flow_bwd)
                    mask = color_mask * occ_mask

                    weighted_sum += w * mask * warped
                    weight_sum += w * mask
                    del nb_gpu, nb_orig, warped, flow_fwd, flow_bwd, mask

            output[i:i + 1] = (weighted_sum / (weight_sum + 1e-8)).cpu()
            weight_maps[i:i + 1] = weight_sum.cpu()
            del ref_gpu, ref_orig, weighted_sum, weight_sum
            if pbar is not None:
                pbar.update(1)

        del model


def _turbo_colormap(t: torch.Tensor) -> torch.Tensor:
    """Attempt to create a Turbo-like colormap for tensor values in [0,1].
    Input: (B,1,H,W) or (B,H,W,1). Output: same spatial dims with 3 channels.
    Returns BHWC RGB [0,1]."""
    # Ensure BHWC with 1 channel
    if t.dim() == 4 and t.shape[1] == 1:
        t = t.permute(0, 2, 3, 1)  # BCHW -> BHWC
    t = t.squeeze(-1)  # (B, H, W)
    t = t.clamp(0, 1)

    # Piecewise-linear turbo approximation (blue -> cyan -> yellow -> red)
    r = (1.5 - torch.abs(t * 4.0 - 3.0)).clamp(0, 1)
    g = (1.5 - torch.abs(t * 4.0 - 2.0)).clamp(0, 1)
    b = (1.5 - torch.abs(t * 4.0 - 1.0)).clamp(0, 1)

    return torch.stack([r, g, b], dim=-1)  # (B, H, W, 3)


def _visualize_noise(noise_bhwc: torch.Tensor, amplify: float,
                     mode: str) -> torch.Tensor:
    """Visualize noise tensor (BHWC, zero-centered) in the chosen mode.
    Returns BHWC [0,1] IMAGE."""
    if mode == "gray":
        return (noise_bhwc * amplify + 0.5).clamp(0, 1)

    elif mode == "heatmap":
        # Per-frame auto-normalized magnitude -> turbo colormap
        mag = torch.mean(noise_bhwc.abs(), dim=-1, keepdim=True)  # (B,H,W,1)
        B = mag.shape[0]
        for i in range(B):
            fmax = mag[i].max()
            if fmax > 1e-8:
                mag[i] = mag[i] / fmax
        return _turbo_colormap(mag)

    elif mode == "signed":
        # Luminance of noise -> positive=red, negative=blue, magnitude=brightness
        lum = torch.mean(noise_bhwc, dim=-1)  # (B, H, W)
        B = lum.shape[0]
        # Per-frame normalize
        for i in range(B):
            amax = lum[i].abs().max()
            if amax > 1e-8:
                lum[i] = lum[i] / amax
        pos = lum.clamp(min=0)  # positive noise
        neg = (-lum).clamp(min=0)  # negative noise (abs)
        r = pos
        g = torch.zeros_like(lum)
        b = neg
        return torch.stack([r, g, b], dim=-1).clamp(0, 1)  # (B,H,W,3)

    else:
        return (noise_bhwc * amplify + 0.5).clamp(0, 1)


class ExtractNoise:
    """Decompose the difference between original and clean into chroma and luma noise."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original": ("IMAGE",),
                "clean": ("IMAGE",),
                "noise_amplify": ("FLOAT", {
                    "default": 5.0, "min": 1.0, "max": 50.0, "step": 0.5,
                    "tooltip": "Amplification factor for noise preview visualization"
                }),
                "color_space": (["HSV", "YCbCr", "LAB"], {
                    "default": "YCbCr",
                    "tooltip": "Color space for separating luma and chroma noise"
                }),
                "noise_preview": (["heatmap", "signed", "gray"], {
                    "default": "heatmap",
                    "tooltip": "Noise preview mode: heatmap (magnitude->color), signed (red=+/blue=-), gray (classic 0.5-centered)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("noise_total", "noise_chroma", "noise_luma")
    FUNCTION = "extract"
    CATEGORY = "FlowDenoise"
    DESCRIPTION = "Extract and separate noise into chroma and luma components for selective denoising."

    def extract(self, original: torch.Tensor, clean: torch.Tensor,
                noise_amplify: float, color_space: str,
                noise_preview: str = "heatmap"):
        # Both inputs are BHWC
        noise_rgb = original - clean  # BHWC

        # Convert to BCHW for color space operations
        orig_bchw = original.permute(0, 3, 1, 2)
        clean_bchw = clean.permute(0, 3, 1, 2)

        # Convert both to chosen color space
        orig_cs = _convert_color_space(orig_bchw, color_space)
        clean_cs = _convert_color_space(clean_bchw, color_space)

        # Noise in color space
        noise_cs = orig_cs - clean_cs

        # Luma channel is channel 0 in all supported spaces (H/Y/L)
        noise_luma_cs = torch.zeros_like(noise_cs)
        noise_luma_cs[:, 0:1] = noise_cs[:, 0:1]

        # Chroma noise: the chroma channels differ
        noise_chroma_cs = torch.zeros_like(noise_cs)
        noise_chroma_cs[:, 1:] = noise_cs[:, 1:]

        # --- noise_total visualization ---
        noise_total = _visualize_noise(noise_rgb, noise_amplify, noise_preview)

        # --- chroma/luma visualization (same mode) ---
        noise_luma_mono = noise_luma_cs[:, 0:1].expand(-1, 3, -1, -1).permute(0, 2, 3, 1)
        noise_luma_vis = _visualize_noise(noise_luma_mono, noise_amplify, noise_preview)

        noise_chroma_bhwc = noise_chroma_cs.permute(0, 2, 3, 1)
        noise_chroma_vis = _visualize_noise(noise_chroma_bhwc, noise_amplify, noise_preview)

        return (noise_total, noise_chroma_vis, noise_luma_vis)


def _gaussian_kernel_2d(kernel_size: int, sigma: float,
                         device: torch.device) -> torch.Tensor:
    """Create a 2D Gaussian kernel for spatial filtering."""
    ax = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel


def _spatial_gaussian_blur(img: torch.Tensor, kernel_size: int,
                            sigma: float) -> torch.Tensor:
    """Apply Gaussian blur to BCHW tensor, per-channel."""
    kernel = _gaussian_kernel_2d(kernel_size, sigma, img.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1,1,K,K]
    C = img.shape[1]
    kernel = kernel.expand(C, -1, -1, -1)  # [C,1,K,K]
    pad = kernel_size // 2
    return F.conv2d(img, kernel, padding=pad, groups=C)


class SelectiveDenoise:
    """Selectively blend original and clean in a chosen color space,
    with independent control over luma and chroma channels."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original": ("IMAGE",),
                "clean": ("IMAGE",),
                "chroma_strength": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How much to denoise chroma (0=keep original, 1=fully clean)"
                }),
                "luma_strength": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How much to denoise luma (0=keep original, 1=fully clean)"
                }),
                "color_space": (["YCbCr", "HSV", "LAB"], {
                    "default": "YCbCr",
                    "tooltip": "Color space for luma/chroma separation"
                }),
                "clamp_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clamp output to [0,1] range"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("denoised",)
    FUNCTION = "denoise"
    CATEGORY = "FlowDenoise"
    DESCRIPTION = "Selectively denoise by blending original and clean in a color space with independent luma/chroma control."

    def denoise(self, original: torch.Tensor, clean: torch.Tensor,
                chroma_strength: float, luma_strength: float,
                color_space: str, clamp_output: bool):

        # BHWC -> BCHW
        orig_bchw = original.permute(0, 3, 1, 2)
        clean_bchw = clean.permute(0, 3, 1, 2)

        # Convert to color space
        orig_cs = _convert_color_space(orig_bchw, color_space)
        clean_cs = _convert_color_space(clean_bchw, color_space)

        # Blend luma (channel 0) and chroma (channels 1,2) independently
        result_cs = orig_cs.clone()
        result_cs[:, 0:1] = torch.lerp(orig_cs[:, 0:1], clean_cs[:, 0:1], luma_strength)
        result_cs[:, 1:] = torch.lerp(orig_cs[:, 1:], clean_cs[:, 1:], chroma_strength)

        # Convert back to RGB
        result_rgb = _convert_back_to_rgb(result_cs, color_space)

        # BCHW -> BHWC
        result = result_rgb.permute(0, 2, 3, 1)
        if clamp_output:
            result = result.clamp(0, 1)

        return (result,)

