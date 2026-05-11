"""Score aggregation aligned with dense BAS evaluation (``map_mine`` path).

Mirrors displacement-aware scoring, dense per-frame fusion averaging across
overlapping clips, then optional ``soft_non_maximum_suppression``.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from custom_ballspotting.data import VideoClip
from custom_ballspotting.model.tdeed import action_probabilities


def soft_non_maximum_suppression(
    scores: np.ndarray,
    class_window: int | list[int] = 1,
    threshold: float = 0.01,
    *,
    show_progress: bool | None = None,
) -> np.ndarray:
    """Port of dudek ``utils.common.soft_non_maximum_suppression`` (NumPy only)."""
    num_frames, num_classes = scores.shape
    suppressed_scores = np.zeros_like(scores)

    if isinstance(class_window, int):
        class_window = [class_window] * num_classes

    use_tqdm = show_progress
    if use_tqdm is None:
        use_tqdm = sys.stderr.isatty()

    iterator = range(num_classes)
    if use_tqdm:
        iterator = tqdm(iterator, desc="val_map soft-NMS")

    for c in iterator:
        window = class_window[c]
        s = scores[:, c].astype(np.float64, copy=True)
        frames = np.arange(num_frames)
        processed = np.zeros(num_frames, dtype=bool)
        output_s = np.zeros(num_frames, dtype=np.float32)

        while True:
            s_masked = s.copy()
            s_masked[processed] = -np.inf

            e1_idx = int(np.argmax(s_masked))
            e1_score = float(s_masked[e1_idx])

            if e1_score < threshold or e1_score == -np.inf:
                break

            output_s[e1_idx] = e1_score
            processed[e1_idx] = True

            distances = np.abs(frames - e1_idx)
            within_window = (distances <= window) & (~processed)
            suppression_factor = (distances[within_window] ** 2) / (window**2 + 1e-8)
            s[within_window] *= suppression_factor

        suppressed_scores[:, c] = output_s

    return suppressed_scores


def _displace_probabilities_timeline(probs: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    """Move each timestep's probability vector by its predicted displacement."""
    t_len, num_classes = probs.shape
    aux = np.zeros_like(probs)
    for t in range(t_len):
        d = int(np.round(float(displacement[t])))
        tgt = max(0, min(t_len - 1, t - d))
        aux[tgt] = np.maximum(aux[tgt], probs[t])
    return aux


def _linear_interpolate_nan_columns(aligned: np.ndarray) -> np.ndarray:
    """Column-wise ``np.interp`` over NaNs (dudek ``linear_interpolate_row`` per column)."""
    out = aligned.copy()
    n_rows = out.shape[0]
    idx = np.arange(n_rows, dtype=np.float64)
    for col in range(out.shape[1]):
        col_data = out[:, col]
        mask = ~np.isnan(col_data)
        if not np.any(mask):
            continue
        out[:, col] = np.interp(idx, idx[mask], col_data[mask])
    return out


def align_clip_predictions_to_video_span(
    clip: VideoClip,
    predictions_matrix: np.ndarray,
) -> np.ndarray:
    """Map clip timestep predictions onto ``[start_frame..end_frame]`` (linear fill)."""
    start_frame = clip.frames[0].original_video_frame_nr
    end_frame = clip.frames[-1].original_video_frame_nr
    span = end_frame - start_frame + 1
    aligned = np.full((span, predictions_matrix.shape[1]), np.nan, dtype=np.float32)
    for i, frame in enumerate(clip.frames):
        idx = frame.original_video_frame_nr - start_frame
        aligned[idx] = predictions_matrix[i]
    return _linear_interpolate_nan_columns(aligned)


def dudek_style_scores_matrix(
    model: torch.nn.Module,
    clips: list[VideoClip],
    loader: DataLoader,
    device: str,
    *,
    num_classes_with_background: int,
) -> np.ndarray:
    """Average overlapping clip predictions into a dense per-frame score matrix."""
    last_frame = max(frame.original_video_frame_nr for clip in clips for frame in clip.frames)
    num_frames = last_frame + 1
    scores_matrix = np.zeros((num_frames, num_classes_with_background), dtype=np.float32)
    support_matrix = np.zeros((num_frames, num_classes_with_background), dtype=np.float32)

    clip_offset = 0
    model.eval()
    use_cuda = device == "cuda"
    with torch.no_grad():
        for batch in loader:
            clip_tensor = batch["clip_tensor"].to(device, non_blocking=use_cuda).float()
            with torch.amp.autocast(device_type=device, enabled=device == "cuda"):
                outputs = model(clip_tensor, inference=True)
                probs_tensor = action_probabilities(outputs)
                displacements = outputs["displacement"]
            probs = probs_tensor.detach().cpu().numpy()
            displ_np = displacements.detach().cpu().numpy()

            for b in range(probs.shape[0]):
                clip = clips[clip_offset + b]
                displaced = _displace_probabilities_timeline(probs[b], displ_np[b])
                aligned = align_clip_predictions_to_video_span(clip, displaced)
                start_frame = clip.frames[0].original_video_frame_nr
                end_frame = clip.frames[-1].original_video_frame_nr
                scores_matrix[start_frame : end_frame + 1] += aligned
                support_matrix[start_frame : end_frame + 1] += 1
            clip_offset += probs.shape[0]

    support_matrix[support_matrix == 0] = 1.0
    return scores_matrix / support_matrix
