import inspect
import json
import logging
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from custom_ballspotting.actions import (
    Action,
    NUM_ACTION_CLASSES,
    NUM_TEAM_ACTION_CLASSES,
    index_to_label,
)
from custom_ballspotting.checkpoints import read_checkpoint_metadata
from custom_ballspotting.data import (
    CustomTDeedDataset,
    VideoRecord,
)
from custom_ballspotting.model.tdeed import CustomTDeedModule

_logger = logging.getLogger(__name__)


DEFAULT_DECODE_THRESHOLDS: dict[str, float] = {
    Action.PASS.value: 0.20,
    Action.PASS_RECEIVED.value: 0.20,
    Action.FREE_KICK.value: 0.25,
    Action.GOAL_KICK.value: 0.25,
    Action.CORNER.value: 0.25,
    Action.THROW_IN.value: 0.25,
    Action.RECOVERY.value: 0.25,
    Action.TACKLE.value: 0.30,
    Action.INTERCEPTION.value: 0.40,
    Action.BALL_OUT_OF_PLAY.value: 0.45,
    Action.CLEARANCE.value: 0.40,
    Action.TAKE_ON.value: 0.40,
    Action.SUBSTITUTION.value: 0.50,
    Action.BLOCK.value: 0.45,
    Action.AERIAL_DUEL.value: 0.40,
    Action.SHOT.value: 0.40,
    Action.SAVE.value: 0.45,
    Action.FOUL.value: 0.50,
    Action.GOAL.value: 0.50,
}

DEFAULT_DECODE_NMS_WINDOW_FRAMES: dict[str, int] = {
    # These are duplicate-decoding windows, not evaluator match tolerances.
    # Keep fast ball actions small so genuine quick sequences can survive.
    Action.PASS.value: 4,
    Action.PASS_RECEIVED.value: 4,
    Action.FREE_KICK.value: 8,
    Action.GOAL_KICK.value: 8,
    Action.CORNER.value: 8,
    Action.THROW_IN.value: 8,
    Action.RECOVERY.value: 8,
    Action.TACKLE.value: 8,
    Action.INTERCEPTION.value: 6,
    Action.BALL_OUT_OF_PLAY.value: 12,
    Action.CLEARANCE.value: 8,
    Action.TAKE_ON.value: 8,
    Action.SUBSTITUTION.value: 40,
    Action.BLOCK.value: 8,
    Action.AERIAL_DUEL.value: 6,
    Action.SHOT.value: 8,
    Action.SAVE.value: 8,
    Action.FOUL.value: 12,
    Action.GOAL.value: 12,
}


def _coerce_infer_param(
    name: str,
    explicit: object,
    train_cfg: dict | None,
    fallback: object,
) -> object:
    if explicit is not None:
        return explicit
    if train_cfg and name in train_cfg and train_cfg[name] is not None:
        return train_cfg[name]
    return fallback


def resolve_infer_video_params(
    model_checkpoint_path: str,
    *,
    clip_frames_count: int | None = None,
    overlap: int | None = None,
    stride: int | None = None,
    frame_target_width: int | None = None,
    frame_target_height: int | None = None,
    features_model_name: str | None = None,
    temporal_shift_mode: str | None = None,
    n_layers: int | None = None,
    sgp_ks: int | None = None,
    sgp_k: int | None = None,
    gaussian_blur_kernel_size: int | None = None,
    val_batch_size: int | None = None,
    inference_threshold: float | None = None,
    decode_thresholds: dict[str, float] | None = None,
    decode_nms_window_frames: dict[str, int] | None = None,
    use_displacement_refinement: bool | None = None,
    displacement_max_frames: int | None = None,
    extract_frames: bool | None = None,
    device: str | None = None,
) -> dict:
    """
    Merge explicit args with ``*.metadata.json`` from training (if present),
    then package defaults. Used by :func:`infer_video` so the model matches
    the trained architecture and clip geometry.
    """
    meta = read_checkpoint_metadata(model_checkpoint_path)
    if meta is not None:
        n_saved = meta.get("num_action_classes")
        if n_saved is not None and int(n_saved) != NUM_ACTION_CLASSES:
            raise ValueError(
                f"Checkpoint expects num_action_classes={n_saved} (see metadata), "
                f"but this install has NUM_ACTION_CLASSES={NUM_ACTION_CLASSES}. "
                "Use a checkpoint trained with the same Action enum / actions.py, or "
                "align the code with the checkpoint."
            )
        n_team_saved = meta.get("num_team_action_classes")
        if n_team_saved is not None and int(n_team_saved) != NUM_TEAM_ACTION_CLASSES:
            raise ValueError(
                f"Checkpoint expects num_team_action_classes={n_team_saved} (see metadata), "
                f"but this install has NUM_TEAM_ACTION_CLASSES={NUM_TEAM_ACTION_CLASSES}. "
                "Use a checkpoint trained with the same Action enum / actions.py."
            )
    else:
        _logger.warning(
            "No sibling .metadata.json for checkpoint %s; using only explicit args "
            "and defaults. Prefer checkpoints saved by this package next to *.metadata.json "
            "so clip and model architecture match training.",
            model_checkpoint_path,
        )

    train_cfg = (meta or {}).get("config") if isinstance((meta or {}).get("config"), dict) else None

    device_resolved = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    extract_resolved = True if extract_frames is None else extract_frames
    threshold_resolved = 0.2 if inference_threshold is None else inference_threshold
    thresholds_resolved = dict(DEFAULT_DECODE_THRESHOLDS)
    thresholds_resolved.update(decode_thresholds or {})
    nms_windows_resolved = dict(DEFAULT_DECODE_NMS_WINDOW_FRAMES)
    nms_windows_resolved.update(decode_nms_window_frames or {})
    use_displacement_resolved = (
        True if use_displacement_refinement is None else use_displacement_refinement
    )
    displacement_max_resolved = 4 if displacement_max_frames is None else displacement_max_frames

    return {
        "clip_frames_count": int(
            _coerce_infer_param("clip_frames_count", clip_frames_count, train_cfg, 100)
        ),
        "overlap": int(_coerce_infer_param("overlap", overlap, train_cfg, 88)),
        "stride": int(_coerce_infer_param("stride", stride, train_cfg, 2)),
        "frame_target_width": int(
            _coerce_infer_param("frame_target_width", frame_target_width, train_cfg, 1280)
        ),
        "frame_target_height": int(
            _coerce_infer_param("frame_target_height", frame_target_height, train_cfg, 720)
        ),
        "features_model_name": str(
            _coerce_infer_param(
                "features_model_name", features_model_name, train_cfg, "regnety_008"
            )
        ),
        "temporal_shift_mode": str(
            _coerce_infer_param("temporal_shift_mode", temporal_shift_mode, train_cfg, "gsf")
        ),
        "n_layers": int(_coerce_infer_param("n_layers", n_layers, train_cfg, 2)),
        "sgp_ks": int(_coerce_infer_param("sgp_ks", sgp_ks, train_cfg, 9)),
        "sgp_k": int(_coerce_infer_param("sgp_k", sgp_k, train_cfg, 4)),
        "gaussian_blur_kernel_size": int(
            _coerce_infer_param(
                "gaussian_blur_kernel_size", gaussian_blur_kernel_size, train_cfg, 5
            )
        ),
        "val_batch_size": int(_coerce_infer_param("val_batch_size", val_batch_size, train_cfg, 1)),
        "inference_threshold": float(threshold_resolved),
        "decode_thresholds": {str(k): float(v) for k, v in thresholds_resolved.items()},
        "decode_nms_window_frames": {str(k): int(v) for k, v in nms_windows_resolved.items()},
        "use_displacement_refinement": bool(use_displacement_resolved),
        "displacement_max_frames": int(displacement_max_resolved),
        "extract_frames": bool(extract_resolved),
        "device": device_resolved,
    }


def infer_video(
    video_path: str,
    model_checkpoint_path: str,
    output_path: str | None = None,
    clip_frames_count: int | None = None,
    overlap: int | None = None,
    stride: int | None = None,
    frame_target_width: int | None = None,
    frame_target_height: int | None = None,
    features_model_name: str | None = None,
    temporal_shift_mode: str | None = None,
    n_layers: int | None = None,
    sgp_ks: int | None = None,
    sgp_k: int | None = None,
    gaussian_blur_kernel_size: int | None = None,
    val_batch_size: int | None = None,
    inference_threshold: float | None = None,
    decode_thresholds: dict[str, float] | None = None,
    decode_nms_window_frames: dict[str, int] | None = None,
    use_displacement_refinement: bool | None = None,
    displacement_max_frames: int | None = None,
    extract_frames: bool | None = None,
    device: str | None = None,
    model: "CustomTDeedModule | None" = None,
    num_workers: int = 0,
    frame_write_workers: int = 8,
) -> dict:
    """Run ball-action spotting inference on a video and return predictions.

    Every extracted frame is fused into dense per-frame logits: clipping uses sliding
    windows, and overlaps are averaged in :func:`score_video`. Larger ``val_batch_size``
    only batches clips for throughput and does **not** drop frames.

    Videos shorter than one temporal clip (``clip_frames_count``) are padded for the model;
    padded tail timesteps are ignored when accumulating scores.

    Parameters
    ----------
    output_path:
        If provided, write the result JSON to this path. If ``None`` (default),
        skip the file write and only return the dict. Useful for API servers.
    model:
        A pre-loaded, warmed-up ``CustomTDeedModule`` already on the target device.
        When given, model loading is skipped entirely — essential for hot-model
        servers where one GPU process handles many requests. When ``None`` (default),
        the model is loaded from ``model_checkpoint_path`` as before.
    num_workers:
        Number of DataLoader worker processes for prefetching clips while the GPU
        runs the current batch. On a single-GPU multi-CPU machine, ``2`` overlaps
        data loading with GPU compute. Default ``0`` is safe and uses the
        per-clip ``ThreadPoolExecutor`` inside ``TDeedClip.from_clip`` instead.
    frame_write_workers:
        Number of threads for parallel frame resize+write during extraction.
        OpenCV C-level calls release the GIL so threads give real CPU parallelism.
        Default ``8`` saturates typical multi-core CPUs without spawning too many
        threads.
    """
    p = resolve_infer_video_params(
        model_checkpoint_path,
        clip_frames_count=clip_frames_count,
        overlap=overlap,
        stride=stride,
        frame_target_width=frame_target_width,
        frame_target_height=frame_target_height,
        features_model_name=features_model_name,
        temporal_shift_mode=temporal_shift_mode,
        n_layers=n_layers,
        sgp_ks=sgp_ks,
        sgp_k=sgp_k,
        gaussian_blur_kernel_size=gaussian_blur_kernel_size,
        val_batch_size=val_batch_size,
        inference_threshold=inference_threshold,
        decode_thresholds=decode_thresholds,
        decode_nms_window_frames=decode_nms_window_frames,
        use_displacement_refinement=use_displacement_refinement,
        displacement_max_frames=displacement_max_frames,
        extract_frames=extract_frames,
        device=device,
    )

    video = VideoRecord(video_path=os.path.abspath(video_path), annotations=[])
    if p["extract_frames"] or not os.path.exists(video.frames_path):
        video.extract_frames(
            stride=p["stride"],
            target_width=p["frame_target_width"],
            target_height=p["frame_target_height"],
            save_all=True,
            write_workers=frame_write_workers,
        )
    clips = []
    clip_len = int(p["clip_frames_count"])
    overlap_frames = int(p["overlap"])
    for continuous_clip in video.get_clips(accepted_gap=p["stride"]):
        clips.extend(continuous_clip.split(clip_len, overlap_frames, pad_if_shorter=True))
    if not clips:
        raise ValueError(
            "No inference clips could be formed (video may have produced zero decoded frames)."
        )
    dataset = CustomTDeedDataset(clips, displacement_radius=0)
    loader = DataLoader(
        dataset,
        batch_size=p["val_batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=p["device"] == "cuda" and num_workers > 0,
    )

    if model is None:
        model = CustomTDeedModule(
            clip_len=p["clip_frames_count"],
            num_actions=NUM_ACTION_CLASSES,
            n_layers=p["n_layers"],
            sgp_ks=p["sgp_ks"],
            sgp_k=p["sgp_k"],
            features_model_name=p["features_model_name"],
            temporal_shift_mode=p["temporal_shift_mode"],
            gaussian_blur_ks=p["gaussian_blur_kernel_size"],
        )
        model.load_all(model_checkpoint_path)
        model.to(p["device"])
        model.eval()

    scores, displacements = score_video(
        model,
        clips,
        loader,
        device=p["device"],
        return_displacements=True,
    )
    fps_infer = float(video.metadata_fps)
    if not math.isfinite(fps_infer) or fps_infer <= 0:
        fps_infer = 25.0
    predictions = scores_to_predictions(
        scores,
        fps=fps_infer,
        threshold=p["inference_threshold"],
        decode_thresholds=p["decode_thresholds"],
        decode_nms_window_frames=p["decode_nms_window_frames"],
        displacements=displacements if p["use_displacement_refinement"] else None,
        displacement_max_frames=p["displacement_max_frames"],
    )
    result = {
        "video_path": video.video_path,
        "fps": fps_infer,
        "predictions": predictions,
    }

    if output_path is not None:
        out_parent = os.path.dirname(os.path.abspath(output_path))
        if out_parent:
            os.makedirs(out_parent, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

    return result


def infer_video_param_names() -> frozenset[str]:
    """Parameter names accepted by :func:`infer_video` (for CLI / config filtering)."""
    return frozenset(inspect.signature(infer_video).parameters)


def score_video(model, clips, loader, device: str, return_displacements: bool = False):
    if not clips:
        raise ValueError("No clips generated for inference.")
    last_frame = max(frame.original_video_frame_nr for clip in clips for frame in clip.frames)
    scores = np.zeros((last_frame + 1, NUM_TEAM_ACTION_CLASSES + 1), dtype=np.float32)
    displacement_sums = np.zeros(last_frame + 1, dtype=np.float32)
    counts = np.zeros((last_frame + 1, 1), dtype=np.float32)

    clip_offset = 0
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc="scoring"):
            use_cuda = device == "cuda"
            clip_tensor = batch["clip_tensor"].to(device, non_blocking=use_cuda).float()
            with torch.amp.autocast(device_type=device, enabled=device == "cuda"):
                outputs = model(clip_tensor, inference=True)
                probs = torch.softmax(outputs["logits"], dim=-1).detach().cpu().numpy()
                displacements = outputs["displacement"].detach().cpu().numpy()
            for batch_idx in range(probs.shape[0]):
                clip = clips[clip_offset + batch_idx]
                cap = clip.logits_aggregate_timesteps
                span = len(clip.frames) if cap is None else cap
                for frame_idx, frame in enumerate(clip.frames):
                    if frame_idx >= span:
                        break
                    original_frame = frame.original_video_frame_nr
                    scores[original_frame] += probs[batch_idx, frame_idx]
                    displacement_sums[original_frame] += displacements[batch_idx, frame_idx]
                    counts[original_frame] += 1
            clip_offset += probs.shape[0]
    averaged_scores = scores / np.maximum(counts, 1.0)
    if not return_displacements:
        return averaged_scores
    averaged_displacements = displacement_sums / np.maximum(counts[:, 0], 1.0)
    return averaged_scores, averaged_displacements


def scores_to_predictions(
    scores,
    fps: float,
    threshold: float,
    decode_thresholds: dict[str, float] | None = None,
    decode_nms_window_frames: dict[str, int] | None = None,
    displacements: np.ndarray | None = None,
    displacement_max_frames: int = 4,
):
    thresholds = dict(DEFAULT_DECODE_THRESHOLDS)
    thresholds.update(decode_thresholds or {})
    nms_windows = dict(DEFAULT_DECODE_NMS_WINDOW_FRAMES)
    nms_windows.update(decode_nms_window_frames or {})

    predictions = []
    for class_index in range(1, NUM_TEAM_ACTION_CLASSES + 1):
        result = index_to_label(class_index)
        if result is None:
            continue
        action, team = result
        class_scores = scores[:, class_index]
        min_score = max(threshold, thresholds.get(action.value, threshold))
        candidate_indices = local_peak_indices(class_scores, min_score)
        if candidate_indices.size == 0:
            continue
        candidates = refine_prediction_candidates(
            candidate_indices,
            class_scores,
            displacements,
            displacement_max_frames=displacement_max_frames,
            last_frame=scores.shape[0] - 1,
        )
        kept = non_maximum_suppression_candidates(
            candidates,
            window_frames=nms_windows.get(action.value, 6),
        )
        for frame_idx, confidence in kept:
            position = int(frame_idx / fps * 1000)
            predictions.append(
                {
                    "label": action.value,
                    "team": team.value,
                    "position": position,
                    "gameTime": format_game_time(position),
                    "confidence": confidence,
                }
            )
    predictions.sort(key=lambda item: item["position"])
    return predictions


def local_peak_indices(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Return one best frame per contiguous above-threshold score island."""
    candidate_indices = np.where(scores >= threshold)[0]
    if candidate_indices.size == 0:
        return candidate_indices

    peaks: list[int] = []
    start = int(candidate_indices[0])
    prev = start
    for raw_idx in candidate_indices[1:]:
        idx = int(raw_idx)
        if idx == prev + 1:
            prev = idx
            continue
        peaks.append(best_index_in_span(scores, start, prev))
        start = prev = idx
    peaks.append(best_index_in_span(scores, start, prev))
    return np.asarray(peaks, dtype=np.int64)


def best_index_in_span(scores: np.ndarray, start: int, end: int) -> int:
    span_scores = scores[start : end + 1]
    return int(start + np.argmax(span_scores))


def refine_prediction_candidates(
    peak_indices: np.ndarray,
    class_scores: np.ndarray,
    displacements: np.ndarray | None,
    *,
    displacement_max_frames: int,
    last_frame: int,
) -> list[tuple[int, float]]:
    candidates: list[tuple[int, float]] = []
    max_abs = max(0, int(displacement_max_frames))
    for raw_idx in peak_indices:
        peak_idx = int(raw_idx)
        refined_idx = peak_idx
        if displacements is not None and max_abs > 0:
            displacement = float(displacements[peak_idx])
            if math.isfinite(displacement) and abs(displacement) <= max_abs:
                refined_idx = int(round(peak_idx - displacement))
                refined_idx = max(0, min(last_frame, refined_idx))
        candidates.append((refined_idx, float(class_scores[peak_idx])))
    return candidates


def non_maximum_suppression_candidates(
    candidates: list[tuple[int, float]],
    window_frames: int,
) -> list[tuple[int, float]]:
    ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
    kept: list[tuple[int, float]] = []
    for frame_idx, confidence in ranked:
        if all(abs(frame_idx - kept_frame) > window_frames for kept_frame, _ in kept):
            kept.append((int(frame_idx), float(confidence)))
    return sorted(kept, key=lambda item: item[0])


def non_maximum_suppression(indices, scores, window_frames: int):
    indices = sorted(indices, key=lambda idx: scores[idx], reverse=True)
    kept: list[int] = []
    for idx in indices:
        if all(abs(idx - kept_idx) > window_frames for kept_idx in kept):
            kept.append(int(idx))
    return sorted(kept)


def format_game_time(position_ms: int) -> str:
    total_seconds = position_ms // 1000
    half = 1 if total_seconds < 45 * 60 else 2
    seconds_in_half = total_seconds if half == 1 else total_seconds - 45 * 60
    return f"{half} - {seconds_in_half // 60:02d}:{seconds_in_half % 60:02d}"
