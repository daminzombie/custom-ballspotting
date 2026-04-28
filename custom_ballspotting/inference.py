import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from custom_ballspotting.actions import (
    ACTION_CONFIGS,
    NUM_ACTION_CLASSES,
    index_to_label,
)
from custom_ballspotting.data import (
    CustomTDeedDataset,
    VideoRecord,
)
from custom_ballspotting.model.tdeed import CustomTDeedModule


def infer_video(
    video_path: str,
    model_checkpoint_path: str,
    output_path: str,
    clip_frames_count: int = 100,
    overlap: int = 88,
    stride: int = 2,
    frame_target_width: int = 1280,
    frame_target_height: int = 720,
    features_model_name: str = "regnety_008",
    temporal_shift_mode: str = "gsf",
    n_layers: int = 2,
    sgp_ks: int = 9,
    sgp_k: int = 4,
    val_batch_size: int = 1,
    inference_threshold: float = 0.2,
    extract_frames: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    video = VideoRecord(video_path=os.path.abspath(video_path), annotations=[])
    if extract_frames or not os.path.exists(video.frames_path):
        video.extract_frames(
            stride=stride,
            target_width=frame_target_width,
            target_height=frame_target_height,
            save_all=True,
        )
    clips = []
    for continuous_clip in video.get_clips(accepted_gap=stride):
        clips.extend(continuous_clip.split(clip_frames_count, overlap))
    dataset = CustomTDeedDataset(clips, displacement_radius=0)
    loader = DataLoader(dataset, batch_size=val_batch_size, shuffle=False)

    model = CustomTDeedModule(
        clip_len=clip_frames_count,
        num_actions=NUM_ACTION_CLASSES,
        n_layers=n_layers,
        sgp_ks=sgp_ks,
        sgp_k=sgp_k,
        features_model_name=features_model_name,
        temporal_shift_mode=temporal_shift_mode,
    )
    model.load_all(model_checkpoint_path)
    model.to(device)
    model.eval()

    scores = score_video(model, clips, loader, device=device)
    predictions = scores_to_predictions(
        scores,
        fps=video.metadata_fps,
        threshold=inference_threshold,
    )
    result = {"video_path": video.video_path, "predictions": predictions}
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def score_video(model, clips, loader, device: str):
    if not clips:
        raise ValueError("No clips generated for inference.")
    last_frame = max(frame.original_video_frame_nr for clip in clips for frame in clip.frames)
    scores = np.zeros((last_frame + 1, NUM_ACTION_CLASSES + 1), dtype=np.float32)
    counts = np.zeros((last_frame + 1, 1), dtype=np.float32)

    clip_offset = 0
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc="scoring"):
            clip_tensor = batch["clip_tensor"].to(device).float()
            with torch.amp.autocast(device_type=device, enabled=device == "cuda"):
                probs = torch.softmax(
                    model(clip_tensor, inference=True)["logits"], dim=-1
                ).detach().cpu().numpy()
            for batch_idx in range(probs.shape[0]):
                clip = clips[clip_offset + batch_idx]
                for frame_idx, frame in enumerate(clip.frames):
                    scores[frame.original_video_frame_nr] += probs[batch_idx, frame_idx]
                    counts[frame.original_video_frame_nr] += 1
            clip_offset += probs.shape[0]
    return scores / np.maximum(counts, 1.0)


def scores_to_predictions(scores, fps: float, threshold: float):
    predictions = []
    for class_index in range(1, NUM_ACTION_CLASSES + 1):
        action = index_to_label(class_index)
        if action is None:
            continue
        class_scores = scores[:, class_index]
        min_score = max(threshold, ACTION_CONFIGS[action].min_score)
        candidate_indices = np.where(class_scores >= min_score)[0]
        if candidate_indices.size == 0:
            continue
        kept = non_maximum_suppression(
            candidate_indices,
            class_scores,
            window_frames=int(ACTION_CONFIGS[action].tolerance_seconds * fps),
        )
        for frame_idx in kept:
            position = int(frame_idx / fps * 1000)
            predictions.append(
                {
                    "label": action.value,
                    "position": position,
                    "gameTime": format_game_time(position),
                    "confidence": float(class_scores[frame_idx]),
                }
            )
    predictions.sort(key=lambda item: item["position"])
    return predictions


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
