"""mAP evaluation for ball action spotting (no team dimension).

The ranking / AP integration in :func:`compute_map` matches dudek
``TDeedMAPEvaluator.compute_map`` (``map_mine``).

:class:`ValMapMetrics` can also include SoccerNet ``mAPevaluateTest`` / ``average_mAP``
(``challenge_mAP``) when ``soccernet_path`` and per-video ``soccernet_game_id`` are set.
"""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from custom_ballspotting.actions import (
    NUM_ACTION_CLASSES,
    Action,
    foreground_column_for_action,
)
from custom_ballspotting.data import CustomTDeedDataset, VideoClip
from custom_ballspotting.inference import score_video
from custom_ballspotting.map_scoring import (
    dudek_style_scores_matrix,
    soft_non_maximum_suppression,
)


@dataclasses.dataclass
class ValMapMetrics:
    """Validation metrics; ``challenge_mAP`` is SoccerNet test-style mAP when available."""

    map_mine: float
    challenge_mAP: float | None = None


@dataclasses.dataclass
class VideoScoredData:
    """Per-video scores and targets needed by :func:`compute_map`."""

    video_id: str
    scores: np.ndarray  # (num_frames, N)  foreground only, no background col
    targets: np.ndarray  # (num_frames, N)  binary, 1 at ground-truth event frames


def scores_fg_to_challenge_results_json(
    scores_fg: np.ndarray,
    fps: float,
    game_path: str,
) -> dict:
    """Dense per-frame predictions JSON for SoccerNet challenge mAP evaluation."""
    predictions: list[dict] = []
    for i in range(scores_fg.shape[0]):
        x = scores_fg[i]
        confidence = float(np.max(x))
        label_col = int(np.argmax(x))
        action = list(Action)[label_col]
        position = int(i / fps * 1000)
        total_seconds = position // 1000
        half = 1 if total_seconds < 45 * 60 else 2
        seconds_in_half = total_seconds if half == 1 else total_seconds - 45 * 60
        game_time = f"{half} - {seconds_in_half // 60:02d}:{seconds_in_half % 60:02d}"
        predictions.append(
            {
                "gameTime": game_time,
                "label": action.value,
                "position": position,
                "confidence": confidence,
                "half": half,
            }
        )
    return {"UrlLocal": game_path, "predictions": predictions}


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Area under the precision-recall curve (11-point interpolation envelope)."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_map(
    video_data: list[VideoScoredData],
    delta_frames: int,
    num_classes: int,
) -> float:
    """Compute mAP@delta_frames over all foreground classes."""
    APs: list[float] = []

    for class_idx in range(num_classes):
        all_predictions: list[dict] = []
        all_ground_truths: dict[str, dict] = {}

        for vd in video_data:
            vid_id = vd.video_id
            class_preds = vd.scores[:, class_idx]
            class_targets = vd.targets[:, class_idx]

            pred_indices = np.where(class_preds > 0)[0]
            for fi, score in zip(pred_indices, class_preds[pred_indices]):
                all_predictions.append(
                    {"video_id": vid_id, "frame_idx": int(fi), "score": float(score)}
                )

            gt_indices = np.where(class_targets == 1)[0].tolist()
            if vid_id not in all_ground_truths:
                all_ground_truths[vid_id] = {
                    "gt_indices": gt_indices,
                    "matches": np.zeros(len(gt_indices), dtype=bool),
                }
            else:
                all_ground_truths[vid_id]["gt_indices"].extend(gt_indices)
                all_ground_truths[vid_id]["matches"] = np.concatenate(
                    [
                        all_ground_truths[vid_id]["matches"],
                        np.zeros(len(gt_indices), dtype=bool),
                    ]
                )

        total_gt = sum(len(v["gt_indices"]) for v in all_ground_truths.values())
        if total_gt == 0:
            APs.append(0.0)
            continue

        all_predictions.sort(key=lambda x: x["score"], reverse=True)
        TP = np.zeros(len(all_predictions))
        FP = np.zeros(len(all_predictions))

        for pred_idx, pred in enumerate(all_predictions):
            vid_id = pred["video_id"]
            frame_idx = pred["frame_idx"]
            gt_info = all_ground_truths.get(vid_id, {"gt_indices": [], "matches": np.zeros(0, dtype=bool)})
            gt_indices = gt_info["gt_indices"]
            matches = gt_info["matches"]

            min_delta = float("inf")
            matched_gt_idx = -1
            for gt_i, gt_frame in enumerate(gt_indices):
                if not matches[gt_i]:
                    delta = abs(frame_idx - gt_frame)
                    if delta <= delta_frames and delta < min_delta:
                        min_delta = delta
                        matched_gt_idx = gt_i

            if matched_gt_idx >= 0:
                TP[pred_idx] = 1
                matches[matched_gt_idx] = True
                all_ground_truths[vid_id]["matches"] = matches
            else:
                FP[pred_idx] = 1

        cum_TP = np.cumsum(TP)
        cum_FP = np.cumsum(FP)
        precisions = cum_TP / (cum_TP + cum_FP + 1e-8)
        recalls = cum_TP / (total_gt + 1e-8)
        APs.append(compute_ap(recalls, precisions))

    return float(np.mean(APs)) if APs else 0.0


def val_map(
    model,
    val_clips: list[VideoClip],
    device: str,
    val_batch_size: int = 1,
    delta_frames: int = 5,
    *,
    dudek_style_scoring: bool = True,
    use_snms: bool = True,
    snms_class_window: int | list[int] = 12,
    snms_threshold: float = 0.01,
    soccernet_path: str | None = None,
    run_soccernet_challenge_map: bool = False,
    soccernet_challenge_metric: str = "at1",
) -> ValMapMetrics:
    """Score all validation clips and compute ``map_mine``; optionally SoccerNet challenge mAP.

    When ``run_soccernet_challenge_map`` is True and ``soccernet_path`` points at the
    SoccerNet dataset (or labels zip), each :class:`~custom_ballspotting.data.VideoRecord`
    must have ``soccernet_game_id`` set (JSON ``soccernet_game_id`` / ``UrlLocal``, or
    inferred from the first three ``video_id`` path segments).
    """
    by_video: dict[str, tuple] = {}
    for clip in val_clips:
        vid_id = clip.source_video.video_id or clip.source_video.video_path
        if vid_id not in by_video:
            by_video[vid_id] = (clip.source_video, [])
        by_video[vid_id][1].append(clip)

    video_data: list[VideoScoredData] = []
    challenge_merged: dict[str, dict] = {}
    challenge_missing: list[str] = []
    model.eval()
    with torch.no_grad():
        for vid_id, (video_record, clips) in by_video.items():
            dataset = CustomTDeedDataset(clips, displacement_radius=0)
            loader = DataLoader(
                dataset,
                batch_size=val_batch_size,
                shuffle=False,
                pin_memory=device == "cuda",
            )
            if dudek_style_scoring:
                full_scores = dudek_style_scores_matrix(
                    model,
                    clips,
                    loader,
                    device,
                    num_classes_with_background=NUM_ACTION_CLASSES + 1,
                )
            else:
                full_scores = score_video(model, clips, loader, device=device)
            num_frames = full_scores.shape[0]

            scores_fg = full_scores[:, 1:]
            if dudek_style_scoring and use_snms:
                scores_fg = soft_non_maximum_suppression(
                    scores_fg,
                    class_window=snms_class_window,
                    threshold=snms_threshold,
                )

            fps = float(video_record.metadata_fps)
            targets = np.zeros((num_frames, NUM_ACTION_CLASSES), dtype=np.float32)
            for ann in video_record.annotations:
                frame = ann.frame_nr(fps)
                if frame < num_frames:
                    class_idx = foreground_column_for_action(ann.label)
                    targets[frame, class_idx] = 1.0

            video_data.append(
                VideoScoredData(video_id=vid_id, scores=scores_fg, targets=targets)
            )

            if run_soccernet_challenge_map and soccernet_path:
                gid = video_record.soccernet_game_id
                if not gid:
                    challenge_missing.append(vid_id)
                    continue
                payload = scores_fg_to_challenge_results_json(scores_fg, fps, gid)
                if gid not in challenge_merged:
                    challenge_merged[gid] = payload
                else:
                    challenge_merged[gid]["predictions"].extend(payload["predictions"])

    map_mine = compute_map(video_data, delta_frames, NUM_ACTION_CLASSES)

    challenge_mAP: float | None = None
    if run_soccernet_challenge_map:
        if not soccernet_path:
            warnings.warn(
                "run_soccernet_challenge_map is True but soccernet_path is empty; "
                "skipping SoccerNet challenge mAP.",
                stacklevel=2,
            )
        elif challenge_missing:
            warnings.warn(
                "Skipping SoccerNet challenge mAP: missing soccernet_game_id "
                f"for video_ids: {challenge_missing[:8]}"
                f"{'...' if len(challenge_missing) > 8 else ''}",
                stacklevel=2,
            )
        elif not challenge_merged:
            warnings.warn(
                "Skipping SoccerNet challenge mAP: no prediction payloads were built.",
                stacklevel=2,
            )
        else:
            try:
                from custom_ballspotting.soccernet_challenge_eval import (
                    run_mapevaluate_test_with_zip,
                )

                games = sorted(challenge_merged.keys())
                results = run_mapevaluate_test_with_zip(
                    games=games,
                    soccernet_path=soccernet_path,
                    game_results_json=dict(challenge_merged),
                    metric=soccernet_challenge_metric,
                )
                challenge_mAP = float(results["mAP"])
            except ImportError as e:
                warnings.warn(f"SoccerNet challenge mAP unavailable: {e}", stacklevel=2)
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"SoccerNet challenge mAP failed: {e}", stacklevel=2)

    return ValMapMetrics(map_mine=map_mine, challenge_mAP=challenge_mAP)