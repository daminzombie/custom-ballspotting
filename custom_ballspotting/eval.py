"""mAP evaluation for team ball action spotting.

Algorithm is ported directly from dudek ``ml/model/tdeed/eval/base.py``:
  - Per video: softmax scores matrix  (frames × 2N)  vs binary targets matrix (frames × 2N).
  - All-classes mAP@delta_frames: match predictions to ground-truth events within
    a symmetric frame-count tolerance, then compute precision/recall → AP per class → mean.

The public entry-point for training is :func:`val_map`.
"""

import dataclasses

import numpy as np
import torch
from torch.utils.data import DataLoader

from custom_ballspotting.actions import NUM_TEAM_ACTION_CLASSES, label_to_index
from custom_ballspotting.data import CustomTDeedDataset, VideoClip
from custom_ballspotting.inference import score_video


@dataclasses.dataclass
class VideoScoredData:
    """Per-video scores and targets needed by :func:`compute_map`."""

    video_id: str
    scores: np.ndarray  # (num_frames, 2*N)  foreground only, no background col
    targets: np.ndarray  # (num_frames, 2*N)  binary, 1 at ground-truth event frames


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
    """Compute mAP@delta_frames over all foreground classes.

    Parameters
    ----------
    video_data:
        One entry per validation video with pre-computed scores and targets.
    delta_frames:
        Symmetric frame-count tolerance for a prediction to count as a TP.
    num_classes:
        Number of foreground classes (``NUM_TEAM_ACTION_CLASSES`` = 2 * N).

    Returns
    -------
    float
        Mean Average Precision in [0, 1].
    """
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
) -> float:
    """Score all validation clips and compute mAP@delta_frames.

    Clips are grouped by source video.  For each video:

    * The model scores every val clip, producing a dense per-frame softmax
      matrix (same logic as :func:`~custom_ballspotting.inference.score_video`).
    * A binary targets matrix is built from the video's ground-truth annotations,
      mapped to frame indices via the video's metadata FPS.

    Parameters
    ----------
    model:
        ``CustomTDeedModule`` already on ``device``, in eval mode.
    val_clips:
        Validation clips produced by :func:`~custom_ballspotting.training.split_by_video`.
        Because ``split_by_video`` splits at the video level, every clip of a
        given video lands in the same split, so the targets matrix is complete.
    device:
        ``"cuda"`` or ``"cpu"``.
    val_batch_size:
        Clips per forward pass.
    delta_frames:
        Frame-count tolerance for TP matching.

    Returns
    -------
    float
        mAP@delta_frames in [0, 1].
    """
    # Group clips by source video
    by_video: dict[str, tuple] = {}
    for clip in val_clips:
        vid_id = clip.source_video.video_id or clip.source_video.video_path
        if vid_id not in by_video:
            by_video[vid_id] = (clip.source_video, [])
        by_video[vid_id][1].append(clip)

    video_data: list[VideoScoredData] = []
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
            # full_scores: (max_frame_nr + 1, 2*N+1) — includes background col 0
            full_scores = score_video(model, clips, loader, device=device)
            num_frames = full_scores.shape[0]

            # Drop background column; keep foreground cols 1..2N
            scores_fg = full_scores[:, 1:]  # (num_frames, 2*N)

            # Build binary targets from annotations
            fps = video_record.metadata_fps
            targets = np.zeros((num_frames, NUM_TEAM_ACTION_CLASSES), dtype=np.float32)
            for ann in video_record.annotations:
                frame = ann.frame_nr(fps)
                if frame < num_frames:
                    # label_to_index returns 1-based; subtract 1 for 0-based fg index
                    class_idx = label_to_index(ann.label, ann.team) - 1
                    targets[frame, class_idx] = 1.0

            video_data.append(
                VideoScoredData(video_id=vid_id, scores=scores_fg, targets=targets)
            )

    return compute_map(video_data, delta_frames, NUM_TEAM_ACTION_CLASSES)
