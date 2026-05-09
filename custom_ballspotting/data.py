import dataclasses
import json
import os
import random
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path
from typing import Iterable

import cv2
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms.v2.functional import hflip
from tqdm import tqdm

from custom_ballspotting.actions import (
    Action,
    TEAM_IGNORE_INDEX,
    Team,
    action_to_index,
    parse_team_string,
    team_to_index,
)
from custom_ballspotting.augmentations import (
    augment_with_camera_movement,
    crop_video,
    resize_frame,
)

GROUND_TRUTH_JSON = "ground_truth.json"
SOCCERNET_BALL_LABELS_JSON = "Labels-ball.json"


@dataclasses.dataclass(frozen=True)
class Annotation:
    label: Action
    position: int
    team: Team = Team.LEFT

    def frame_nr(self, fps: float) -> int:
        return int(self.position / 1000 * fps)


@dataclasses.dataclass(frozen=True)
class Frame:
    frame_path: str
    annotation: Annotation | None = None

    @property
    def original_video_frame_nr(self) -> int:
        return int(Path(self.frame_path).stem)


@dataclasses.dataclass
class VideoRecord:
    video_path: str
    annotations: list[Annotation]
    video_id: str | None = None
    #: Path fragment ``league/season/match`` for SoccerNet ``mAPevaluateTest`` (ground-truth ZIP).
    #: Set via ``ground_truth.json`` / ``Labels-ball.json`` keys ``soccernet_game_id`` or ``UrlLocal``,
    #: or inferred as the first three segments of ``video_id`` when applicable.
    soccernet_game_id: str | None = None

    @cached_property
    def metadata_fps(self) -> float:
        capture = cv2.VideoCapture(self.video_path)
        try:
            return float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()

    @cached_property
    def frames_path(self) -> str:
        base = os.path.basename(self.video_path)
        return os.path.join(os.path.dirname(self.video_path), f".frames_{base}")

    def play_video(self):
        capture = cv2.VideoCapture(self.video_path)
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield frame
        finally:
            capture.release()

    def extract_frames(
        self,
        stride: int = 2,
        target_width: int = 1280,
        target_height: int = 720,
        radius_seconds: int | None = None,
        save_all: bool = False,
        write_workers: int = 8,
    ):
        os.makedirs(self.frames_path, exist_ok=True)
        forced_frames = {ann.frame_nr(self.metadata_fps) for ann in self.annotations}
        if radius_seconds is not None:
            radius = int(radius_seconds * self.metadata_fps)
            expanded = set(forced_frames)
            for frame_nr in forced_frames:
                expanded |= set(range(frame_nr - radius, frame_nr + radius + 1, stride))
            forced_frames = expanded

        label = self.video_id or Path(self.video_path).stem
        frames_path = self.frames_path

        def _resize_and_write(frame_nr: int, frame) -> None:
            resized = resize_frame(frame, target_height=target_height, target_width=target_width)
            if not cv2.imwrite(os.path.join(frames_path, f"{frame_nr}.jpg"), resized):
                raise RuntimeError(f"Failed to save frame {frame_nr} for {self.video_path}")

        # Decode is sequential (H.264 requires it). cv2.resize and cv2.imwrite both
        # release the GIL, so write_workers threads give real CPU parallelism and
        # overlap with the next cap.read() call, making resize+write nearly free.
        futures = []
        with ThreadPoolExecutor(max_workers=write_workers) as pool:
            for frame_nr, frame in tqdm(
                enumerate(self.play_video()), desc=f"extracting {label}"
            ):
                if not save_all and frame_nr % stride != 0 and frame_nr not in forced_frames:
                    continue
                futures.append(pool.submit(_resize_and_write, frame_nr, frame))

            for fut in futures:
                fut.result()

    @property
    def frames(self) -> list[Frame]:
        if not os.path.exists(self.frames_path):
            raise FileNotFoundError(f"Frames missing at {self.frames_path}; extract first.")
        annotations_by_frame = {
            ann.frame_nr(self.metadata_fps): ann for ann in self.annotations
        }
        frame_files = sorted(
            os.listdir(self.frames_path), key=lambda name: int(Path(name).stem)
        )
        return [
            Frame(
                frame_path=os.path.join(self.frames_path, frame_file),
                annotation=annotations_by_frame.get(int(Path(frame_file).stem)),
            )
            for frame_file in frame_files
        ]

    def get_clips(self, accepted_gap: int = 2) -> list["VideoClip"]:
        clips: list[VideoClip] = []
        current: list[Frame] = []
        for frame in self.frames:
            if current and frame.original_video_frame_nr - current[-1].original_video_frame_nr > accepted_gap:
                clips.append(VideoClip(current, self))
                current = []
            current.append(frame)
        if current:
            clips.append(VideoClip(current, self))
        return clips


@dataclasses.dataclass(frozen=True)
class VideoClip:
    frames: list[Frame]
    source_video: VideoRecord
    #: If set (e.g. short video padded with repeated boundary frames up to temporal length),
    #: only logits at timesteps ``0 .. n-1`` are fused into ``score_video``. Padded repeats
    #: still feed the model but must not skew per-frame aggregates.
    logits_aggregate_timesteps: int | None = None

    @property
    def has_events(self) -> bool:
        return any(frame.annotation is not None for frame in self.frames)

    @property
    def unique_annotations(self) -> list[Annotation]:
        return [frame.annotation for frame in self.frames if frame.annotation is not None]

    def split(
        self, clip_frames_count: int, overlap: int, *, pad_if_shorter: bool = False
    ) -> list["VideoClip"]:
        step = clip_frames_count - overlap
        if step <= 0:
            raise ValueError("overlap must be smaller than clip_frames_count")
        clips: list[VideoClip] = []
        n = len(self.frames)
        if n == 0:
            return clips

        if n < clip_frames_count:
            if not pad_if_shorter:
                return clips

            # Inference-only: temporal model requires ``clip_frames_count`` tensors. Pad using
            # the final extracted frame; ``logits_aggregate_timesteps`` skips padded timesteps
            # during score fusion. Training callers keep ``pad_if_shorter=False`` (skip shorts).
            tail = list(self.frames) + [self.frames[-1]] * (clip_frames_count - n)
            return [
                VideoClip(
                    tail,
                    self.source_video,
                    logits_aggregate_timesteps=n,
                )
            ]

        for i in range(0, n, step):
            frames = self.frames[i : i + clip_frames_count]
            if len(frames) == clip_frames_count:
                clips.append(VideoClip(frames, self.source_video))

        # Always cover the tail. If the last regular clip does not reach the final
        # frame, anchor one more clip at the end of the sequence. score_video handles
        # overlapping clips via per-frame score averaging, so double-counting is fine.
        if n >= clip_frames_count:
            if not clips or clips[-1].frames[-1] != self.frames[-1]:
                clips.append(VideoClip(self.frames[-clip_frames_count:], self.source_video))

        return clips


@dataclasses.dataclass
class TDeedClip:
    origin: VideoClip
    clip_tensor: torch.Tensor
    action_label_ids: torch.Tensor
    team_label_ids: torch.Tensor
    displacement: torch.Tensor
    profile: dict[str, float] | None = None

    @classmethod
    def from_clip(
        cls,
        clip: VideoClip,
        displacement_radius: int = 4,
        flip_proba: float = 0.0,
        camera_move_proba: float = 0.0,
        crop_proba: float = 0.0,
        crop_size: float = 0.9,
        device: str | None = None,
        image_executor: ThreadPoolExecutor | None = None,
        profile: bool = False,
        profile_label: str = "",
    ):
        item_start_t = time.perf_counter()
        num_frames = len(clip.frames)
        action_label_ids = torch.zeros(num_frames, dtype=torch.long)
        team_label_ids = torch.full((num_frames,), TEAM_IGNORE_INDEX, dtype=torch.long)
        displacement = torch.zeros(num_frames, dtype=torch.float32)

        flip = random.random() < flip_proba

        for idx, frame in enumerate(clip.frames):
            if frame.annotation is None:
                continue
            team = frame.annotation.team.flip() if flip else frame.annotation.team
            action_idx = action_to_index(frame.annotation.label)
            team_idx = team_to_index(team)
            valid_offsets = range(
                max(-displacement_radius, -idx),
                min(displacement_radius, num_frames - idx - 1) + 1,
            )
            for offset in valid_offsets:
                target_idx = idx + offset
                action_label_ids[target_idx] = action_idx
                team_label_ids[target_idx] = team_idx
                displacement[idx + offset] = float(offset)
        label_done_t = time.perf_counter()

        def load_image(path: str):
            img = torchvision.io.read_image(path)
            return hflip(img) if flip else img

        frame_paths = [frame.frame_path for frame in clip.frames]
        if image_executor is None:
            with ThreadPoolExecutor() as executor:
                imgs = list(executor.map(load_image, frame_paths))
        else:
            imgs = list(image_executor.map(load_image, frame_paths))
        read_done_t = time.perf_counter()
        clip_tensor = torch.stack(imgs, dim=0)
        stack_done_t = time.perf_counter()
        if device is not None:
            clip_tensor = clip_tensor.to(device)
            if profile and torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
        move_done_t = time.perf_counter()
        if random.random() < camera_move_proba:
            clip_tensor = augment_with_camera_movement(clip_tensor)
        if random.random() < crop_proba:
            clip_tensor = crop_video(
                clip_tensor,
                crop_size_h=int(clip_tensor.shape[2] * crop_size),
                crop_size_w=int(clip_tensor.shape[3] * crop_size),
            )
        if profile and device is not None and torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        aug_done_t = time.perf_counter()
        profile_metrics = {
            "label_s": label_done_t - item_start_t,
            "read_s": read_done_t - label_done_t,
            "stack_s": stack_done_t - read_done_t,
            "move_s": move_done_t - stack_done_t,
            "augment_s": aug_done_t - move_done_t,
            "total_s": aug_done_t - item_start_t,
        }
        if profile:
            print(
                "data profile "
                f"{profile_label} "
                f"frames={num_frames} "
                f"flip={int(flip)} "
                f"device={device or 'cpu'} "
                f"label_s={profile_metrics['label_s']:.4f} "
                f"read_s={profile_metrics['read_s']:.4f} "
                f"stack_s={profile_metrics['stack_s']:.4f} "
                f"move_s={profile_metrics['move_s']:.4f} "
                f"augment_s={profile_metrics['augment_s']:.4f} "
                f"total_s={profile_metrics['total_s']:.4f}",
                flush=True,
            )
        return cls(
            origin=clip,
            # Match dudek's training path: when training on CUDA, each clip is moved
            # sample-by-sample before DataLoader collation, avoiding a huge CPU batch copy.
            clip_tensor=clip_tensor.float() if device is not None else clip_tensor,
            action_label_ids=action_label_ids.to(device) if device is not None else action_label_ids,
            team_label_ids=team_label_ids.to(device) if device is not None else team_label_ids,
            displacement=displacement.to(device) if device is not None else displacement,
            profile=profile_metrics if profile else None,
        )


class CustomTDeedDataset(Dataset):
    def __init__(
        self,
        clips: list[VideoClip],
        displacement_radius: int = 4,
        flip_proba: float = 0.0,
        camera_move_proba: float = 0.0,
        crop_proba: float = 0.0,
        even_choice_proba: float = 0.0,
        enforced_epoch_size: int | None = None,
        device: str | None = None,
        profile_items: int = 0,
    ):
        self.clips = clips
        self.displacement_radius = displacement_radius
        self.flip_proba = flip_proba
        self.camera_move_proba = camera_move_proba
        self.crop_proba = crop_proba
        self.even_choice_proba = even_choice_proba
        self.enforced_epoch_size = enforced_epoch_size
        self.device = device
        self._image_executor = ThreadPoolExecutor()
        self.profile_items = profile_items
        self._profile_seen = 0
        self.clip_ids_by_label: dict[Action, list[int]] = {action: [] for action in Action}
        for idx, clip in enumerate(self.clips):
            for annotation in clip.unique_annotations:
                self.clip_ids_by_label[annotation.label].append(idx)

    def __del__(self):
        executor = getattr(self, "_image_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def __getstate__(self):
        # ThreadPoolExecutor is not pickleable / pickling queues breaks DataLoader workers
        # (spawn on Windows picks the whole dataset).
        state = self.__dict__.copy()
        state.pop("_image_executor", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._image_executor = ThreadPoolExecutor()

    def __len__(self):
        return self.enforced_epoch_size or len(self.clips)

    def __getitem__(self, idx):
        original_idx = idx
        if self.enforced_epoch_size is not None:
            idx = random.randrange(len(self.clips))
        if self.even_choice_proba and random.random() < self.even_choice_proba:
            populated = [ids for ids in self.clip_ids_by_label.values() if ids]
            if populated:
                idx = random.choice(random.choice(populated))
        profile = self._profile_seen < self.profile_items
        if profile:
            self._profile_seen += 1
        item = TDeedClip.from_clip(
            self.clips[idx],
            displacement_radius=self.displacement_radius,
            flip_proba=self.flip_proba,
            camera_move_proba=self.camera_move_proba,
            crop_proba=self.crop_proba,
            device=self.device,
            image_executor=self._image_executor,
            profile=profile,
            profile_label=f"item={self._profile_seen}/{self.profile_items} idx={idx} requested={original_idx}",
        )
        out = {
            "clip_tensor": item.clip_tensor,
            "action_label_ids": item.action_label_ids,
            "team_label_ids": item.team_label_ids,
            "displacement": item.displacement,
        }
        if self.profile_items:
            metrics = item.profile or {}
            out["profile_label_s"] = torch.tensor(metrics.get("label_s", 0.0), dtype=torch.float64)
            out["profile_read_s"] = torch.tensor(metrics.get("read_s", 0.0), dtype=torch.float64)
            out["profile_stack_s"] = torch.tensor(metrics.get("stack_s", 0.0), dtype=torch.float64)
            out["profile_move_s"] = torch.tensor(metrics.get("move_s", 0.0), dtype=torch.float64)
            out["profile_augment_s"] = torch.tensor(metrics.get("augment_s", 0.0), dtype=torch.float64)
            out["profile_total_s"] = torch.tensor(metrics.get("total_s", 0.0), dtype=torch.float64)
            out["profile_count"] = torch.tensor(1.0 if item.profile else 0.0, dtype=torch.float64)
        return out


def find_first_mp4(directory: str | Path) -> str | None:
    """Return absolute path to the first ``*.mp4`` regular file (lexicographic by name)."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return None
    mp4s = sorted(p for p in dir_path.glob("*.mp4") if p.is_file())
    return str(mp4s[0].resolve()) if mp4s else None


def infer_soccernet_game_id_from_video_id(video_id: str | None) -> str | None:
    if not video_id:
        return None
    parts = [p for p in str(video_id).replace("\\", "/").split("/") if p]
    if len(parts) >= 3:
        return "/".join(parts[:3])
    return None


def annotations_from_ground_truth_payload(
    raw: dict,
    *,
    skip_unknown_labels: bool = True,
    unknown_labels_acc: set[str] | None = None,
    random_team_when_na: bool = True,
) -> list[Annotation]:
    """Parse SoccerNet-style `ground_truth.json` annotations.

    Each annotation may optionally carry a ``"team"`` field (``"left"``,
    ``"right"``, or ``"not applicable"``, plus common aliases such as
    ``"not_applicable"`` or ``"n/a"``).  When absent the annotation defaults
    to ``Team.LEFT``.  Unrecognised team strings also default to ``Team.LEFT``.
    When ``random_team_when_na`` is True, ``"not applicable"`` (and recognised
    NA aliases) is resolved to left or right at random (same idea as dudek
    ``random_team_when_no_team``).
    """
    out: list[Annotation] = []
    for item in raw.get("annotations", []):
        label_raw = item["label"]
        try:
            action = Action(label_raw)
        except ValueError:
            if skip_unknown_labels:
                if unknown_labels_acc is not None:
                    unknown_labels_acc.add(str(label_raw))
                continue
            raise
        pos = int(item["position"])
        team = parse_team_string(item.get("team"))
        if random_team_when_na and team == Team.NOT_APPLICABLE:
            team = Team.RIGHT if random.random() > 0.5 else Team.LEFT
        out.append(Annotation(label=action, position=pos, team=team))
    return out


def video_record_from_clip_dir(
    clip_dir: Path,
    dataset_root: Path,
    *,
    unknown_labels_acc: set[str] | None = None,
    random_team_when_na: bool = True,
) -> VideoRecord | None:
    """One clip directory: first `*.mp4` + labels JSON."""
    mp4 = find_first_mp4(clip_dir)
    if mp4 is None:
        return None
    labels_path = clip_dir / GROUND_TRUTH_JSON
    if not labels_path.is_file():
        labels_path = clip_dir / SOCCERNET_BALL_LABELS_JSON
    if not labels_path.is_file():
        return None
    with open(labels_path, "r") as f:
        raw = json.load(f)
    annotations = annotations_from_ground_truth_payload(
        raw,
        unknown_labels_acc=unknown_labels_acc,
        random_team_when_na=random_team_when_na,
    )
    try:
        rel = clip_dir.relative_to(dataset_root)
    except ValueError:
        rel = clip_dir.resolve()
    video_id = str(rel).replace(os.sep, "/")
    game = raw.get("soccernet_game_id") or raw.get("UrlLocal")
    if not game:
        game = infer_soccernet_game_id_from_video_id(video_id)
    return VideoRecord(
        video_path=mp4,
        annotations=annotations,
        video_id=video_id,
        soccernet_game_id=game,
    )


def load_dataset_records(
    dataset_root: str,
    *,
    random_team_when_na: bool = True,
) -> list[VideoRecord]:
    """
    Load clips under dataset_root (recursive): each folder that contains
    ``ground_truth.json`` or dudek/SoccerNet ``Labels-ball.json`` uses the
    lexicographically first ``*.mp4`` regular file in that folder.
    """
    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset_root is not a directory: {root}")
    clip_dirs = sorted(
        {p.parent for p in root.rglob(GROUND_TRUTH_JSON)}
        | {p.parent for p in root.rglob(SOCCERNET_BALL_LABELS_JSON)}
    )
    unknown_labels: set[str] = set()
    records: list[VideoRecord] = []
    for clip_dir in clip_dirs:
        rec = video_record_from_clip_dir(
            clip_dir,
            root,
            unknown_labels_acc=unknown_labels,
            random_team_when_na=random_team_when_na,
        )
        if rec is not None:
            records.append(rec)
    if unknown_labels:
        sample = ", ".join(sorted(unknown_labels)[:12])
        more = f" (+{len(unknown_labels) - 12} types)" if len(unknown_labels) > 12 else ""
        warnings.warn(
            f"Skipped annotations whose labels are not in Action enum (types: {sample}{more}). "
            "Extend Action in actions.py or map labels upstream.",
            stacklevel=2,
        )
    return records


def build_clips(
    records: Iterable[VideoRecord],
    clip_frames_count: int,
    overlap: int,
    accepted_gap: int = 2,
) -> list[VideoClip]:
    clips: list[VideoClip] = []
    for record in records:
        for continuous_clip in record.get_clips(accepted_gap=accepted_gap):
            clips.extend(continuous_clip.split(clip_frames_count, overlap))
    return clips
