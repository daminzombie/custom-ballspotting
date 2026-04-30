import dataclasses
import json
import os
import random
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

from custom_ballspotting.actions import Action, label_to_index
from custom_ballspotting.augmentations import (
    augment_with_camera_movement,
    crop_video,
    resize_frame,
)

GROUND_TRUTH_JSON = "ground_truth.json"


@dataclasses.dataclass(frozen=True)
class Annotation:
    label: Action
    position: int

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

    @property
    def has_events(self) -> bool:
        return any(frame.annotation is not None for frame in self.frames)

    @property
    def unique_annotations(self) -> list[Annotation]:
        return [frame.annotation for frame in self.frames if frame.annotation is not None]

    def split(self, clip_frames_count: int, overlap: int) -> list["VideoClip"]:
        step = clip_frames_count - overlap
        if step <= 0:
            raise ValueError("overlap must be smaller than clip_frames_count")
        clips = []
        for i in range(0, len(self.frames), step):
            frames = self.frames[i : i + clip_frames_count]
            if len(frames) == clip_frames_count:
                clips.append(VideoClip(frames, self.source_video))

        # Always cover the tail. If the last regular clip does not reach the final
        # frame, anchor one more clip at the end of the sequence. score_video handles
        # overlapping clips via per-frame score averaging, so double-counting is fine.
        if len(self.frames) >= clip_frames_count:
            if not clips or clips[-1].frames[-1] != self.frames[-1]:
                clips.append(VideoClip(self.frames[-clip_frames_count:], self.source_video))

        return clips


@dataclasses.dataclass
class TDeedClip:
    origin: VideoClip
    clip_tensor: torch.Tensor
    label_ids: torch.Tensor
    displacement: torch.Tensor

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
    ):
        num_frames = len(clip.frames)
        label_ids = torch.zeros(num_frames, dtype=torch.long)
        displacement = torch.zeros(num_frames, dtype=torch.float32)
        for idx, frame in enumerate(clip.frames):
            if frame.annotation is None:
                continue
            label_idx = label_to_index(frame.annotation.label)
            valid_offsets = range(
                max(-displacement_radius, -idx),
                min(displacement_radius, num_frames - idx - 1) + 1,
            )
            for offset in valid_offsets:
                label_ids[idx + offset] = label_idx
                displacement[idx + offset] = float(offset)

        flip = random.random() < flip_proba

        def load_image(path: str):
            img = torchvision.io.read_image(path)
            return hflip(img) if flip else img

        with ThreadPoolExecutor() as executor:
            imgs = list(executor.map(load_image, [frame.frame_path for frame in clip.frames]))
        clip_tensor = torch.stack(imgs, dim=0)
        if device is not None:
            clip_tensor = clip_tensor.to(device)
        if random.random() < camera_move_proba:
            clip_tensor = augment_with_camera_movement(clip_tensor)
        if random.random() < crop_proba:
            clip_tensor = crop_video(
                clip_tensor,
                crop_size_h=int(clip_tensor.shape[2] * crop_size),
                crop_size_w=int(clip_tensor.shape[3] * crop_size),
            )
        return cls(
            origin=clip,
            # Match dudek's training path: when training on CUDA, each clip is moved
            # sample-by-sample before DataLoader collation, avoiding a huge CPU batch copy.
            clip_tensor=clip_tensor.float() if device is not None else clip_tensor,
            label_ids=label_ids.to(device) if device is not None else label_ids,
            displacement=displacement.to(device) if device is not None else displacement,
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
    ):
        self.clips = clips
        self.displacement_radius = displacement_radius
        self.flip_proba = flip_proba
        self.camera_move_proba = camera_move_proba
        self.crop_proba = crop_proba
        self.even_choice_proba = even_choice_proba
        self.enforced_epoch_size = enforced_epoch_size
        self.device = device
        self.clip_ids_by_label: dict[Action, list[int]] = {action: [] for action in Action}
        for idx, clip in enumerate(self.clips):
            for annotation in clip.unique_annotations:
                self.clip_ids_by_label[annotation.label].append(idx)

    def __len__(self):
        return self.enforced_epoch_size or len(self.clips)

    def __getitem__(self, idx):
        if self.enforced_epoch_size is not None:
            idx = random.randrange(len(self.clips))
        if self.even_choice_proba and random.random() < self.even_choice_proba:
            populated = [ids for ids in self.clip_ids_by_label.values() if ids]
            if populated:
                idx = random.choice(random.choice(populated))
        item = TDeedClip.from_clip(
            self.clips[idx],
            displacement_radius=self.displacement_radius,
            flip_proba=self.flip_proba,
            camera_move_proba=self.camera_move_proba,
            crop_proba=self.crop_proba,
            device=self.device,
        )
        return {
            "clip_tensor": item.clip_tensor,
            "label_ids": item.label_ids,
            "displacement": item.displacement,
        }


def find_first_mp4(directory: str | Path) -> str | None:
    """Return absolute path to the first ``*.mp4`` regular file (lexicographic by name)."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return None
    mp4s = sorted(p for p in dir_path.glob("*.mp4") if p.is_file())
    return str(mp4s[0].resolve()) if mp4s else None


def annotations_from_ground_truth_payload(
    raw: dict,
    *,
    skip_unknown_labels: bool = True,
    unknown_labels_acc: set[str] | None = None,
) -> list[Annotation]:
    """Parse SoccerNet-style `ground_truth.json` annotations."""
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
        out.append(Annotation(label=action, position=pos))
    return out


def video_record_from_clip_dir(
    clip_dir: Path,
    dataset_root: Path,
    *,
    unknown_labels_acc: set[str] | None = None,
) -> VideoRecord | None:
    """One clip directory: first `*.mp4` + `ground_truth.json`."""
    mp4 = find_first_mp4(clip_dir)
    if mp4 is None:
        return None
    gt_path = clip_dir / GROUND_TRUTH_JSON
    if not gt_path.is_file():
        return None
    with open(gt_path, "r") as f:
        raw = json.load(f)
    annotations = annotations_from_ground_truth_payload(
        raw, unknown_labels_acc=unknown_labels_acc
    )
    try:
        rel = clip_dir.relative_to(dataset_root)
    except ValueError:
        rel = clip_dir.resolve()
    video_id = str(rel).replace(os.sep, "/")
    return VideoRecord(video_path=mp4, annotations=annotations, video_id=video_id)


def load_dataset_records(dataset_root: str) -> list[VideoRecord]:
    """
    Load clips under dataset_root (recursive): each folder that contains
    ``ground_truth.json`` uses the lexicographically first ``*.mp4`` regular file in that folder.
    """
    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset_root is not a directory: {root}")
    clip_dirs = sorted({p.parent for p in root.rglob(GROUND_TRUTH_JSON)})
    unknown_labels: set[str] = set()
    records: list[VideoRecord] = []
    for clip_dir in clip_dirs:
        rec = video_record_from_clip_dir(clip_dir, root, unknown_labels_acc=unknown_labels)
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
