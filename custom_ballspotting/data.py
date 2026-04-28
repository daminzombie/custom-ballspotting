import dataclasses
import json
import os
import random
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

    @classmethod
    def from_json(cls, raw: dict, manifest_dir: str):
        video_path = raw["video_path"]
        if not os.path.isabs(video_path):
            video_path = os.path.normpath(os.path.join(manifest_dir, video_path))
        annotations = [
            Annotation(label=Action(item["label"]), position=int(item["position"]))
            for item in raw.get("annotations", [])
        ]
        return cls(
            video_path=video_path,
            annotations=annotations,
            video_id=raw.get("video_id") or Path(video_path).stem,
        )

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
    ):
        os.makedirs(self.frames_path, exist_ok=True)
        forced_frames = {ann.frame_nr(self.metadata_fps) for ann in self.annotations}
        if radius_seconds is not None:
            radius = int(radius_seconds * self.metadata_fps)
            expanded = set(forced_frames)
            for frame_nr in forced_frames:
                expanded |= set(range(frame_nr - radius, frame_nr + radius + 1, stride))
            forced_frames = expanded

        for frame_nr, frame in tqdm(
            enumerate(self.play_video()), desc=f"extracting {self.video_id}"
        ):
            if not save_all and frame_nr % stride != 0 and frame_nr not in forced_frames:
                continue
            frame = resize_frame(frame, target_height=target_height, target_width=target_width)
            if not cv2.imwrite(os.path.join(self.frames_path, f"{frame_nr}.jpg"), frame):
                raise RuntimeError(f"Failed to save frame {frame_nr} for {self.video_path}")

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
            clip_tensor=clip_tensor.float(),
            label_ids=label_ids,
            displacement=displacement,
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
    ):
        self.clips = clips
        self.displacement_radius = displacement_radius
        self.flip_proba = flip_proba
        self.camera_move_proba = camera_move_proba
        self.crop_proba = crop_proba
        self.even_choice_proba = even_choice_proba
        self.enforced_epoch_size = enforced_epoch_size
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
        )
        return {
            "clip_tensor": item.clip_tensor,
            "label_ids": item.label_ids,
            "displacement": item.displacement,
        }


def load_manifest(manifest_path: str) -> list[VideoRecord]:
    with open(manifest_path, "r") as f:
        raw = json.load(f)
    videos_raw = raw["videos"] if isinstance(raw, dict) and "videos" in raw else raw
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    return [VideoRecord.from_json(item, manifest_dir=manifest_dir) for item in videos_raw]


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
