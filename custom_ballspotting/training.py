import json
import os
import random
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from custom_ballspotting.actions import ACTION_CONFIGS, Action, NUM_ACTION_CLASSES, NUM_TEAM_ACTION_CLASSES
from custom_ballspotting.checkpoints import render_checkpoint_path, write_checkpoint_metadata
from custom_ballspotting.data import (
    CustomTDeedDataset,
    VideoClip,
    build_clips,
    load_dataset_records,
)
from custom_ballspotting.model.tdeed import CustomTDeedModule


@dataclass
class TrainConfig:
    clip_frames_count: int = 100
    overlap: int = 88
    displacement_radius: int = 4
    features_model_name: str = "regnety_008"
    temporal_shift_mode: str = "gsf"
    n_layers: int = 2
    sgp_ks: int = 9
    sgp_k: int = 4
    gaussian_blur_kernel_size: int = 5
    nr_epochs: int = 25
    warm_up_epochs: int = 1
    learning_rate: float = 0.0003
    train_batch_size: int = 1
    val_batch_size: int = 1
    acc_grad_iter: int = 8
    flip_proba: float = 0.1
    camera_move_proba: float = 0.1
    crop_proba: float = 0.1
    even_choice_proba: float = 0.0
    train_split: float = 0.8  # used only when run_validation is true
    run_validation: bool = False  # if false: all clips train, checkpoint on lowest train loss
    enforce_train_epoch_size: int | None = None
    enforce_val_epoch_size: int | None = None
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_model(
    clips: list[VideoClip],
    save_as: str,
    pretrained_checkpoint_path: str | None = None,
    experiment_name: str = "custom_tdeed",
    config: TrainConfig | None = None,
) -> CustomTDeedModule:
    config = config or TrainConfig()
    save_as = render_checkpoint_path(save_as, experiment_name=experiment_name)
    if config.run_validation:
        train_clips, val_clips = split_by_video(clips, config.train_split, config.random_seed)
    else:
        train_clips = clips
        val_clips = []
    train_dataset = CustomTDeedDataset(
        train_clips,
        displacement_radius=config.displacement_radius,
        flip_proba=config.flip_proba,
        camera_move_proba=config.camera_move_proba,
        crop_proba=config.crop_proba,
        even_choice_proba=config.even_choice_proba,
        enforced_epoch_size=config.enforce_train_epoch_size,
        device=config.device if config.device == "cuda" else None,
    )
    val_dataset = (
        CustomTDeedDataset(
            val_clips,
            displacement_radius=config.displacement_radius,
            enforced_epoch_size=config.enforce_val_epoch_size,
            device=config.device if config.device == "cuda" else None,
        )
        if config.run_validation and val_clips
        else None
    )

    model = CustomTDeedModule(
        clip_len=config.clip_frames_count,
        num_actions=NUM_ACTION_CLASSES,
        n_layers=config.n_layers,
        sgp_ks=config.sgp_ks,
        sgp_k=config.sgp_k,
        features_model_name=config.features_model_name,
        temporal_shift_mode=config.temporal_shift_mode,
        gaussian_blur_ks=config.gaussian_blur_kernel_size,
    )
    if pretrained_checkpoint_path:
        model.load_backbone(pretrained_checkpoint_path)
    model.to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda") if config.device == "cuda" else None
    use_cuda = config.device == "cuda"
    # When using CUDA, datasets already return CUDA tensors to match dudek's
    # train-challenge input path. Pinned memory only applies to CPU tensors.
    pin_memory = use_cuda and train_dataset.device is None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            pin_memory=pin_memory,
        )
        if val_dataset is not None
        else None
    )
    optimizer_steps_per_epoch = max(1, len(train_loader) // config.acc_grad_iter)
    warmup_steps = optimizer_steps_per_epoch * config.warm_up_epochs
    total_steps = max(1, (config.nr_epochs - config.warm_up_epochs) * optimizer_steps_per_epoch)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, total_iters=max(1, warmup_steps)),
            CosineAnnealingLR(optimizer, T_max=total_steps),
        ],
        milestones=[max(1, warmup_steps)],
    )
    writer = SummaryWriter(log_dir=f"runs/{experiment_name}_{time.time()}")
    writer.add_text("train/config", json.dumps(config.__dict__, indent=2, default=str), 0)

    # Weight vector has 2*N+1 entries: background=1.0, then LEFT-team weights,
    # then RIGHT-team weights (same per-action weight for both teams).
    per_action_weights = [ACTION_CONFIGS[action].weight for action in Action]
    class_weights = torch.tensor(
        [1.0] + per_action_weights + per_action_weights,
        dtype=torch.float32,
        device=config.device,
    )
    best_metric = float("inf")
    best_metric_name = "val_loss" if config.run_validation and val_loader is not None else "train_loss"
    for epoch in range(config.nr_epochs):
        print(f"Epoch {epoch + 1}/{config.nr_epochs}", flush=True)
        train_loss = run_epoch(
            model,
            train_loader,
            config.device,
            class_weights,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            acc_grad_iter=config.acc_grad_iter,
            epoch_index=epoch,
            nr_epochs=config.nr_epochs,
            phase="train",
        )
        if val_loader is not None:
            val_loss = run_epoch(
                model,
                val_loader,
                config.device,
                class_weights,
                epoch_index=epoch,
                nr_epochs=config.nr_epochs,
                phase="val",
            )
        else:
            val_loss = float("nan")
        writer.add_scalar("loss/train", train_loss, epoch)
        if val_loader is not None:
            writer.add_scalar("loss/val", val_loss, epoch)
        criterion_loss = val_loss if val_loader is not None else train_loss
        should_save = criterion_loss == criterion_loss and criterion_loss < best_metric  # not NaN
        if should_save:
            best_metric = criterion_loss
            os.makedirs(os.path.dirname(os.path.abspath(save_as)) or ".", exist_ok=True)
            torch.save(model.state_dict(), save_as)
            metric_payload = {
                "checkpoint_path": save_as,
                "experiment_name": experiment_name,
                "epoch": epoch,
                "selection_metric": best_metric_name,
                "best_metric": best_metric,
                "train_loss": train_loss,
                "val_loss": val_loss if val_loader is not None else None,
                "pretrained_checkpoint_path": pretrained_checkpoint_path,
                "config": config.__dict__,
                    "num_action_classes": NUM_ACTION_CLASSES,
                    "num_team_action_classes": NUM_TEAM_ACTION_CLASSES,
                "num_train_clips": len(train_clips),
                "num_val_clips": len(val_clips),
                "run_validation": config.run_validation,
            }
            if val_loader is not None:
                metric_payload["best_val_loss"] = best_metric  # backwards-compatible alias
            else:
                metric_payload["best_train_loss"] = best_metric
            write_checkpoint_metadata(save_as, metric_payload)
    return model


def train_from_dataset(
    save_as: str,
    dataset_root: str,
    pretrained_checkpoint_path: str | None = None,
    experiment_name: str = "custom_tdeed",
    config: TrainConfig | None = None,
) -> CustomTDeedModule:
    config = config or TrainConfig()
    records = load_dataset_records(dataset_root)
    clips = build_clips(
        records,
        clip_frames_count=config.clip_frames_count,
        overlap=config.overlap,
        accepted_gap=2,
    )
    if not clips:
        raise ValueError("No clips found. Run frame extraction before training.")
    return train_model(
        clips,
        save_as=save_as,
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        experiment_name=experiment_name,
        config=config,
    )


def run_epoch(
    model,
    loader,
    device,
    class_weights,
    optimizer=None,
    scaler=None,
    scheduler=None,
    acc_grad_iter: int = 1,
    epoch_index: int | None = None,
    nr_epochs: int | None = None,
    phase: str = "train",
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    if training:
        optimizer.zero_grad()
    context = torch.enable_grad() if training else torch.no_grad()
    if epoch_index is not None and nr_epochs is not None:
        tqdm_desc = f"{phase} epoch {epoch_index + 1}/{nr_epochs}"
    else:
        tqdm_desc = phase
    with context:
        for batch_idx, batch in enumerate(tqdm(loader, total=len(loader), desc=tqdm_desc)):
            use_cuda = device == "cuda"
            clip_tensor = batch["clip_tensor"]
            label_ids = batch["label_ids"]
            displacement = batch["displacement"]
            if clip_tensor.device.type != device:
                clip_tensor = clip_tensor.to(device, non_blocking=use_cuda)
            if label_ids.device.type != device:
                label_ids = label_ids.to(device, non_blocking=use_cuda)
            if displacement.device.type != device:
                displacement = displacement.to(device, non_blocking=use_cuda)
            clip_tensor = clip_tensor.float()
            label_ids = label_ids.long()
            displacement = displacement.float()
            with torch.amp.autocast(device_type=device, enabled=device == "cuda"):
                outputs = model(clip_tensor, inference=not training)
                logits = outputs["logits"].reshape(-1, NUM_TEAM_ACTION_CLASSES + 1)
                labels = label_ids.reshape(-1)
                cls_loss = F.cross_entropy(logits, labels, weight=class_weights)
                displ_loss = F.mse_loss(outputs["displacement"], displacement)
                loss = 1.5 * cls_loss + displ_loss
            total_loss += float(loss.detach().cpu())
            if training:
                backward_only = (batch_idx + 1) % acc_grad_iter != 0
                if scaler is None:
                    loss.backward()
                    if not backward_only:
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                else:
                    scaler.scale(loss).backward()
                    if not backward_only:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()
    return total_loss / max(1, len(loader))


def split_by_video(
    clips: list[VideoClip], train_split: float, random_seed: int
) -> tuple[list[VideoClip], list[VideoClip]]:
    by_video: dict[str, list[VideoClip]] = {}
    for clip in clips:
        by_video.setdefault(clip.source_video.video_id or clip.source_video.video_path, []).append(clip)
    video_ids = sorted(by_video)
    random.Random(random_seed).shuffle(video_ids)
    split_idx = max(1, int(len(video_ids) * train_split))
    train_ids = set(video_ids[:split_idx])
    train_clips = [clip for video_id in train_ids for clip in by_video[video_id]]
    val_clips = [clip for video_id in video_ids[split_idx:] for clip in by_video[video_id]]
    if not val_clips:
        val_clips = train_clips
    return train_clips, val_clips
