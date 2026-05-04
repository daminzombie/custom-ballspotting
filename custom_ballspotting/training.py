import json
import os
import random
import sys
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
from custom_ballspotting.eval import val_map
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
    train_split: float = 0.9  # used only when run_validation is true
    run_validation: bool = True  # select checkpoints by held-out validation metric by default
    eval_metric: str = "map"  # "map" or "loss"; "map" requires run_validation=True
    map_delta_frames: int = 5  # frame-count tolerance for mAP TP matching
    map_start_epoch: int = 3  # skip mAP eval for early epochs; fall back to val_loss before this
    enforce_train_epoch_size: int | None = None
    enforce_val_epoch_size: int | None = None
    log_every_steps: int = 1
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
    use_map = config.eval_metric == "map" and config.run_validation and val_loader is not None
    if config.eval_metric == "map" and not config.run_validation:
        print(
            "Warning: eval_metric='map' requires run_validation=True; falling back to 'loss'.",
            flush=True,
        )
        use_map = False

    # best_metric direction depends on the active criterion:
    #   loss → minimise (start at +inf); map → maximise (start at 0).
    # Before map_start_epoch we also fall back to val_loss so the first useful
    # checkpoint is not withheld until mAP evaluation kicks in.
    best_loss_metric = float("inf")       # tracks best loss regardless of eval_metric
    best_map_metric = 0.0
    best_metric_name = (
        "val_map" if use_map else
        ("val_loss" if config.run_validation and val_loader is not None else "train_loss")
    )
    print(
        "Training started "
        f"train_clips={len(train_clips)} val_clips={len(val_clips)} "
        f"train_steps_per_epoch={len(train_loader)} "
        f"val_steps_per_epoch={len(val_loader) if val_loader is not None else 0} "
        f"eval_metric={config.eval_metric} map_start_epoch={config.map_start_epoch} "
        f"log_every_steps={config.log_every_steps}",
        flush=True,
    )
    train_start = time.perf_counter()
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
            writer=writer,
            log_every_steps=config.log_every_steps,
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
                writer=writer,
                log_every_steps=config.log_every_steps,
            )
        else:
            val_loss = float("nan")

        writer.add_scalar("loss/train", train_loss, epoch)
        if val_loader is not None:
            writer.add_scalar("loss/val", val_loss, epoch)

        # mAP validation — only when configured and past warm-up period
        epoch_map: float | None = None
        if use_map and epoch >= config.map_start_epoch:
            print(
                f"  Computing mAP@{config.map_delta_frames}f on {len(val_clips)} val clips …",
                flush=True,
            )
            model.eval()
            epoch_map = val_map(
                model,
                val_clips,
                device=config.device,
                val_batch_size=config.val_batch_size,
                delta_frames=config.map_delta_frames,
            )
            model.train()
            writer.add_scalar("val/map", epoch_map, epoch)
            print(f"  val_map={epoch_map:.6f}", flush=True)

        writer.flush()

        # Determine whether to save a new best checkpoint
        if use_map and epoch >= config.map_start_epoch and epoch_map is not None:
            should_save = epoch_map > best_map_metric
            if should_save:
                best_map_metric = epoch_map
        else:
            # Loss-based fallback: covers (a) eval_metric="loss" and (b) early
            # epochs before mAP kicks in when eval_metric="map".
            criterion_loss = val_loss if val_loader is not None else train_loss
            should_save = criterion_loss == criterion_loss and criterion_loss < best_loss_metric
            if should_save:
                best_loss_metric = criterion_loss

        epochs_done = epoch + 1
        total_elapsed = time.perf_counter() - train_start
        avg_epoch_s = total_elapsed / epochs_done
        remaining_epochs = config.nr_epochs - epochs_done
        train_eta_s = avg_epoch_s * remaining_epochs
        summary_parts = [
            f"Epoch summary epoch={epochs_done}/{config.nr_epochs}",
            f"train_loss={train_loss:.6f}",
        ]
        if val_loader is not None:
            summary_parts.append(f"val_loss={val_loss:.6f}")
        if epoch_map is not None:
            summary_parts.append(f"val_map={epoch_map:.6f}")
        summary_parts.append(f"epoch={_format_duration(avg_epoch_s)}")
        if remaining_epochs > 0:
            summary_parts.append(f"train_eta={_format_duration(train_eta_s)}")
        if should_save:
            summary_parts.append("★ checkpoint saved")
        print(" ".join(summary_parts), flush=True)

        if should_save:
            os.makedirs(os.path.dirname(os.path.abspath(save_as)) or ".", exist_ok=True)
            torch.save(model.state_dict(), save_as)
            active_metric_name = (
                "val_map" if (use_map and epoch >= config.map_start_epoch)
                else ("val_loss" if val_loader is not None else "train_loss")
            )
            active_best = (
                best_map_metric if (use_map and epoch >= config.map_start_epoch)
                else best_loss_metric
            )
            metric_payload = {
                "checkpoint_path": save_as,
                "experiment_name": experiment_name,
                "epoch": epoch,
                "selection_metric": active_metric_name,
                "best_metric": active_best,
                "train_loss": train_loss,
                "val_loss": val_loss if val_loader is not None else None,
                "val_map": epoch_map,
                "pretrained_checkpoint_path": pretrained_checkpoint_path,
                "config": config.__dict__,
                "num_action_classes": NUM_ACTION_CLASSES,
                "num_team_action_classes": NUM_TEAM_ACTION_CLASSES,
                "num_train_clips": len(train_clips),
                "num_val_clips": len(val_clips),
                "run_validation": config.run_validation,
            }
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
    writer: SummaryWriter | None = None,
    log_every_steps: int = 1,
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    log_every_steps = max(1, log_every_steps)
    if training:
        optimizer.zero_grad()
    context = torch.enable_grad() if training else torch.no_grad()
    if epoch_index is not None and nr_epochs is not None:
        tqdm_desc = f"{phase} epoch {epoch_index + 1}/{nr_epochs}"
    else:
        tqdm_desc = phase
    epoch_start = time.perf_counter()
    with context:
        progress = tqdm(
            loader,
            total=len(loader),
            desc=tqdm_desc,
            disable=not sys.stderr.isatty(),
        )
        for batch_idx, batch in enumerate(progress):
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
            loss_value = float(loss.detach().cpu())
            cls_loss_value = float(cls_loss.detach().cpu())
            displ_loss_value = float(displ_loss.detach().cpu())
            total_loss += loss_value
            running_loss = total_loss / (batch_idx + 1)
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
            if writer is not None:
                global_step = (epoch_index or 0) * len(loader) + batch_idx
                writer.add_scalar(f"loss_step/{phase}", loss_value, global_step)
                writer.add_scalar(f"loss_step/{phase}_running", running_loss, global_step)
                writer.add_scalar(f"loss_step/{phase}_cls", cls_loss_value, global_step)
                writer.add_scalar(f"loss_step/{phase}_displacement", displ_loss_value, global_step)
            if (batch_idx + 1) % log_every_steps == 0 or (batch_idx + 1) == len(loader):
                lr = optimizer.param_groups[0]["lr"] if optimizer is not None else None
                steps_done = batch_idx + 1
                elapsed = time.perf_counter() - epoch_start
                avg_step_s = elapsed / steps_done
                epoch_eta_s = avg_step_s * (len(loader) - steps_done)
                print(
                    _format_step_log(
                        phase=phase,
                        epoch_index=epoch_index,
                        nr_epochs=nr_epochs,
                        batch_idx=batch_idx,
                        num_batches=len(loader),
                        loss=loss_value,
                        running_loss=running_loss,
                        cls_loss=cls_loss_value,
                        displ_loss=displ_loss_value,
                        lr=lr,
                        avg_step_s=avg_step_s,
                        epoch_eta_s=epoch_eta_s,
                    ),
                    flush=True,
                )
    return total_loss / max(1, len(loader))


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as H:MM:SS or M:SS."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _format_step_log(
    *,
    phase: str,
    epoch_index: int | None,
    nr_epochs: int | None,
    batch_idx: int,
    num_batches: int,
    loss: float,
    running_loss: float,
    cls_loss: float,
    displ_loss: float,
    lr: float | None,
    avg_step_s: float | None = None,
    epoch_eta_s: float | None = None,
) -> str:
    epoch_value = f"{epoch_index + 1}/{nr_epochs}" if epoch_index is not None and nr_epochs is not None else "unknown"
    lr_value = f" lr={lr:.8f}" if lr is not None else ""
    timing_value = ""
    if avg_step_s is not None:
        timing_value += f" step={avg_step_s:.2f}s"
    if epoch_eta_s is not None:
        timing_value += f" eta={_format_duration(epoch_eta_s)}"
    return (
        f"{phase} step "
        f"epoch={epoch_value} "
        f"step={batch_idx + 1}/{num_batches} "
        f"loss={loss:.6f} "
        f"running_loss={running_loss:.6f} "
        f"cls_loss={cls_loss:.6f} "
        f"displacement_loss={displ_loss:.6f}"
        f"{lr_value}"
        f"{timing_value}"
    )


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
