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

from custom_ballspotting.actions import (
    Action,
    NUM_ACTION_CLASSES,
    TRAINING_CE_RELATIVE_WEIGHTS,
)
from custom_ballspotting.checkpoints import (
    epoch_checkpoint_dirs,
    render_checkpoint_path,
    write_checkpoint_metadata,
)
from custom_ballspotting.data import (
    CustomTDeedDataset,
    VideoClip,
    build_clips,
    load_dataset_records,
)
from custom_ballspotting.eval import val_map
from custom_ballspotting.model.tdeed import CustomTDeedModule

def _log_val_per_class_map(
    *,
    writer: SummaryWriter,
    epoch: int,
    per_class: dict[str, float] | None,
) -> None:
    if not per_class:
        return
    for action in Action:
        writer.add_scalar(f"val/map_ap/{action.value}", per_class[action.value], epoch)
    writer.add_text(
        "val/per_class_map_mine",
        json.dumps(per_class, indent=2, sort_keys=True),
        epoch,
    )


def _print_val_per_class_map(delta_frames: int, per_class: dict[str, float]) -> None:
    label_w = max(len(a.value) for a in Action)
    print(f"  per-class AP @ {delta_frames}f:", flush=True)
    for action in Action:
        print(f"    {action.value:<{label_w}}  {per_class[action.value]:.6f}", flush=True)


# Non-TTY (e.g. PM2, systemd): plain step logs at most every ~N batches so logs stay usable.
_STEP_LOG_PLAIN_DIVISOR = 50
_STEP_LOG_PLAIN_CAP = 256


@dataclass
class TrainConfig:
    clip_frames_count: int = 100
    overlap: int = 88
    displacement_radius: int = 4
    features_model_name: str = "regnety_002"
    temporal_shift_mode: str = "gsf"
    n_layers: int = 2
    sgp_ks: int = 9
    sgp_k: int = 4
    gaussian_blur_kernel_size: int = 5
    nr_epochs: int = 25
    warm_up_epochs: int = 1
    learning_rate: float = 0.0003
    #: AdamW decoupled weight decay. Default ``0`` preserves older configs exactly.
    weight_decay: float = 0.0
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
    # Validation mAP: match dudek ``map_mine`` score pipeline (default) vs legacy ``score_video`` only.
    val_map_dudek_style_scoring: bool = True
    val_map_use_snms: bool = True
    val_map_snms_class_window: int = 12
    val_map_snms_threshold: float = 0.01
    #: Directory or zip with SoccerNet ``Labels-ball.json`` per game (``league/season/match/``).
    soccernet_path: str | None = None
    #: Run SoccerNet ``mAPevaluateTest`` / ``average_mAP`` during validation.
    val_run_soccernet_challenge_map: bool = False
    soccernet_challenge_metric: str = "at1"
    enforce_train_epoch_size: int | None = None
    enforce_val_epoch_size: int | None = None
    # Min interval between **plain-text** step lines when stderr is not a TTY (PM2, CI).
    # With a TTY, a tqdm bar is used instead. Effective plain interval is at least this
    # and at least max(1, min(256, batches // 50)).
    log_every_steps: int = 1
    # Print detailed timing for the first N train batches. This synchronizes CUDA
    # around timed sections, so keep it off except when diagnosing throughput.
    train_profile_steps: int = 0
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    #: Multiplies :data:`~custom_ballspotting.actions.TRAINING_CE_RELATIVE_WEIGHTS`
    #: for every foreground class; background CE weight is always ``1.0``.
    #: E.g. ``5.0`` with pass relative ``1.0`` → pass gets CE weight ``5.0`` vs bg ``1.0``.
    ce_foreground_scale: float = 5.0
    #: Save ``epochs/epoch_NNN.pt`` and ``metadata/epoch_NNN.metadata.json`` each epoch.
    save_epoch_checkpoints: bool = True


def _device_type(device: str) -> str:
    return torch.device(device).type


def train_model(
    clips: list[VideoClip],
    save_as: str,
    pretrained_checkpoint_path: str | None = None,
    experiment_name: str = "custom_tdeed",
    config: TrainConfig | None = None,
) -> CustomTDeedModule:
    config = config or TrainConfig()
    device_type = _device_type(config.device)
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
        device=config.device if device_type == "cuda" else None,
        profile_items=min(config.train_profile_steps * config.train_batch_size, 16),
    )
    val_dataset = (
        CustomTDeedDataset(
            val_clips,
            displacement_radius=config.displacement_radius,
            enforced_epoch_size=config.enforce_val_epoch_size,
            device=config.device if device_type == "cuda" else None,
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda") if device_type == "cuda" else None
    use_cuda = device_type == "cuda"
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
    run_log_dir = f"runs/{experiment_name}_{time.time()}"
    writer = SummaryWriter(log_dir=run_log_dir)
    writer.add_text("train/config", json.dumps(config.__dict__, indent=2, default=str), 0)

    epoch_summary_path = os.path.join(run_log_dir, "epoch_summary.log")

    # Action CE weight vector: background=1.0, then one weight per action.
    action_class_weights = torch.tensor(
        [1.0]
        + [
            config.ce_foreground_scale * TRAINING_CE_RELATIVE_WEIGHTS[action]
            for action in Action
        ],
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
    best_challenge_mAP = 0.0
    print(
        "Training started "
        f"train_clips={len(train_clips)} val_clips={len(val_clips)} "
        f"train_steps_per_epoch={len(train_loader)} "
        f"val_steps_per_epoch={len(val_loader) if val_loader is not None else 0} "
        f"eval_metric={config.eval_metric} map_start_epoch={config.map_start_epoch} "
        f"log_every_steps={config.log_every_steps} epoch_summary={epoch_summary_path}",
        flush=True,
    )
    train_start = time.perf_counter()
    epoch_summary_log_file = None
    try:
        epoch_summary_log_file = open(epoch_summary_path, "w", encoding="utf-8")
        epoch_summary_log_file.write(
            f"# training_started experiment={experiment_name} "
            f"t={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
        )
        epoch_summary_log_file.flush()
        for epoch in range(config.nr_epochs):
            print(f"Epoch {epoch + 1}/{config.nr_epochs}", flush=True)
            train_wall_start = time.perf_counter()
            train_loss = run_epoch(
                model,
                train_loader,
                config.device,
                action_class_weights,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                acc_grad_iter=config.acc_grad_iter,
                epoch_index=epoch,
                nr_epochs=config.nr_epochs,
                phase="train",
                writer=writer,
                log_every_steps=config.log_every_steps,
                profile_steps=config.train_profile_steps,
            )
            train_wall_s = time.perf_counter() - train_wall_start
            val_wall_s = 0.0
            if val_loader is not None:
                val_wall_start = time.perf_counter()
                val_loss = run_epoch(
                    model,
                    val_loader,
                    config.device,
                    action_class_weights,
                    epoch_index=epoch,
                    nr_epochs=config.nr_epochs,
                    phase="val",
                    writer=writer,
                    log_every_steps=config.log_every_steps,
                )
                val_wall_s = time.perf_counter() - val_wall_start
            else:
                val_loss = float("nan")

            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("timing/train_wall_s", train_wall_s, epoch)
            if val_loader is not None:
                writer.add_scalar("loss/val", val_loss, epoch)
                writer.add_scalar("timing/val_wall_s", val_wall_s, epoch)

            # mAP validation — ``map_mine`` plus optional SoccerNet ``mAPevaluateTest`` (dudek eval).
            epoch_map: float | None = None
            epoch_per_class_map: dict[str, float] | None = None
            epoch_challenge_map: float | None = None
            val_map_wall_s: float | None = None
            if use_map and epoch >= config.map_start_epoch:
                print(
                    f"  Computing val metrics (map_mine"
                    f"{' + SoccerNet challenge' if config.val_run_soccernet_challenge_map else ''}) "
                    f"@{config.map_delta_frames}f on {len(val_clips)} val clips …",
                    flush=True,
                )
                model.eval()
                map_wall_start = time.perf_counter()
                metrics = val_map(
                    model,
                    val_clips,
                    device=config.device,
                    val_batch_size=config.val_batch_size,
                    delta_frames=config.map_delta_frames,
                    dudek_style_scoring=config.val_map_dudek_style_scoring,
                    use_snms=config.val_map_use_snms,
                    snms_class_window=config.val_map_snms_class_window,
                    snms_threshold=config.val_map_snms_threshold,
                    soccernet_path=config.soccernet_path,
                    run_soccernet_challenge_map=config.val_run_soccernet_challenge_map,
                    soccernet_challenge_metric=config.soccernet_challenge_metric,
                )
                val_map_wall_s = time.perf_counter() - map_wall_start
                epoch_map = metrics.map_mine
                epoch_per_class_map = metrics.per_class_map
                epoch_challenge_map = metrics.challenge_mAP
                model.train()
                writer.add_scalar("val/map_mine", epoch_map, epoch)
                writer.add_scalar("val/map_wall_s", val_map_wall_s, epoch)
                if epoch_challenge_map is not None:
                    writer.add_scalar("val/challenge_mAP", epoch_challenge_map, epoch)
                if epoch_per_class_map is not None:
                    _log_val_per_class_map(
                        writer=writer,
                        epoch=epoch,
                        per_class=epoch_per_class_map,
                    )
                parts = [f"  map_mine={epoch_map:.6f}"]
                if epoch_challenge_map is not None:
                    parts.append(f"challenge_mAP={epoch_challenge_map:.6f}")
                parts.append(f"wall_s={val_map_wall_s:.2f}")
                print(" ".join(parts), flush=True)
                if epoch_per_class_map is not None:
                    _print_val_per_class_map(config.map_delta_frames, epoch_per_class_map)

            writer.flush()

            # Determine whether to save a new best checkpoint (dudek prefers challenge mAP when present).
            should_save = False
            if use_map and epoch >= config.map_start_epoch and epoch_map is not None:
                if epoch_challenge_map is not None:
                    should_save = epoch_challenge_map > best_challenge_mAP
                    if should_save:
                        best_challenge_mAP = epoch_challenge_map
                else:
                    should_save = epoch_map > best_map_metric
                    if should_save:
                        best_map_metric = epoch_map
            elif not (use_map and epoch >= config.map_start_epoch):
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
            lr_end = optimizer.param_groups[0]["lr"]
            summary_parts = [
                f"Epoch summary epoch={epochs_done}/{config.nr_epochs}",
                f"train_loss={train_loss:.6f}",
                f"train_wall_s={train_wall_s:.2f}",
                f"lr={lr_end:.6g}",
            ]
            if val_loader is not None:
                summary_parts.append(f"val_loss={val_loss:.6f}")
                summary_parts.append(f"val_wall_s={val_wall_s:.2f}")
            if epoch_map is not None:
                summary_parts.append(f"map_mine={epoch_map:.6f}")
            if epoch_per_class_map is not None:
                summary_parts.append(
                    "per_class_map_mine="
                    + json.dumps(epoch_per_class_map, sort_keys=True, separators=(",", ":"))
                )
            if epoch_challenge_map is not None:
                summary_parts.append(f"challenge_mAP={epoch_challenge_map:.6f}")
            if val_map_wall_s is not None:
                summary_parts.append(f"val_map_wall_s={val_map_wall_s:.2f}")
            if use_map and epoch >= config.map_start_epoch:
                summary_parts.append(f"best_map_mine={best_map_metric:.6f}")
                if config.val_run_soccernet_challenge_map:
                    summary_parts.append(f"best_challenge_mAP={best_challenge_mAP:.6f}")
            else:
                summary_parts.append(f"best_loss={best_loss_metric:.6f}")
            summary_parts.append(f"avg_epoch={_format_duration(avg_epoch_s)}")
            if remaining_epochs > 0:
                summary_parts.append(f"train_eta={_format_duration(train_eta_s)}")
            if should_save:
                summary_parts.append("★ best checkpoint saved")
            summary_line = " ".join(summary_parts)
            print(summary_line, flush=True)
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime())
            epoch_summary_log_file.write(ts + summary_line + "\n")
            epoch_summary_log_file.flush()

            if config.save_epoch_checkpoints:
                epochs_dir, meta_dir = epoch_checkpoint_dirs(save_as)
                os.makedirs(epochs_dir, exist_ok=True)
                os.makedirs(meta_dir, exist_ok=True)
                epoch_ckpt_name = f"epoch_{epochs_done:03d}.pt"
                epoch_ckpt_path = os.path.join(epochs_dir, epoch_ckpt_name)
                torch.save(model.state_dict(), epoch_ckpt_path)
                epoch_metric_payload = {
                    "checkpoint_kind": "epoch",
                    "checkpoint_path": epoch_ckpt_path,
                    "experiment_name": experiment_name,
                    "epoch": epoch,
                    "epoch_display": epochs_done,
                    "is_new_best": should_save,
                    "train_loss": train_loss,
                    "val_loss": val_loss if val_loader is not None else None,
                    "val_map_mine": epoch_map,
                    "val_per_class_map_mine": epoch_per_class_map,
                    "val_challenge_mAP": epoch_challenge_map,
                    "best_checkpoint_path": save_as,
                    "best_map_mine": best_map_metric,
                    "best_challenge_mAP": best_challenge_mAP,
                    "best_loss_metric": best_loss_metric,
                    "pretrained_checkpoint_path": pretrained_checkpoint_path,
                    "config": config.__dict__,
                    "head_type": "action_only",
                    "num_action_classes": NUM_ACTION_CLASSES,
                    "num_train_clips": len(train_clips),
                    "num_val_clips": len(val_clips),
                    "run_validation": config.run_validation,
                }
                meta_file = os.path.join(meta_dir, f"epoch_{epochs_done:03d}.metadata.json")
                write_checkpoint_metadata(
                    epoch_ckpt_path,
                    epoch_metric_payload,
                    metadata_file=meta_file,
                )

            if should_save:
                os.makedirs(os.path.dirname(os.path.abspath(save_as)) or ".", exist_ok=True)
                torch.save(model.state_dict(), save_as)
                if epoch_challenge_map is not None:
                    active_metric_name = "val_challenge_mAP"
                    active_best = best_challenge_mAP
                elif use_map and epoch >= config.map_start_epoch:
                    active_metric_name = "val_map_mine"
                    active_best = best_map_metric
                elif val_loader is not None:
                    active_metric_name = "val_loss"
                    active_best = best_loss_metric
                else:
                    active_metric_name = "train_loss"
                    active_best = best_loss_metric
                metric_payload = {
                    "checkpoint_kind": "best",
                    "checkpoint_path": save_as,
                    "experiment_name": experiment_name,
                    "epoch": epoch,
                    "selection_metric": active_metric_name,
                    "best_metric": active_best,
                    "train_loss": train_loss,
                    "val_loss": val_loss if val_loader is not None else None,
                    "val_map_mine": epoch_map,
                    "val_per_class_map_mine": epoch_per_class_map,
                    "val_challenge_mAP": epoch_challenge_map,
                    "pretrained_checkpoint_path": pretrained_checkpoint_path,
                    "config": config.__dict__,
                    "head_type": "action_only",
                    "num_action_classes": NUM_ACTION_CLASSES,
                    "num_train_clips": len(train_clips),
                    "num_val_clips": len(val_clips),
                    "run_validation": config.run_validation,
                }
                write_checkpoint_metadata(save_as, metric_payload)
    finally:
        if epoch_summary_log_file is not None:
            epoch_summary_log_file.close()
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
    action_class_weights,
    optimizer=None,
    scaler=None,
    scheduler=None,
    acc_grad_iter: int = 1,
    epoch_index: int | None = None,
    nr_epochs: int | None = None,
    phase: str = "train",
    writer: SummaryWriter | None = None,
    log_every_steps: int = 1,
    profile_steps: int = 0,
):
    training = optimizer is not None
    device_type = _device_type(device)
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
    n_batches = len(loader)
    use_tty = sys.stderr.isatty()
    plain_log_every = max(
        log_every_steps,
        max(1, min(_STEP_LOG_PLAIN_CAP, n_batches // _STEP_LOG_PLAIN_DIVISOR)),
    )
    with context:
        iterable = loader
        pbar = None
        if use_tty and n_batches > 0:
            iterable = tqdm(
                loader,
                total=n_batches,
                desc=tqdm_desc,
                mininterval=0.5,
                file=sys.stderr,
            )
            pbar = iterable
        prev_step_end = time.perf_counter()
        for batch_idx, batch in enumerate(iterable):
            batch_ready_t = time.perf_counter()
            use_cuda = device_type == "cuda"
            profile_this_step = training and batch_idx < profile_steps

            def _sync_if_profile() -> None:
                if profile_this_step and use_cuda:
                    torch.cuda.synchronize(device)

            clip_tensor = batch["clip_tensor"]
            action_label_ids = batch["action_label_ids"]
            displacement = batch["displacement"]
            move_start_t = time.perf_counter()
            if clip_tensor.device.type != device_type:
                clip_tensor = clip_tensor.to(device, non_blocking=use_cuda)
            if action_label_ids.device.type != device_type:
                action_label_ids = action_label_ids.to(device, non_blocking=use_cuda)
            if displacement.device.type != device_type:
                displacement = displacement.to(device, non_blocking=use_cuda)
            clip_tensor = clip_tensor.float()
            action_label_ids = action_label_ids.long()
            displacement = displacement.float()
            _sync_if_profile()
            forward_start_t = time.perf_counter()
            with torch.amp.autocast(device_type=device_type, enabled=use_cuda):
                outputs = model(clip_tensor, inference=not training)
                action_logits = outputs["action_logits"].reshape(-1, NUM_ACTION_CLASSES + 1)
                action_labels = action_label_ids.reshape(-1)
                cls_loss = F.cross_entropy(
                    action_logits,
                    action_labels,
                    weight=action_class_weights,
                )
                displ_loss = F.mse_loss(outputs["displacement"], displacement)
                loss = 1.5 * cls_loss + displ_loss
            _sync_if_profile()
            scalar_start_t = time.perf_counter()
            loss_value = float(loss.detach().cpu())
            cls_loss_value = float(cls_loss.detach().cpu())
            displ_loss_value = float(displ_loss.detach().cpu())
            scalar_end_t = time.perf_counter()
            total_loss += loss_value
            steps_done = batch_idx + 1
            running_loss = total_loss / steps_done
            backward_start_t = time.perf_counter()
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
            _sync_if_profile()
            backward_end_t = time.perf_counter()
            lr_now = optimizer.param_groups[0]["lr"] if optimizer is not None else None
            log_start_t = time.perf_counter()
            if writer is not None:
                global_step = (epoch_index or 0) * n_batches + batch_idx
                writer.add_scalar(f"loss_step/{phase}", loss_value, global_step)
                writer.add_scalar(f"loss_step/{phase}_running", running_loss, global_step)
                writer.add_scalar(f"loss_step/{phase}_cls", cls_loss_value, global_step)
                writer.add_scalar(
                    f"loss_step/{phase}_displacement", displ_loss_value, global_step
                )
                if training and lr_now is not None:
                    writer.add_scalar("train/lr", lr_now, global_step)
            log_end_t = time.perf_counter()
            elapsed = time.perf_counter() - epoch_start
            avg_step_s = elapsed / steps_done
            epoch_eta_s = avg_step_s * (n_batches - steps_done)
            if profile_this_step:
                total_profile_s = log_end_t - batch_ready_t
                data_wait_s = batch_ready_t - prev_step_end
                data_profile = ""
                profile_count = float(batch.get("profile_count", torch.tensor(0.0)).sum().item())
                if profile_count:
                    data_profile = (
                        f" data_label_avg_s={float(batch['profile_label_s'].sum().item()) / profile_count:.4f}"
                        f" data_read_avg_s={float(batch['profile_read_s'].sum().item()) / profile_count:.4f}"
                        f" data_stack_avg_s={float(batch['profile_stack_s'].sum().item()) / profile_count:.4f}"
                        f" data_move_avg_s={float(batch['profile_move_s'].sum().item()) / profile_count:.4f}"
                        f" data_augment_avg_s={float(batch['profile_augment_s'].sum().item()) / profile_count:.4f}"
                        f" data_total_avg_s={float(batch['profile_total_s'].sum().item()) / profile_count:.4f}"
                        f" data_profile_items={profile_count:.0f}"
                    )
                print(
                    "train profile "
                    f"epoch={epoch_index + 1 if epoch_index is not None else '?'} "
                    f"step={steps_done}/{n_batches} "
                    f"shape={tuple(clip_tensor.shape)} "
                    f"dtype={clip_tensor.dtype} "
                    f"device={clip_tensor.device} "
                    f"data_wait_s={data_wait_s:.4f} "
                    f"move_s={forward_start_t - move_start_t:.4f} "
                    f"forward_loss_s={scalar_start_t - forward_start_t:.4f} "
                    f"scalar_sync_s={scalar_end_t - scalar_start_t:.4f} "
                    f"backward_optim_s={backward_end_t - backward_start_t:.4f} "
                    f"tb_log_s={log_end_t - log_start_t:.4f} "
                    f"total_after_data_s={total_profile_s:.4f}"
                    f"{data_profile}",
                    flush=True,
                )
            if pbar is not None:
                postfix = {
                    "loss": f"{loss_value:.4f}",
                    "run": f"{running_loss:.4f}",
                    "cls": f"{cls_loss_value:.4f}",
                    "disp": f"{displ_loss_value:.4f}",
                    "t": f"{avg_step_s:.2f}s",
                    "eta": _format_duration(epoch_eta_s),
                }
                if lr_now is not None:
                    postfix["lr"] = f"{lr_now:.1e}"
                pbar.set_postfix(postfix, refresh=False)
            elif n_batches > 0 and (
                steps_done % plain_log_every == 0 or steps_done == n_batches
            ):
                print(
                    _format_step_log(
                        phase=phase,
                        epoch_index=epoch_index,
                        nr_epochs=nr_epochs,
                        batch_idx=batch_idx,
                        num_batches=n_batches,
                        loss=loss_value,
                        running_loss=running_loss,
                        cls_loss=cls_loss_value,
                        displ_loss=displ_loss_value,
                        lr=lr_now,
                        avg_step_s=avg_step_s,
                        epoch_eta_s=epoch_eta_s,
                    ),
                    flush=True,
                )
            prev_step_end = time.perf_counter()
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
