import click

from custom_ballspotting.config import (
    dataclass_from_dict,
    load_json_config,
    merge_values,
    resolve_config_path,
)
from custom_ballspotting.checkpoints import render_checkpoint_path
from custom_ballspotting.data import find_first_mp4, load_dataset_records
from custom_ballspotting.inference import infer_video as infer_video_fn, infer_video_param_names
from custom_ballspotting.training import TrainConfig, train_from_dataset


@click.group()
def cli():
    """Custom team ball action spotting CLI."""


@cli.command("extract-frames")
@click.option("--config", "config_path", type=str, required=True)
@click.option("--stride", type=int, default=None)
@click.option("--frame_target_width", type=int, default=None)
@click.option("--frame_target_height", type=int, default=None)
@click.option("--radius_seconds", type=int, default=None)
@click.option("--save_all", type=bool, default=None)
def extract_frames(
    config_path: str,
    stride: int | None,
    frame_target_width: int | None,
    frame_target_height: int | None,
    radius_seconds: int | None,
    save_all: bool | None,
):
    values = merge_values(
        load_json_config(config_path),
        {
            "stride": stride,
            "frame_target_width": frame_target_width,
            "frame_target_height": frame_target_height,
            "radius_seconds": radius_seconds,
            "save_all": save_all,
        },
    )
    dataset_root = resolve_config_path(_required(values, "dataset_root"), config_path)
    for record in load_dataset_records(dataset_root):
        record.extract_frames(
            stride=values.get("stride", 2),
            target_width=values.get("frame_target_width", 1280),
            target_height=values.get("frame_target_height", 720),
            radius_seconds=values.get("radius_seconds"),
            save_all=values.get("save_all", False),
        )


def _train_options(command):
    command = click.option("--config", "config_path", type=str, required=True)(command)
    command = click.option("--pretrained_checkpoint_path", type=str, default=None)(command)
    command = click.option("--save_as", type=str, default=None)(command)
    command = click.option("--experiment_name", type=str, default=None)(command)
    command = click.option("--clip_frames_count", type=int, default=None)(command)
    command = click.option("--overlap", type=int, default=None)(command)
    command = click.option("--displacement_radius", type=int, default=None)(command)
    command = click.option("--features_model_name", type=str, default=None)(command)
    command = click.option("--temporal_shift_mode", type=str, default=None)(command)
    command = click.option("--n_layers", type=int, default=None)(command)
    command = click.option("--sgp_ks", type=int, default=None)(command)
    command = click.option("--sgp_k", type=int, default=None)(command)
    command = click.option("--gaussian_blur_kernel_size", type=int, default=None)(command)
    command = click.option("--nr_epochs", type=int, default=None)(command)
    command = click.option("--warm_up_epochs", type=int, default=None)(command)
    command = click.option("--learning_rate", type=float, default=None)(command)
    command = click.option("--train_batch_size", type=int, default=None)(command)
    command = click.option("--val_batch_size", type=int, default=None)(command)
    command = click.option("--acc_grad_iter", type=int, default=None)(command)
    command = click.option("--flip_proba", type=float, default=None)(command)
    command = click.option("--camera_move_proba", type=float, default=None)(command)
    command = click.option("--crop_proba", type=float, default=None)(command)
    command = click.option("--even_choice_proba", type=float, default=None)(command)
    command = click.option("--train_split", type=float, default=None)(command)
    command = click.option("--enforce_train_epoch_size", type=int, default=None)(command)
    command = click.option("--enforce_val_epoch_size", type=int, default=None)(command)
    command = click.option("--random_seed", type=int, default=None)(command)
    command = click.option("--device", type=str, default=None)(command)
    command = click.option(
        "--run-validation/--no-run-validation",
        "run_validation",
        default=None,
        help="Use a train/val split and select best checkpoint by val loss; "
        "default off (matches dudek train-challenge). Omitted keeps JSON default.",
    )(command)
    return command


@cli.command("train")
@_train_options
def train(**kwargs):
    _run_train_command(require_pretrained=False, force_no_pretrained=False, **kwargs)


@cli.command("pretrain")
@_train_options
def pretrain(**kwargs):
    """Train from dataset_root without loading a pretrained checkpoint."""
    _run_train_command(require_pretrained=False, force_no_pretrained=True, **kwargs)


@cli.command("posttrain")
@_train_options
def posttrain(**kwargs):
    """Fine-tune from an existing T-DEED/custom-ballspotting checkpoint."""
    _run_train_command(require_pretrained=True, force_no_pretrained=False, **kwargs)


def _run_train_command(require_pretrained: bool, force_no_pretrained: bool, **kwargs):
    config_path = kwargs.pop("config_path")
    config_values = load_json_config(config_path)
    values = merge_values(config_values, kwargs)
    dataset_root = resolve_config_path(_required(values, "dataset_root"), config_path)
    experiment_name = values.get("experiment_name", "custom_tdeed")
    save_as_template = values.get("save_as")
    if save_as_template is not None:
        save_as_template = resolve_config_path(save_as_template, config_path)
    save_as = render_checkpoint_path(save_as_template, experiment_name=experiment_name)
    pretrained_checkpoint_path = None if force_no_pretrained else values.get("pretrained_checkpoint_path")
    if pretrained_checkpoint_path:
        pretrained_checkpoint_path = resolve_config_path(pretrained_checkpoint_path, config_path)
    if require_pretrained and not pretrained_checkpoint_path:
        raise click.ClickException("--pretrained_checkpoint_path is required for posttrain")
    config = dataclass_from_dict(TrainConfig, values)
    train_from_dataset(
        save_as=save_as,
        dataset_root=dataset_root,
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        experiment_name=experiment_name,
        config=config,
    )
    click.echo(f"Saved best checkpoint to {save_as}")


@cli.command("infer-video")
@click.option("--config", "config_path", type=str, default=None)
@click.option("--video_path", type=str, default=None, help="Video file for inference.")
@click.option(
    "--video_dir",
    type=str,
    default=None,
    help="Directory containing a single .mp4 (uses lexicographically first *.mp4).",
)
@click.option("--model_checkpoint_path", type=str, default=None)
@click.option("--output_path", type=str, default=None)
@click.option("--clip_frames_count", type=int, default=None)
@click.option("--overlap", type=int, default=None)
@click.option("--stride", type=int, default=None)
@click.option("--frame_target_width", type=int, default=None)
@click.option("--frame_target_height", type=int, default=None)
@click.option("--features_model_name", type=str, default=None)
@click.option("--temporal_shift_mode", type=str, default=None)
@click.option("--n_layers", type=int, default=None)
@click.option("--sgp_ks", type=int, default=None)
@click.option("--sgp_k", type=int, default=None)
@click.option("--gaussian_blur_kernel_size", type=int, default=None)
@click.option("--val_batch_size", type=int, default=None)
@click.option("--inference_threshold", type=float, default=None)
@click.option("--extract_frames", type=bool, default=None)
@click.option("--device", type=str, default=None)
@click.option(
    "--num_workers",
    type=int,
    default=None,
    help="DataLoader worker processes for prefetching clips during inference. "
    "2 overlaps CPU data loading with GPU compute on multi-core machines. Default 0.",
)
@click.option(
    "--frame_write_workers",
    type=int,
    default=None,
    help="Threads for parallel frame resize+write during extraction. "
    "Default 8 saturates multi-core CPUs.",
)
def infer_video(config_path: str | None, **kwargs):
    values = merge_values(load_json_config(config_path), kwargs)
    video_dir = values.get("video_dir")
    video_path = values.get("video_path")
    if video_dir is not None:
        video_dir_resolved = resolve_config_path(video_dir, config_path)
        mp4 = find_first_mp4(video_dir_resolved)
        if mp4 is None:
            raise click.ClickException(f"No .mp4 file found in {video_dir_resolved}")
        values["video_path"] = mp4
    elif video_path is not None:
        values["video_path"] = resolve_config_path(video_path, config_path)
    else:
        raise click.ClickException("Provide --video_path or --video_dir")
    values.pop("video_dir", None)
    values["model_checkpoint_path"] = resolve_config_path(
        _required(values, "model_checkpoint_path"), config_path
    )
    values["output_path"] = resolve_config_path(
        values.get("output_path", "predictions.json"), config_path
    )
    infer_params = infer_video_param_names()
    filtered = {k: v for k, v in values.items() if k in infer_params}
    result = infer_video_fn(**filtered)
    click.echo(f"Saved {len(result['predictions'])} predictions to {values['output_path']}")


def _required(values: dict, key: str):
    value = values.get(key)
    if value is None:
        raise click.ClickException(f"Missing required option: --{key}")
    return value
