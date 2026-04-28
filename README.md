# custom-ballspotting

`custom-ballspotting` is a reusable Python package for no-team ball action spotting with custom labels. It keeps the useful T-DEED/DUDEK ideas: a RegNet + temporal shift backbone, SGP-Mixer temporal head, displacement loss, class weighting, frame/clip sampling, and video augmentations. It removes SoccerNet team-specific assumptions, so the model output is:

```text
background + N custom actions
```

For the current action set, `N = 15`, so the classifier has `16` classes.

## Package Design

The package is library-first and CLI-second:

```text
custom_ballspotting/
  actions.py       # custom action vocabulary, weights, thresholds, tolerances
  data.py          # manifest loading, frame extraction, clip creation, dataset
  training.py      # reusable training API
  inference.py     # reusable inference API
  cli.py           # thin command-line wrapper
  model/
    tdeed.py       # no-team T-DEED model
    layers.py      # SGP-Mixer and temporal shift layers
    shift.py
```

You can use it from another Python project:

```python
from custom_ballspotting.training import TrainConfig, train_from_manifest
from custom_ballspotting.inference import infer_video
```

or from the terminal:

```bash
custom-ballspotting --help
```

## Install

From this directory:

```bash
uv pip install -e . --python ../dude.k/.venv/Scripts/python.exe
```

Or with a normal Python environment that has `pip`:

```bash
pip install -e .
```

After install:

```bash
custom-ballspotting --help
```

On this Windows workspace, if you are reusing the DUDEK venv, call:

```bash
../dude.k/.venv/Scripts/custom-ballspotting.exe --help
```

## Custom Actions

The action vocabulary is defined in `custom_ballspotting/actions.py`:

```python
class Action(str, Enum):
    PASS = "pass"
    PASS_RECEIVED = "pass_received"
    RECOVERY = "recovery"
    TACKLE = "tackle"
    INTERCEPTION = "interception"
    BALL_OUT_OF_PLAY = "ball_out_of_play"
    CLEARANCE = "clearance"
    TAKE_ON = "take_on"
    SUBSTITUTION = "substitution"
    BLOCK = "block"
    AERIAL_DUEL = "aerial_duel"
    SHOT = "shot"
    SAVE = "save"
    FOUL = "foul"
    GOAL = "goal"
```

Each class also has an `ActionConfig`:

```python
ActionConfig(weight, min_score, tolerance_seconds)
```

Those values are used for training class weights and inference filtering/NMS. For rare but important classes like `goal`, `foul`, or `save`, a higher class weight is useful when the dataset is small.

## Annotation Manifest

Training data is described by a JSON manifest. Use either a top-level list or a top-level object with `videos`.

```json
{
  "videos": [
    {
      "video_id": "match_001",
      "video_path": "videos/match_001.mp4",
      "annotations": [
        {"position": 14240, "label": "pass"},
        {"position": 15280, "label": "recovery"},
        {"position": 25040, "label": "shot"}
      ]
    }
  ]
}
```

Rules:

- `position` is milliseconds from the start of the video.
- `label` must match one of the values in `Action`.
- `video_path` can be absolute or relative to the manifest file.
- `video_id` is optional but recommended for stable train/validation splitting.
- There is no team field.

## Frame Extraction

Frames must be extracted before training. The extractor saves frames next to each video under `.frames_<video_name>/`.

Using flags:

```bash
custom-ballspotting extract-frames \
  --manifest_path="../data/custom/train.json" \
  --stride=2 \
  --frame_target_width=1280 \
  --frame_target_height=720 \
  --radius_seconds=8
```

Using a config file:

```bash
custom-ballspotting extract-frames --config configs/extract_frames.example.json
```

For training, you usually do not need every frame. `radius_seconds=8` keeps frames around annotations plus stride frames, which is more storage-efficient for sparse action spotting. For inference on a full video, frames are extracted with `save_all=true` internally.

## Config Files

All main commands accept `--config <json-file>`. CLI flags override values from the config.

Example configs are provided:

```text
configs/extract_frames.example.json
configs/final_posttrain_from_tdeed.example.json
configs/inference.example.json
configs/extract_frames_720p.example.json
configs/final_posttrain_from_tdeed_720p.example.json
configs/inference_720p.example.json
```

Paths inside config files are resolved relative to the config file location.

## Input Resolution

The default workflow extracts 720p frames:

```json
{
  "frame_target_width": 1280,
  "frame_target_height": 720
}
```

This matches your final dataset style: source videos are commonly 1280x720 or
1920x1080, and ball/contact detail can be too small after resizing to 224.

The default training clip settings are:

```json
{
  "clip_frames_count": 100,
  "overlap": 88
}
```

These are intentional: the T-DEED SoccerNetBall checkpoint was trained with
100-frame clips, and `overlap=88` increases the number of training samples.

The 720p tradeoff is large:

- much higher GPU memory,
- slower frame extraction,
- slower training/inference,
- larger extracted-frame folders.

The default configs therefore use small per-step batches and gradient accumulation:

```json
{
  "train_batch_size": 1,
  "val_batch_size": 1,
  "acc_grad_iter": 8
}
```

If your GPU runs out of memory at 720p, first try lowering
`train_batch_size`/`val_batch_size` to 1 if they are not already, then reduce
augmentation. Lowering `clip_frames_count` should be treated as a separate model
shape change because checkpoint compatibility is best with `clip_frames_count=100`.

## Chosen Training Strategy

For this project, the recommended first workflow is deliberately simple:

```text
T-DEED SoccerNetBall checkpoint
  -> long posttraining on the final custom product dataset
  -> final custom product checkpoint
```

We are **not** doing continued pretraining on old broadcast action-spotting classes for now, because those labels do not map cleanly to the final custom ball-action labels. Training on mismatched labels would require a separate temporary label schema/head and is intentionally out of the primary workflow.

So the main training command is:

```bash
custom-ballspotting posttrain --config configs/final_posttrain_from_tdeed.example.json
```

This is "posttraining" because it starts from an existing T-DEED checkpoint and adapts it to your final product dataset.

## Checkpoint Storage

All project checkpoints should live under the central repo folder:

```text
checkpoints/
```

The folder is local-only and ignored by `.gitignore`. It is not meant to be
committed to this repository. When a checkpoint is ready to share or deploy,
upload the selected `.pt` file and its `.metadata.json` sidecar to your
Hugging Face model repository.

Training configs use timestamped names by default:

```json
{
  "save_as": "../checkpoints/{experiment_name}_{timestamp}_best.pt"
}
```

At runtime this becomes a clear, unique path like:

```text
checkpoints/custom_final_product_posttrain_720p_20260428_073012_best.pt
```

For each saved best checkpoint, the trainer also writes a sidecar metadata file:

```text
checkpoints/custom_final_product_posttrain_720p_20260428_073012_best.metadata.json
```

The metadata records the experiment name, epoch, validation loss, source
checkpoint, training config, and number of train/validation clips.

For inference, set `model_checkpoint_path` to the exact timestamped checkpoint
you want to evaluate. This is intentional: it avoids silently using the wrong
run when several experiments exist.

## Posttrain From T-DEED/SoccerNetBall

Use the T-DEED checkpoint as a backbone initializer and train a new no-team custom head on your final product dataset:

```bash
custom-ballspotting posttrain --config configs/final_posttrain_from_tdeed.example.json
```

Important: the architecture settings must match the checkpoint backbone:

```json
{
  "features_model_name": "regnety_008",
  "temporal_shift_mode": "gsf",
  "clip_frames_count": 100,
  "n_layers": 2,
  "sgp_ks": 9,
  "sgp_k": 4
}
```

The T-DEED checkpoint is loaded with `load_backbone()`, so only these parts are transferred:

```text
_features.*
_temp_fine.*
```

The custom classifier head is new:

```text
background + 15 custom actions
```

This is the recommended path when your custom dataset is small.

The default final config runs longer than a smoke test:

```json
{
  "nr_epochs": 30,
  "train_batch_size": 2,
  "val_batch_size": 2,
  "acc_grad_iter": 4,
  "even_choice_proba": 0.25
}
```

That gives lower GPU memory pressure while still accumulating gradients over multiple steps.

## Optional Second-Stage Fine-Tuning

Do not run the same dataset twice with the same settings. That is just more epochs.

If you later want a second stage on the same final product dataset, make it intentionally different, for example:

- lower `learning_rate`,
- fewer augmentations,
- smaller enforced epoch sizes for quick iteration,
- a future `head-only` or `freeze_backbone` mode.

At the moment, the package trains the full model during posttraining. It does not yet expose a head-only freeze mode.

## Memory-Efficient Training

For smaller GPUs, lower batch size and use gradient accumulation:

```json
{
  "train_batch_size": 1,
  "val_batch_size": 1,
  "acc_grad_iter": 8
}
```

This keeps the effective batch size similar while reducing GPU memory.

For quick smoke tests:

```json
{
  "nr_epochs": 2,
  "enforce_train_epoch_size": 500,
  "enforce_val_epoch_size": 100
}
```

## Inference

From a posttrained checkpoint:

```bash
custom-ballspotting infer-video --config configs/inference.example.json
```

For a 720p-trained checkpoint:

```bash
custom-ballspotting infer-video --config configs/inference_720p.example.json
```

Or direct:

```bash
custom-ballspotting infer-video \
  --video_path="../data/videos/sample.mp4" \
  --model_checkpoint_path="../checkpoints/custom_final_product_posttrain_720p_YYYYMMDD_HHMMSS_best.pt" \
  --output_path="../predictions/sample_predictions.json" \
  --clip_frames_count=100 \
  --overlap=88 \
  --stride=2 \
  --frame_target_width=1280 \
  --frame_target_height=720 \
  --inference_threshold=0.2
```

Input video resolution can be 1920x1080 or any normal video size. Frames are resized to the configured target size before inference.
For a model trained with 720p extracted frames, inference should use the same
`frame_target_width`, `frame_target_height`, `clip_frames_count`, and `overlap`
as training.

Output format:

```json
{
  "video_path": "videos/sample.mp4",
  "predictions": [
    {
      "label": "pass",
      "position": 14240,
      "gameTime": "1 - 00:14",
      "confidence": 0.78
    }
  ]
}
```

Inference applies per-class `min_score` and `tolerance_seconds` from `ACTION_CONFIGS`.

## Python API

Training from another project:

```python
from custom_ballspotting.training import TrainConfig, train_from_manifest

config = TrainConfig(
    clip_frames_count=100,
    overlap=88,
    nr_epochs=10,
    train_batch_size=1,
    val_batch_size=1,
    acc_grad_iter=8,
)

train_from_manifest(
    manifest_path="data/custom/posttrain.json",
    pretrained_checkpoint_path="checkpoints/tdeed_checkpoint_best.pt",
    save_as="checkpoints/{experiment_name}_{timestamp}_best.pt",
    experiment_name="my_custom_posttrain",
    config=config,
)
```

Inference from another project:

```python
from custom_ballspotting.inference import infer_video

result = infer_video(
    video_path="videos/sample.mp4",
    model_checkpoint_path="checkpoints/my_custom_posttrain_YYYYMMDD_HHMMSS_best.pt",
    output_path="predictions/sample.json",
)
```

Using lower-level APIs:

```python
from custom_ballspotting.data import load_manifest, build_clips
from custom_ballspotting.training import train_model, TrainConfig

records = load_manifest("data/custom/train.json")
clips = build_clips(records, clip_frames_count=100, overlap=88)
train_model(
    clips,
    save_as="checkpoints/{experiment_name}_{timestamp}_best.pt",
    config=TrainConfig(nr_epochs=5),
)
```

## Recommended Workflow

1. Define or update `custom_ballspotting/actions.py`.
2. Export your annotations to the manifest format.
3. Extract frames for training manifests.
4. Posttrain from a strong T-DEED/SoccerNetBall checkpoint.
5. Run inference on sample videos.
6. Review false positives/negatives and adjust:
   - annotation quality,
   - `ActionConfig.min_score`,
   - `ActionConfig.tolerance_seconds`,
   - class weights,
   - augmentation settings.

For your current situation, start with:

```bash
custom-ballspotting extract-frames --config configs/extract_frames.example.json
custom-ballspotting posttrain --config configs/final_posttrain_from_tdeed.example.json
custom-ballspotting infer-video --config configs/inference.example.json
```

Equivalent explicit 720p aliases are also available:

```bash
custom-ballspotting extract-frames --config configs/extract_frames_720p.example.json
custom-ballspotting posttrain --config configs/final_posttrain_from_tdeed_720p.example.json
custom-ballspotting infer-video --config configs/inference_720p.example.json
```
