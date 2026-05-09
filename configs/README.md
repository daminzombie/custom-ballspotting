# Example Configs

Primary 720p workflow:

- `extract_frames.example.json`
- `final_posttrain_from_tdeed.example.json`
- `inference.example.json`

Equivalent explicit 720p aliases:

- `extract_frames_720p.example.json`
- `final_posttrain_from_tdeed_720p.example.json`
- `inference_720p.example.json`

The default configs now use 1280x720 frames with `clip_frames_count=100` and
`overlap=88` to match the T-DEED checkpoint shape and increase training samples.
They use batch size 1 plus gradient accumulation because 720p frames are much
heavier than low-resolution crops.

Advanced/experimental configs:

- `pretrain.example.json`: only use if your source data already uses the final custom labels.
- `posttrain_from_tdeed.example.json`: older generic example kept for reference.
- `posttrain_from_custom.example.json`: only useful after adding explicit full-checkpoint resume or staged fine-tuning behavior.

For the current product plan, use `final_posttrain_from_tdeed.example.json`.
