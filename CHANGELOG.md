# Changelog

All notable changes to this project should be documented in this file.

This project follows a simple versioned changelog format inspired by Keep a
Changelog.

## [0.1.0] - 2026-04-27

### Added

- Initial package structure for no-team custom ball action spotting.
- T-DEED style model, temporal shift layers, and SGP-Mixer head.
- Dataset discovery from clip folders (`dataset_root` + `ground_truth.json`), frame extraction, clip splitting, and dataset logic.
- Training, pretraining, posttraining, and single-video inference APIs.
- `custom-ballspotting` command-line interface.
- Example JSON configs for extraction, pretraining, posttraining, and inference.
