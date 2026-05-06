"""SoccerNet challenge-style mAP (``average_mAP`` / ``mAPevaluateTest``), aligned with dudek.

Requires the ``soccernet`` package: ``pip install 'custom-ballspotting[challenge]'`` or ``pip install soccernet``.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import zipfile

import numpy as np

# ---------------------------------------------------------------------------
# Vendored from dudek ``ml/model/tdeed/eval/legacy.py`` (SoccerNet protocol).
# ---------------------------------------------------------------------------

try:
    from SoccerNet.Evaluation.ActionSpotting import average_mAP
    from SoccerNet.Evaluation.utils import LoadJsonFromZip
except ImportError as _e:  # pragma: no cover - optional dependency
    average_mAP = None  # type: ignore[misc, assignment]
    LoadJsonFromZip = None  # type: ignore[misc, assignment]
    _SOCCERNET_IMPORT_ERROR = _e
else:
    _SOCCERNET_IMPORT_ERROR = None

FPS_SN = 25


def _ensure_soccernet() -> None:
    if _SOCCERNET_IMPORT_ERROR is not None:
        raise ImportError(
            "SoccerNet challenge mAP needs the ``soccernet`` package. "
            "Install with: pip install 'custom-ballspotting[challenge]' "
            "or pip install 'soccernet>=0.1.61'."
        ) from _SOCCERNET_IMPORT_ERROR


def compute_amAP(
    targets_numpy,
    detections_numpy,
    closests_numpy,
    framerate=25,
    metric="tight",
    event_team=False,
):
    _ensure_soccernet()

    if metric == "loose":
        deltas = np.arange(12) * 5 + 5
    elif metric == "tight":
        deltas = np.arange(5) * 1 + 1
    elif metric == "at1":
        deltas = np.array([1])
    elif metric == "at2":
        deltas = np.array([2])
    elif metric == "at3":
        deltas = np.array([3])
    elif metric == "at4":
        deltas = np.array([4])
    elif metric == "at5":
        deltas = np.array([5])
    else:
        raise ValueError(f"Unknown metric {metric!r}")

    if event_team:
        ntargets = np.zeros(targets_numpy[0].shape[1])
        for i in range(len(targets_numpy)):
            ntargets += targets_numpy[i].sum(axis=0)

    (
        mAP,
        mAP_per_class,
        mAP_visible,
        mAP_per_class_visible,
        mAP_unshown,
        mAP_per_class_unshown,
    ) = average_mAP(
        targets_numpy,
        detections_numpy,
        closests_numpy,
        framerate=framerate,
        deltas=deltas,
    )

    if event_team:
        mAP_per_class = mAP_per_class * ntargets
        mAP_per_class = np.array(
            [
                (
                    (mAP_per_class[i * 2] + mAP_per_class[(i * 2) + 1])
                    / (ntargets[i * 2] + ntargets[i * 2 + 1])
                    if (ntargets[i * 2] + ntargets[i * 2 + 1]) > 0
                    else np.nan
                )
                for i in range(len(mAP_per_class) // 2)
            ]
        )
        mAP = np.nanmean(mAP_per_class)

        mAP_per_class_visible = mAP_per_class_visible * ntargets
        mAP_per_class_visible = np.array(
            [
                (
                    (mAP_per_class_visible[i * 2] + mAP_per_class_visible[(i * 2) + 1])
                    / (ntargets[i * 2] + ntargets[i * 2 + 1])
                    if (ntargets[i * 2] + ntargets[i * 2 + 1]) > 0
                    else np.nan
                )
                for i in range(len(mAP_per_class_visible) // 2)
            ]
        )
        mAP_visible = np.nanmean(mAP_per_class_visible)

        mAP_per_class_unshown = mAP_per_class_unshown * ntargets
        mAP_per_class_unshown = np.array(
            [
                (
                    (mAP_per_class_unshown[i * 2] + mAP_per_class_unshown[(i * 2) + 1])
                    / (ntargets[i * 2] + ntargets[i * 2 + 1])
                    if (ntargets[i * 2] + ntargets[i * 2 + 1]) > 0
                    else np.nan
                )
                for i in range(len(mAP_per_class_unshown) // 2)
            ]
        )
        mAP_unshown = np.nanmean(mAP_per_class_unshown)

    return {
        "mAP": mAP,
        "mAP_per_class": mAP_per_class,
        "mAP_visible": mAP_visible,
        "mAP_per_class_visible": mAP_per_class_visible,
        "mAP_unshown": mAP_unshown,
        "mAP_per_class_unshown": mAP_per_class_unshown,
    }


def label2vector(
    labels,
    num_classes=17,
    framerate=2,
    version=2,
    EVENT_DICTIONARY=None,
    event_team=False,
):
    if EVENT_DICTIONARY is None:
        EVENT_DICTIONARY = {}
    vector_size = 120 * 60 * framerate

    label_half1 = np.zeros((vector_size, num_classes))

    for annotation in labels["annotations"]:
        time = annotation["gameTime"]
        event = annotation["label"]

        half = int(time[0])

        minutes = int(time[-5:-3])
        seconds = int(time[-2::])
        if "position" in annotation:
            frame = int(framerate * (int(annotation["position"]) / 1000))
        else:
            frame = framerate * (seconds + 60 * minutes)

        if not event_team:
            label = EVENT_DICTIONARY[event] - 1
        else:
            event = event + "-" + annotation["team"]
            label = EVENT_DICTIONARY[event] - 1

        value = 1
        if "visibility" in annotation.keys():
            if annotation["visibility"] == "not shown":
                value = -1

        if half == 1:
            frame = min(frame, vector_size - 1)
            label_half1[frame][label] = value

    return label_half1


def predictions2vector(
    predictions,
    num_classes=17,
    version=2,
    framerate=2,
    EVENT_DICTIONARY=None,
    event_team=False,
):
    if EVENT_DICTIONARY is None:
        EVENT_DICTIONARY = {}
    vector_size = 120 * 60 * framerate

    prediction_half1 = np.zeros((vector_size, num_classes)) - 1

    for annotation in predictions["predictions"]:
        time = int(annotation["position"])
        event = annotation["label"]

        frame = int(framerate * (time / 1000))

        if not event_team:
            label = EVENT_DICTIONARY[event] - 1
        else:
            event = event + "-" + annotation["team"]
            label = EVENT_DICTIONARY[event] - 1

        value = annotation["confidence"]

        frame = min(frame, vector_size - 1)
        prediction_half1[frame][label] = value

    return prediction_half1


def mAPevaluateTest(
    games,
    SoccerNet_path,
    Predictions_path,
    prediction_file="results_spotting.json",
    printed=False,
    event_team=True,
    metric="at1",
    event_dictionary: dict[str, int] | None = None,
):
    _ensure_soccernet()
    if event_dictionary is None:
        raise ValueError("event_dictionary is required")

    detections_numpy = []
    targets_numpy = []
    closests_numpy = []

    classes = event_dictionary

    for game in games:
        if zipfile.is_zipfile(SoccerNet_path):
            labels = LoadJsonFromZip(SoccerNet_path, os.path.join(game, "Labels-ball.json"))
        else:
            with open(os.path.join(SoccerNet_path, game, "Labels-ball.json"), encoding="utf-8") as f:
                labels = json.load(f)
        num_classes = max(classes.values())
        labels = label2vector(
            labels,
            num_classes=num_classes,
            version=2,
            EVENT_DICTIONARY=classes,
            framerate=FPS_SN,
            event_team=event_team,
        )

        if zipfile.is_zipfile(Predictions_path):
            predictions = LoadJsonFromZip(Predictions_path, os.path.join(game, prediction_file))
        else:
            with open(
                os.path.join(Predictions_path, game, prediction_file),
                encoding="utf-8",
            ) as f:
                predictions = json.load(f)
        predictions = predictions2vector(
            predictions,
            num_classes=num_classes,
            version=2,
            EVENT_DICTIONARY=classes,
            framerate=FPS_SN,
            event_team=event_team,
        )

        targets_numpy.append(labels)
        detections_numpy.append(predictions)

        closest_numpy = np.zeros(labels.shape) - 1
        for c in np.arange(labels.shape[-1]):
            indexes = np.where(labels[:, c] != 0)[0].tolist()
            if len(indexes) == 0:
                continue
            indexes.insert(0, -indexes[0])
            indexes.append(2 * closest_numpy.shape[0])
            for i in np.arange(len(indexes) - 2) + 1:
                start = max(0, (indexes[i - 1] + indexes[i]) // 2)
                stop = min(closest_numpy.shape[0], (indexes[i] + indexes[i + 1]) // 2)
                closest_numpy[start:stop, c] = labels[indexes[i], c]
        closests_numpy.append(closest_numpy)

    return compute_amAP(
        targets_numpy,
        detections_numpy,
        closests_numpy,
        framerate=FPS_SN,
        metric=metric,
        event_team=event_team,
    )


def team_bas_event_dictionary() -> dict[str, int]:
    """``EVENT_DICTIONARY`` keys ``{label}-{team}`` matching dudek / ``TdeedVideoClip``."""
    from custom_ballspotting.actions import Action, Team

    classes: dict[str, int] = {}
    for i, action in enumerate(Action):
        classes[f"{action.value}-{Team.LEFT.value}"] = i + 1
        classes[f"{action.value}-{Team.RIGHT.value}"] = i + 1 + len(Action)
    return classes


def run_mapevaluate_test_with_zip(
    *,
    games: list[str],
    soccernet_path: str,
    game_results_json: dict[str, dict],
    metric: str = "at1",
) -> dict:
    """Write ``results_spotting.json`` per game into a temp zip and run ``mAPevaluateTest``."""
    _ensure_soccernet()
    if not games:
        raise ValueError("games list is empty")

    tmp_root = tempfile.mkdtemp(prefix="custom-bs-sn-")
    zip_base = os.path.join(tempfile.gettempdir(), f"custom-bs-sol-{time.time()}")
    zip_path = f"{zip_base}.zip"
    try:
        for game_id, payload in game_results_json.items():
            game_dir = os.path.join(tmp_root, game_id)
            os.makedirs(game_dir, exist_ok=True)
            out_path = os.path.join(game_dir, "results_spotting.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

        shutil.make_archive(zip_base, "zip", tmp_root)

        return mAPevaluateTest(
            games,
            soccernet_path,
            zip_path,
            prediction_file="results_spotting.json",
            printed=False,
            event_team=True,
            metric=metric,
            event_dictionary=team_bas_event_dictionary(),
        )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        if os.path.isfile(zip_path):
            try:
                os.remove(zip_path)
            except OSError:
                pass
