from enum import Enum
from typing import NamedTuple


class Action(str, Enum):
    PASS = "pass"
    PASS_RECEIVED = "pass_received"
    FREE_KICK = "free_kick"
    GOAL_KICK = "goal_kick"
    CORNER = "corner"
    THROW_IN = "throw_in"
    RECOVERY = "recovery"
    TACKLE = "tackle"
    INTERCEPTION = "interception"
    BALL_OUT_OF_PLAY_CLEAR = "ball_out_of_play_clear"
    BALL_OUT_OF_PLAY_DISTANT = "ball_out_of_play_distant"
    CLEARANCE = "clearance"
    TAKE_ON = "take_on"
    SUBSTITUTION = "substitution"
    BLOCK = "block"
    AERIAL_DUEL = "aerial_duel"
    SHOT = "shot"
    SAVE = "save"
    FOUL = "foul"
    GOAL = "goal"


class ActionConfig(NamedTuple):
    #: Inference / post-processing scale (not used for training CE; see
    #: :data:`TRAINING_CE_RELATIVE_WEIGHTS`).
    weight: float
    min_score: float
    tolerance_seconds: float


ACTION_CONFIGS: dict[Action, ActionConfig] = {
    Action.PASS: ActionConfig(1.0, 0.0, 1.0),
    Action.PASS_RECEIVED: ActionConfig(1.4, 0.0, 1.0),
    Action.FREE_KICK: ActionConfig(1.41, 0.0, 1.5),
    Action.GOAL_KICK: ActionConfig(1.42, 0.0, 1.5),
    Action.CORNER: ActionConfig(1.43, 0.0, 1.5),
    Action.THROW_IN: ActionConfig(1.44, 0.0, 1.5),
    Action.RECOVERY: ActionConfig(1.5, 0.0, 1.5),
    Action.TACKLE: ActionConfig(2.5, 0.1, 1.5),
    Action.INTERCEPTION: ActionConfig(2.8, 0.5, 2.0),
    Action.BALL_OUT_OF_PLAY_CLEAR: ActionConfig(2.9, 0.5, 2.0),
    Action.BALL_OUT_OF_PLAY_DISTANT: ActionConfig(2.9, 0.5, 2.0),
    Action.CLEARANCE: ActionConfig(3.1, 0.5, 2.0),
    Action.TAKE_ON: ActionConfig(3.2, 0.5, 2.0),
    Action.SUBSTITUTION: ActionConfig(4.2, 0.5, 2.0),
    Action.BLOCK: ActionConfig(4.2, 0.5, 2.0),
    Action.AERIAL_DUEL: ActionConfig(4.3, 0.5, 2.0),
    Action.SHOT: ActionConfig(4.7, 0.5, 2.0),
    Action.SAVE: ActionConfig(7.3, 0.5, 2.0),
    Action.FOUL: ActionConfig(7.7, 0.5, 2.5),
    Action.GOAL: ActionConfig(10.9, 0.5, 3.0),
}

# Cross-entropy only: relative importance among actions (most frequent ≈ 1.0).
# Final CE weight for each foreground class is ``ce_foreground_scale * value``;
# background stays 1.0. Independent of :attr:`ActionConfig.weight`.
TRAINING_CE_RELATIVE_WEIGHTS: dict[Action, float] = {
    Action.PASS: 1.0,
    Action.PASS_RECEIVED: 1.4,
    Action.FREE_KICK: 1.41,
    Action.GOAL_KICK: 1.42,
    Action.CORNER: 1.43,
    Action.THROW_IN: 1.44,
    Action.RECOVERY: 1.5,
    Action.TACKLE: 2.5,
    Action.INTERCEPTION: 2.8,
    Action.BALL_OUT_OF_PLAY_CLEAR: 2.9,
    Action.BALL_OUT_OF_PLAY_DISTANT: 2.9,
    Action.CLEARANCE: 3.1,
    Action.TAKE_ON: 3.2,
    Action.SUBSTITUTION: 4.2,
    Action.BLOCK: 4.2,
    Action.AERIAL_DUEL: 4.3,
    Action.SHOT: 4.7,
    Action.SAVE: 7.3,
    Action.FOUL: 7.7,
    Action.GOAL: 10.9,
}

if len(TRAINING_CE_RELATIVE_WEIGHTS) != len(Action):
    raise RuntimeError(
        "TRAINING_CE_RELATIVE_WEIGHTS must define exactly one entry per Action enum member"
    )

ACTION_CLASS_INDEX: dict[str, int] = {
    action.value: idx for idx, action in enumerate(Action)
}
NUM_ACTION_CLASSES: int = len(ACTION_CLASS_INDEX)


def action_to_index(action: Action | str) -> int:
    """Return the action-head class index for an action (background is 0)."""
    return ACTION_CLASS_INDEX[Action(action).value] + 1


def index_to_action(index: int) -> Action | None:
    """Decode an action-head class index, or None for background / out of range."""
    if index <= 0 or index > NUM_ACTION_CLASSES:
        return None
    return list(Action)[index - 1]


def foreground_column_for_action(action: Action | str) -> int:
    """Column index in mAP / dense score grids without background (0 .. N-1)."""
    return ACTION_CLASS_INDEX[Action(action).value]
