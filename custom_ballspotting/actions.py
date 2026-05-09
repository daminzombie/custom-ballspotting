from enum import Enum
from typing import NamedTuple


class Team(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    NOT_APPLICABLE = "not applicable"

    def flip(self) -> "Team":
        if self == Team.LEFT:
            return Team.RIGHT
        if self == Team.RIGHT:
            return Team.LEFT
        return Team.NOT_APPLICABLE


def parse_team_string(raw: str | None) -> Team:
    """Parse a dataset ``team`` field into :class:`Team`.

    Accepts enum values (``left`` / ``right`` / ``not applicable``), common
    variants such as ``not_applicable`` or ``n/a``, and falls back to
    ``Team.LEFT`` for missing or unrecognised values (same behaviour as the
    previous try/except default).
    """
    if raw is None:
        return Team.LEFT
    s = str(raw).strip()
    if not s:
        return Team.LEFT
    lower = s.lower()
    if lower == "n/a":
        return Team.NOT_APPLICABLE
    normalized = lower.replace("_", " ")
    try:
        return Team(normalized)
    except ValueError:
        return Team.LEFT


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
    Action.BALL_OUT_OF_PLAY: ActionConfig(2.9, 0.5, 2.0),
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
    Action.BALL_OUT_OF_PLAY: 2.9,
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
# Total foreground classes = N actions × 2 teams; head output = 2*N + 1 (incl. background)
NUM_TEAM_ACTION_CLASSES: int = 2 * NUM_ACTION_CLASSES
TEAM_CLASS_INDEX: dict[Team, int] = {
    Team.LEFT: 0,
    Team.RIGHT: 1,
}
TEAM_IGNORE_INDEX = -100


def action_to_index(action: Action | str) -> int:
    """Return the action-head class index for an action (background is 0)."""
    return ACTION_CLASS_INDEX[Action(action).value] + 1


def index_to_action(index: int) -> Action | None:
    """Decode an action-head class index, or None for background / out of range."""
    if index <= 0 or index > NUM_ACTION_CLASSES:
        return None
    return list(Action)[index - 1]


def team_to_index(team: Team) -> int:
    """Return the team-head class index, or TEAM_IGNORE_INDEX when not supervised."""
    return TEAM_CLASS_INDEX.get(team, TEAM_IGNORE_INDEX)


def label_to_index(action: Action | str, team: Team = Team.LEFT) -> int:
    """Return the model class index for a (action, team) pair.

    Layout (background = 0):
      indices 1 .. N          → LEFT  team, actions[0..N-1]
      indices N+1 .. 2*N      → RIGHT team, actions[0..N-1]
    """
    action = Action(action)
    base = ACTION_CLASS_INDEX[action.value] + 1  # 1-based
    if team == Team.RIGHT:
        base += NUM_ACTION_CLASSES
    return base


def index_to_label(index: int) -> tuple[Action, Team] | None:
    """Decode a class index back to (Action, Team), or None for background (0)."""
    if index == 0:
        return None
    actions = list(Action)
    if index <= NUM_ACTION_CLASSES:
        return actions[index - 1], Team.LEFT
    right_index = index - NUM_ACTION_CLASSES
    if right_index <= NUM_ACTION_CLASSES:
        return actions[right_index - 1], Team.RIGHT
    return None
