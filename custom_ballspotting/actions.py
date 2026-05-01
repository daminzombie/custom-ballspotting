from enum import Enum
from typing import NamedTuple


class Team(str, Enum):
    LEFT = "left"
    RIGHT = "right"

    def flip(self) -> "Team":
        return Team.RIGHT if self == Team.LEFT else Team.LEFT


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

ACTION_CLASS_INDEX: dict[str, int] = {
    action.value: idx for idx, action in enumerate(Action)
}
NUM_ACTION_CLASSES: int = len(ACTION_CLASS_INDEX)
# Total foreground classes = N actions × 2 teams; head output = 2*N + 1 (incl. background)
NUM_TEAM_ACTION_CLASSES: int = 2 * NUM_ACTION_CLASSES


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
