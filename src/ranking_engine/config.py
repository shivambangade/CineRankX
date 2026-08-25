"""Tunable weights for the six ranking objectives.

Weights are read from eval_config.yaml (`ranking_objectives.weights`) and are
never hardcoded in the ranking logic — the ranker only ever asks this object
for a weight by name, so retuning is a config edit, not a code edit.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# The six objectives, in the canonical order used for reporting/breakdowns.
OBJECTIVES = (
    "relevance",
    "diversity",
    "novelty",
    "coverage",
    "popularity_quality",
    "predicted_rating",
)

_DEFAULT_CONFIG_PATH = Path("eval_config.yaml")


_DEFAULT_GENRE_GATE_FLOOR = 0.25


def load_genre_gate_floor(path: Path | str = _DEFAULT_CONFIG_PATH) -> float:
    """Read ranking_objectives.genre_relevance_gate.floor from eval_config.yaml.

    Kept in config for the same reason the weights are: the gate's strength is
    a tuning decision, not a fact about the algorithm. Falls back to the module
    default when the block is absent, so an older config still loads.
    """
    try:
        with open(path) as f:
            config = yaml.safe_load(f)
        floor = config["ranking_objectives"]["genre_relevance_gate"]["floor"]
    except (OSError, KeyError, TypeError):
        return _DEFAULT_GENRE_GATE_FLOOR

    floor = float(floor)
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"genre_relevance_gate.floor must be in [0, 1], got {floor}")
    return floor


@dataclass
class RankingWeights:
    """Weights for the six objectives, normalized to sum to 1.0.

    Normalizing means a user retuning weights in eval_config.yaml only has to
    express *relative* importance ("double the diversity weight") without also
    rebalancing the other five to keep the total at 1.0. Final scores stay on a
    comparable [0, 1] scale across different weight settings, so two runs'
    scores can be compared directly.
    """

    weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        missing = [o for o in OBJECTIVES if o not in self.weights]
        if missing:
            raise ValueError(f"Missing weights for objectives: {missing}")
        unknown = [k for k in self.weights if k not in OBJECTIVES]
        if unknown:
            raise ValueError(f"Unknown objectives in weights: {unknown}")
        for name, value in self.weights.items():
            if value < 0:
                raise ValueError(f"Weight for '{name}' is negative: {value}")

        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("Ranking weights sum to zero; at least one must be positive.")
        self.weights = {name: float(value) / total for name, value in self.weights.items()}

    def __getitem__(self, objective: str) -> float:
        return self.weights[objective]

    def items(self):
        """Iterate objectives in canonical OBJECTIVES order, not dict order."""
        return [(name, self.weights[name]) for name in OBJECTIVES]

    def with_overrides(self, **overrides: float) -> "RankingWeights":
        """Return a new RankingWeights with some weights replaced.

        Used for weight-sensitivity experiments (e.g. "what changes if
        diversity goes 0.15 -> 0.60?") without mutating shared config state.
        """
        merged = dict(self.weights)
        merged.update(overrides)
        return RankingWeights(merged)

    @classmethod
    def from_config(cls, path: Path | str = _DEFAULT_CONFIG_PATH) -> "RankingWeights":
        """Load weights from eval_config.yaml's ranking_objectives.weights block."""
        with open(path) as f:
            config = yaml.safe_load(f)
        try:
            weights = config["ranking_objectives"]["weights"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{path} has no ranking_objectives.weights block") from exc
        return cls(dict(weights))
