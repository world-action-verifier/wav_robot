from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TrainerState:
    step: int
    env_step: int = 0
    round_id: int = -1
    history: Dict[str, float] = field(default_factory=dict)
