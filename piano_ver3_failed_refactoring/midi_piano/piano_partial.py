from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PianoPartial:
    note_id: int
    partial_id: int
    partial_freq: float
    partial_tau: float
    partial_amp: float

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PianoPartial):
            return False
        return self.note_id == other.note_id and self.partial_id == other.partial_id
    
    def __hash__(self) -> int:
        return hash((self.note_id, self.partial_id))
