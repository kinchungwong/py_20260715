from dataclasses import dataclass

@dataclass(frozen=True)
class PianoNote:
    note_id: int
    velocity: int
