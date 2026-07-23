"""
schema.py

Defines the metadata format used throughout the project. Metadata is
created by the labelling pipeline and used by Conditions D and E, as
well as by the benchmark.

The seven memory types are defined in one place to keep the labeller
and Condition D consistent. Maintaining separate lists could cause them
to drift apart and lead to silent routing errors.
"""

from dataclasses import dataclass, field

# The seven relational memory types defined in DESIGN §5.5.
# Update the benchmark property taxonomy before adding new types.
MEMORY_TYPES = (
    "fact",
    "preference",
    "promise",
    "conflict",
    "emotional_state",
    "relationship_update",
    "narrative_event",
)


@dataclass
class MemoryMetadata:
    """Metadata assigned when a memory is written, without access to probes.

    Salience and importance are assigned by the LLM on a scale from 0 to 1.
    Salience affects the decay rate used by Condition E. Importance is stored
    and reported separately but is not currently used by Condition E.

    presentation_log stores the timestamp of the initial write and each later
    retrieval into context, following the decision in §5.3. These timestamps
    are used to calculate B_i.
    """

    memory_id: str
    speaker: str
    session_id: int
    turn_index: int
    timestamp: float
    type: str
    salience: float
    importance: float
    presentation_log: list[float] = field(default_factory=list)

    def __post_init__(self):
        if self.type not in MEMORY_TYPES:
            raise ValueError(
                f"Unknown memory type {self.type!r}. "
                f"Must be one of {MEMORY_TYPES}."
            )
