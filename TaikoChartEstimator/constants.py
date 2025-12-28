"""
Centralized Constants for TaikoChartEstimator

Consolidates all difficulty mappings, note types, and star ranges
to avoid duplication across modules.
"""

from typing import Dict, Tuple

# =============================================================================
# Note Types
# =============================================================================

NOTE_TYPES = [
    "Don",  # 0
    "Ka",  # 1
    "DonBig",  # 2
    "KaBig",  # 3
    "Roll",  # 4
    "RollBig",  # 5
    "Balloon",  # 6
    "BalloonAlt",  # 7
    "EndOf",  # 8
]

NOTE_TYPE_TO_ID: Dict[str, int] = {
    note_type: i for i, note_type in enumerate(NOTE_TYPES)
}
NUM_NOTE_TYPES = len(NOTE_TYPES)
PAD_TOKEN_ID = NUM_NOTE_TYPES  # 9 for padding

# =============================================================================
# Difficulty Classes
# =============================================================================

# Original 5 classes
DIFFICULTY_CLASSES = ["easy", "normal", "hard", "oni", "ura"]

# Merged classes (ura -> oni)
DIFFICULTY_CLASSES_MERGED = ["easy", "normal", "hard", "oni_ura"]
NUM_DIFFICULTY_CLASSES = len(DIFFICULTY_CLASSES)
NUM_DIFFICULTY_CLASSES_MERGED = len(DIFFICULTY_CLASSES_MERGED)

# Difficulty name -> class ID mapping (handles both cases)
DIFFICULTY_TO_ID: Dict[str, int] = {}
for i, d in enumerate(DIFFICULTY_CLASSES):
    DIFFICULTY_TO_ID[d] = i
    DIFFICULTY_TO_ID[d.capitalize()] = i

# Difficulty ordering for ranking comparisons
DIFFICULTY_ORDER: Dict[str, int] = {
    "easy": 0,
    "Easy": 0,
    "normal": 1,
    "Normal": 1,
    "hard": 2,
    "Hard": 2,
    "oni": 3,
    "Oni": 3,
    "ura": 4,
    "Ura": 4,
}

# =============================================================================
# Star Ranges per Difficulty
# =============================================================================

# Star ranges by difficulty index
STAR_RANGES_BY_ID: Dict[int, Tuple[int, int]] = {
    0: (1, 5),  # easy
    1: (1, 7),  # normal
    2: (1, 8),  # hard
    3: (1, 10),  # oni
    4: (1, 10),  # ura
}

# Star ranges by difficulty name (includes capitalized versions)
STAR_RANGES_BY_NAME: Dict[str, Tuple[int, int]] = {
    "easy": (1, 5),
    "Easy": (1, 5),
    "normal": (1, 7),
    "Normal": (1, 7),
    "hard": (1, 8),
    "Hard": (1, 8),
    "oni": (1, 10),
    "Oni": (1, 10),
    "ura": (1, 10),
    "Ura": (1, 10),
}

# =============================================================================
# Helper Functions
# =============================================================================


def merge_difficulty_class(class_id: int) -> int:
    """Merge ura (4) into oni (3) for classification."""
    return 3 if class_id == 4 else class_id


def get_difficulty_name(class_id: int, merged: bool = False) -> str:
    """Get difficulty name from class ID."""
    if merged:
        return DIFFICULTY_CLASSES_MERGED[min(class_id, 3)]
    return DIFFICULTY_CLASSES[class_id]
