"""
TaikoChartEstimator Data Pipeline - Version 2

Provides event tokenization, dataset loading, and audio processing for
MIL-based Taiko chart difficulty estimation.

V2 Changes:
- Removed gogo feature
- Added local_density and measure_index
- Measure-based windowing for structural alignment
"""

from .audio import AudioProcessor
from .dataset import (
    ChartBag,
    SongGroup,
    TaikoChartDataset,
    WithinSongBatchSampler,
    WithinSongPairSampler,
    collate_chart_bags,
)
from .tokenizer import (
    DIFFICULTY_ORDER,
    NOTE_TYPE_TO_ID,
    NOTE_TYPES,
    EventToken,
    EventTokenizer,
)

__all__ = [
    "EventToken",
    "EventTokenizer",
    "NOTE_TYPES",
    "NOTE_TYPE_TO_ID",
    "DIFFICULTY_ORDER",
    "TaikoChartDataset",
    "ChartBag",
    "SongGroup",
    "WithinSongBatchSampler",
    "WithinSongPairSampler",
    "collate_chart_bags",
    "AudioProcessor",
]
