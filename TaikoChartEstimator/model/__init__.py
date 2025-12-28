"""
TaikoChartEstimator Model Package

Provides the MIL-based difficulty estimation model with:
- Instance encoder (Transformer-based)
- MIL aggregator with multi-branch attention
- Multi-head outputs (raw score, difficulty class, star rating)
"""

from .aggregator import GatedMILAggregator, MILAggregator
from .encoder import InstanceEncoder, TCNInstanceEncoder
from .heads import DifficultyClassifier, MonotonicCalibrator, RawScoreHead
from .losses import (
    CensoredRegressionLoss,
    CurriculumScheduler,
    TotalLoss,
    WithinSongRankingLoss,
)
from .model import ModelConfig, ModelOutput, TaikoChartEstimator

__all__ = [
    "InstanceEncoder",
    "TCNInstanceEncoder",
    "MILAggregator",
    "GatedMILAggregator",
    "RawScoreHead",
    "DifficultyClassifier",
    "MonotonicCalibrator",
    "TaikoChartEstimator",
    "ModelConfig",
    "ModelOutput",
    "WithinSongRankingLoss",
    "CensoredRegressionLoss",
    "TotalLoss",
    "CurriculumScheduler",
]
