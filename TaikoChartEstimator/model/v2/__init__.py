"""
TaikoChartEstimator Model Package - Version 2

Provides the MIL-based difficulty estimation model with:
- Instance encoder (Transformer-based)
- MIL aggregator with multi-branch attention
- Multi-head outputs (raw score, difficulty class, star rating)
- Interpretability module for feature attribution

V2 Changes:
- Updated features: local_density replaces gogo
- New interpretability module
"""

from .aggregator import GatedMILAggregator, MILAggregator
from .encoder import InstanceEncoder, TCNInstanceEncoder
from .heads import DifficultyClassifier, MonotonicCalibrator, RawScoreHead
from .interpretability import (
    ChartInterpreter,
    DiscretePatternAttributor,
    FeatureAttributor,
    FeatureContribution,
    InterpretabilityReport,
)
from .losses import (
    CensoredRegressionLoss,
    CurriculumScheduler,
    TotalLoss,
    WithinSongRankingLoss,
)
from .model import ModelConfig, ModelOutput, TaikoChartEstimator, create_model

__all__ = [
    # Encoder
    "InstanceEncoder",
    "TCNInstanceEncoder",
    # Aggregator
    "MILAggregator",
    "GatedMILAggregator",
    # Heads
    "RawScoreHead",
    "DifficultyClassifier",
    "MonotonicCalibrator",
    # Model
    "TaikoChartEstimator",
    "ModelConfig",
    "ModelOutput",
    "create_model",
    # Losses
    "WithinSongRankingLoss",
    "CensoredRegressionLoss",
    "TotalLoss",
    "CurriculumScheduler",
    # Interpretability (new in v2)
    "FeatureAttributor",
    "DiscretePatternAttributor",
    "ChartInterpreter",
    "FeatureContribution",
    "InterpretabilityReport",
]
