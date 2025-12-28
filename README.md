# TaikoChartEstimator

MIL-based Taiko chart difficulty estimator that predicts difficulty class and star rating from note charts. The model uses Transformer instance encoders, multi-branch attention MIL pooling, monotonic calibration, and multi-task losses (classification, censored regression, within-song ranking) with optional curriculum scheduling.

- Multi-instance learning over beat-aligned windows with stochastic top-k masking to avoid attention collapse
- Multi-task objectives with censored regression for boundary stars and ranking loss for within-song monotonicity
- Transformer instance encoder, multi-branch or gated attention aggregator, monotonic spline/MLP calibrator
- TensorBoard logging, curriculum scheduling, and HuggingFace checkpoints

Our goals are simple:

1. **Star-Level Granularity**: Move beyond traditional 1-10 integer star ratings to provide continuous sub-star difficulty scores (e.g., 9.3 vs 9.7), offering a more precise difficulty metric.
2. **High-Difficulty Separation**: Address "10-star inflation" by accurately tiering top-level charts, distinguishing between entry-level 10-star songs and those that significantly exceed the nominal boundary.
3. **Sectional Interpretability**: Provide section-by-section difficulty analysis to identify which specific segments contribute most to the overall rating, giving clear insights into the chart's complexity.

## Installation

Install the project dependencies:

```bash
uv sync
```

The default dataset is hosted on HuggingFace (`JacobLinCool/taiko-1000-parsed`), which is gated, so request access if needed.

## Training

Launch training with the provided CLI:

```bash
uv run -m TaikoChartEstimator.train --batch-size 16 --epochs 100 --use-curriculum
```

Key arguments:

- `--encoder-type`: `transformer` (default) or `tcn`
- `--n-branches`: attention branches for MIL pooling (set `--encoder-type tcn` to switch encoders)
- `--lambda-*`: weights for classification, star regression, and ranking losses
- `--use-curriculum`: anneal loss weights over training steps
- `--overfit-batch`: small-batch debug mode

Outputs:

- `outputs/<timestamp>/args.json`: run configuration
- `outputs/<timestamp>/pretrained/{best,final}/`: `config.json`, `model.safetensors`, `README.md`
- `outputs/<timestamp>/checkpoint_epoch*.pt`: traditional checkpoints with optimizer state
- `runs/<timestamp>/`: TensorBoard logs

## Evaluation

Evaluate a saved checkpoint (HuggingFace directory or `.pt` file):

```bash
uv run -m TaikoChartEstimator.eval.evaluator --checkpoint pretrained/model
```

Artifacts:

- `eval_results/metrics.json`: metrics for difficulty, star regression, monotonicity, decompression, MIL health
- `eval_results/report.md`: human-readable report

## Inference Example

```python
import torch
from TaikoChartEstimator.data import TaikoChartDataset, collate_chart_bags
from TaikoChartEstimator.model import TaikoChartEstimator

# Load pretrained model (directory from training outputs)
model = TaikoChartEstimator.from_pretrained("pretrained/model").eval()

# Prepare a single chart
dataset = TaikoChartDataset(split="test")
batch = collate_chart_bags([dataset[0]])

with torch.no_grad():
    result = model.predict(
        batch["instances"],
        batch["instance_masks"],
        batch["instance_counts"],
    )

print(result["difficulty_class"], result["display_star"].tolist())
```

## Architecture

```
Input: Chart Notes (segments → notes)
         ↓
   Event Tokenizer
         ↓
   Beat-Aligned Windows
         ↓
   Instance Encoder (Transformer)
         ↓
   MIL Aggregator (3-way pooling + multi-branch attention)
         ↓
   ┌─────────┼─────────┐
   ↓         ↓         ↓
Head A    Head B    Head C
Raw Score Difficulty Monotonic
 (s ∈ ℝ)   Class     Calibrator
                        ↓
                    Star Rating
```

## Key Concepts

### 1. Unbounded Raw Score

The model outputs a raw difficulty score `s ∈ ℝ` that is **not bounded** to 1-10. This allows:

- Fine-grained ranking of 10-star charts (which vary from ~10.0 to potentially 12+)
- Natural handling of both very easy and very hard charts

### 2. Within-Song Monotonicity

For the same song, we enforce: `s(easy) < s(normal) < s(hard) < s(oni) < s(ura)`

This is done via a hinge loss on within-song pairs during training.

### 3. Censored Regression

Star labels at boundaries (1 and 10) are treated as **censored** observations:

- `star = 10` means true difficulty is **≥ 10** (right-censored)
- `star = 1` means true difficulty is **≤ 1** (left-censored)

The loss only penalizes predictions that violate these bounds.

## Outputs

The model produces:

| Output | Description |
|--------|-------------|
| `raw_score` | Unbounded continuous score `s ∈ ℝ` |
| `difficulty_class` | Predicted class (0-4: easy/normal/hard/oni/ura) |
| `raw_star` | Calibrated star (can be < 1 or > 10) |
| `display_star` | Star clipped to valid range per difficulty |

## Evaluation Metrics

### Difficulty Classification

- Macro-F1, Balanced Accuracy
- ±1 Tolerance Accuracy (ordinal-aware)

### Star Regression

- MAE, RMSE, Spearman ρ (on uncensored samples)
- Right/Left censor violation rates

### Monotonicity

- Within-song violation rate
- Mean Kendall τ per song

### Decompression

- Std of 10-star predictions (higher = better separation)
- P90-P50, P99-P90 gaps in 10-star predictions

### MIL Health

- Attention entropy (higher = more distributed)
- Effective instance count
- Top-5% attention mass (lower = less collapse)
