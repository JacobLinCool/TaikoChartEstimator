"""
Taiko Chart Dataset for MIL-based Difficulty Estimation - Version 2

Loads data from JacobLinCool/taiko-1000-parsed and provides:
- ChartBag: A single chart with its instances (windows)
- SongGroup: All difficulty charts for a single song (for ranking loss)
- Within-song pair sampling for training

V2 Changes:
- Uses v2 tokenizer with measure-based windowing
- local_density as explicit feature
- Measure indices for structural alignment
"""

from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset, Sampler

# Import from centralized constants
from ...constants import (
    DIFFICULTY_CLASSES,
    DIFFICULTY_ORDER,
    NOTE_TYPE_TO_ID,
)
from ...constants import (
    DIFFICULTY_TO_ID as DIFFICULTY_TO_CLASS_ID,
)
from ...constants import (
    STAR_RANGES_BY_NAME as STAR_RANGES,
)
from .audio import AudioProcessor
from .tokenizer import EventToken, EventTokenizer


@dataclass
class ChartBag:
    """
    A single chart represented as a bag of instances for MIL.

    Attributes:
        song_id: Unique identifier for the song
        difficulty: Difficulty level (easy/normal/hard/oni/ura)
        difficulty_class_id: Integer class ID for difficulty
        star: Star rating from label (1-10)
        is_right_censored: True if star == max for difficulty (label is lower bound)
        is_left_censored: True if star == min for difficulty (label is upper bound)
        instances: List of token tensors for each window
        instance_masks: Attention masks for each instance
        instance_measures: (start_measure, end_measure) for each instance
        audio_mel: Optional full mel spectrogram for the song
    """

    song_id: str
    difficulty: str
    difficulty_class_id: int
    star: int
    is_right_censored: bool
    is_left_censored: bool
    instances: list[torch.Tensor] = field(default_factory=list)
    instance_masks: list[torch.Tensor] = field(default_factory=list)
    instance_measures: list[tuple[int, int]] = field(default_factory=list)
    audio_mel: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self.instances)


@dataclass
class SongGroup:
    """
    All charts for a single song, for within-song ranking loss.

    Charts are ordered by difficulty (easy < normal < hard < oni < ura).
    """

    song_id: str
    charts: list[ChartBag] = field(default_factory=list)

    def get_ranking_pairs(self) -> list[tuple[ChartBag, ChartBag]]:
        """
        Get all adjacent difficulty pairs for ranking loss.

        Returns:
            List of (easier_chart, harder_chart) tuples
        """
        # Sort by difficulty order
        sorted_charts = sorted(
            self.charts, key=lambda c: DIFFICULTY_ORDER.get(c.difficulty, 0)
        )

        pairs = []
        for i in range(len(sorted_charts) - 1):
            pairs.append((sorted_charts[i], sorted_charts[i + 1]))

        return pairs


class TaikoChartDataset(Dataset):
    """
    PyTorch Dataset for Taiko chart difficulty estimation - V2.

    Loads from HuggingFace dataset and provides ChartBag instances.
    Uses measure-based windowing for structural alignment.
    """

    def __init__(
        self,
        split: str = "train",
        dataset_name: str = "JacobLinCool/taiko-1000-parsed",
        window_measures: list[int] = [2, 4],
        hop_measures: int = 2,
        max_instances_per_chart: int = 64,
        max_tokens_per_instance: int = 128,
        include_audio: bool = False,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize dataset.

        Args:
            split: Dataset split ("train" or "test")
            dataset_name: HuggingFace dataset name
            window_measures: Window sizes in measures for multi-scale
            hop_measures: Hop size in measures
            max_instances_per_chart: Maximum instances to keep per chart
            max_tokens_per_instance: Maximum tokens per instance
            include_audio: Whether to load and process audio
            cache_dir: Cache directory for dataset
        """
        self.split = split
        self.window_measures = window_measures
        self.hop_measures = hop_measures
        self.max_instances_per_chart = max_instances_per_chart
        self.max_tokens_per_instance = max_tokens_per_instance
        self.include_audio = include_audio

        # Initialize processors (v2 tokenizer)
        self.tokenizer = EventTokenizer()
        self.audio_processor = AudioProcessor() if include_audio else None

        # Load HuggingFace dataset
        self.hf_dataset = load_dataset(
            dataset_name,
            split=split,
            cache_dir=cache_dir,
        )

        # Build index of all charts (song_idx, difficulty)
        self._build_chart_index()

    def _build_chart_index(self):
        """Build an index of all available charts across songs."""
        self.chart_index: list[tuple[int, str]] = []  # (song_idx, difficulty)
        self.song_groups: dict[int, SongGroup] = {}  # song_idx -> SongGroup

        difficulties = ["easy", "normal", "hard", "oni", "ura"]

        for song_idx in range(len(self.hf_dataset)):
            song = self.hf_dataset[song_idx]
            song_id = f"song_{song_idx}"

            # Check which difficulties are available
            available_diffs = []
            for diff in difficulties:
                if diff in song and song[diff] is not None:
                    diff_data = song[diff]
                    # Check if it has valid segments
                    if diff_data.get("segments") and len(diff_data["segments"]) > 0:
                        self.chart_index.append((song_idx, diff))
                        available_diffs.append(diff)

            # Create song group
            if available_diffs:
                self.song_groups[song_idx] = SongGroup(song_id=song_id)

    def __len__(self) -> int:
        return len(self.chart_index)

    def _process_chart(
        self,
        song_data: dict,
        song_idx: int,
        difficulty: str,
    ) -> ChartBag:
        """Process a single chart into a ChartBag."""
        song_id = f"song_{song_idx}"
        diff_data = song_data[difficulty]

        # Get star rating and censoring info
        star = diff_data.get("level", 5)  # Default to 5 if missing
        min_star, max_star = STAR_RANGES.get(difficulty, (1, 10))
        is_right_censored = star >= max_star
        is_left_censored = star <= min_star

        # Get difficulty class ID
        diff_class_id = DIFFICULTY_TO_CLASS_ID.get(difficulty, 0)

        # Tokenize chart notes (v2 with measure indices and density)
        segments = diff_data.get("segments", [])
        tokens = self.tokenizer.tokenize_chart(segments)

        # Create multi-scale windows (measure-based in v2)
        all_instances = []
        all_masks = []
        all_measures = []

        for window_size in self.window_measures:
            windows = self.tokenizer.create_windows(
                tokens,
                window_measures=window_size,
                hop_measures=self.hop_measures,
            )

            for window_tokens in windows:
                if not window_tokens:
                    continue

                # Convert to tensor
                tensor, mask = self.tokenizer.tokens_to_tensor(
                    window_tokens,
                    max_length=self.max_tokens_per_instance,
                )

                # Pad to max length
                tensor, mask = self.tokenizer.pad_sequence(
                    tensor, mask, self.max_tokens_per_instance
                )

                # Record measure range
                start_measure = window_tokens[0].measure_index
                end_measure = window_tokens[-1].measure_index

                all_instances.append(tensor)
                all_masks.append(mask)
                all_measures.append((start_measure, end_measure))

        # Limit number of instances
        if len(all_instances) > self.max_instances_per_chart:
            # Sample uniformly
            indices = np.linspace(
                0, len(all_instances) - 1, self.max_instances_per_chart, dtype=int
            )
            all_instances = [all_instances[i] for i in indices]
            all_masks = [all_masks[i] for i in indices]
            all_measures = [all_measures[i] for i in indices]

        # Process audio if requested
        audio_mel = None
        if self.include_audio and "audio" in song_data:
            audio_data = song_data["audio"]
            if audio_data is not None:
                waveform = audio_data.get("array")
                sr = audio_data.get("sampling_rate", 22050)
                if waveform is not None:
                    audio_mel = self.audio_processor.process_audio(waveform, sr)

        return ChartBag(
            song_id=song_id,
            difficulty=difficulty,
            difficulty_class_id=diff_class_id,
            star=star,
            is_right_censored=is_right_censored,
            is_left_censored=is_left_censored,
            instances=all_instances,
            instance_masks=all_masks,
            instance_measures=all_measures,
            audio_mel=audio_mel,
        )

    def __getitem__(self, idx: int) -> ChartBag:
        song_idx, difficulty = self.chart_index[idx]
        song_data = self.hf_dataset[song_idx]
        return self._process_chart(song_data, song_idx, difficulty)

    def get_song_group(self, song_idx: int) -> SongGroup:
        """
        Get all charts for a song as a SongGroup.

        Args:
            song_idx: Index in the HuggingFace dataset

        Returns:
            SongGroup with all available difficulty charts
        """
        song_data = self.hf_dataset[song_idx]
        song_id = f"song_{song_idx}"
        group = SongGroup(song_id=song_id)

        for diff in DIFFICULTY_CLASSES:
            if diff in song_data and song_data[diff] is not None:
                diff_data = song_data[diff]
                if diff_data.get("segments") and len(diff_data["segments"]) > 0:
                    chart = self._process_chart(song_data, song_idx, diff)
                    group.charts.append(chart)

        return group

    def get_all_song_indices(self) -> list[int]:
        """Get list of unique song indices in the dataset."""
        return list(self.song_groups.keys())


class WithinSongBatchSampler(Sampler[list[int]]):
    """
    BatchSampler that ensures each batch contains complete song groups.

    This prevents ranking loss from being broken by batch boundaries that
    split charts from the same song into different batches.
    """

    def __init__(
        self,
        dataset: TaikoChartDataset,
        min_batch_size: int = 16,
        shuffle: bool = True,
        seed: int = 2025,
    ):
        """
        Initialize batch sampler.

        Args:
            dataset: The TaikoChartDataset
            min_batch_size: Minimum number of charts per batch
            shuffle: Whether to shuffle songs each epoch
            seed: Random seed
        """
        self.dataset = dataset
        self.min_batch_size = min_batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        # Build song to chart indices mapping
        self.song_to_charts: dict[int, list[int]] = {}
        for chart_idx, (song_idx, diff) in enumerate(dataset.chart_index):
            if song_idx not in self.song_to_charts:
                self.song_to_charts[song_idx] = []
            self.song_to_charts[song_idx].append(chart_idx)

        self.song_indices = list(self.song_to_charts.keys())

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches of chart indices, with complete song groups."""
        song_order = self.song_indices.copy()
        if self.shuffle:
            self.rng.shuffle(song_order)

        current_batch: list[int] = []

        for song_idx in song_order:
            chart_indices = self.song_to_charts[song_idx].copy()
            if self.shuffle:
                self.rng.shuffle(chart_indices)

            # Add all charts from this song to current batch
            current_batch.extend(chart_indices)

            # Yield batch when we have enough samples
            if len(current_batch) >= self.min_batch_size:
                yield current_batch
                current_batch = []

        # Yield remaining samples
        if current_batch:
            yield current_batch

    def __len__(self) -> int:
        # Approximate number of batches
        total_charts = len(self.dataset)
        return max(1, total_charts // self.min_batch_size)


# Keep old class name as alias for backward compatibility
WithinSongPairSampler = WithinSongBatchSampler


def collate_chart_bags(bags: list[ChartBag], max_seq_len: int = 128) -> dict:
    """
    Collate function for ChartBag instances.

    Args:
        bags: List of ChartBag instances to collate
        max_seq_len: Fallback sequence length for padding empty instances

    Returns a dictionary suitable for model input.
    """
    # Stack instances: need to handle variable numbers
    max_instances = max(len(b.instances) for b in bags)

    # Infer sequence length from first non-empty bag, or use parameter
    inferred_seq_len = max_seq_len
    for bag in bags:
        if bag.instances:
            inferred_seq_len = bag.instances[0].shape[0]
            break

    # Pad instances to same count
    batch_instances = []
    batch_masks = []
    instance_counts = []

    for bag in bags:
        instances = bag.instances
        masks = bag.instance_masks

        # Pad to max_instances
        n_pad = max_instances - len(instances)
        if n_pad > 0:
            # Infer shape from existing instances or use fallback
            pad_shape = instances[0].shape if instances else (inferred_seq_len, 6)
            instances = instances + [torch.zeros(pad_shape) for _ in range(n_pad)]
            masks = masks + [torch.zeros(pad_shape[0]) for _ in range(n_pad)]

        batch_instances.append(torch.stack(instances))
        batch_masks.append(torch.stack(masks))
        instance_counts.append(len(bag.instances))

    return {
        "instances": torch.stack(batch_instances),  # [B, N, L, 6]
        "instance_masks": torch.stack(batch_masks),  # [B, N, L]
        "instance_counts": torch.tensor(instance_counts),  # [B]
        "difficulty_class": torch.tensor([b.difficulty_class_id for b in bags]),  # [B]
        "star": torch.tensor([b.star for b in bags], dtype=torch.float32),  # [B]
        "is_right_censored": torch.tensor([b.is_right_censored for b in bags]),  # [B]
        "is_left_censored": torch.tensor([b.is_left_censored for b in bags]),  # [B]
        "song_ids": [b.song_id for b in bags],  # List[str]
        "difficulties": [b.difficulty for b in bags],  # List[str]
    }
