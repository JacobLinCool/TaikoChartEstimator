"""
Event Tokenizer for Taiko Chart Notes - Version 2

Converts raw chart note data into event tokens suitable for sequence modeling.
Key changes from v1:
- Removed gogo feature
- Added explicit local_density feature
- Added measure_index for exact measure-based windowing
- Windowing based on measure indices instead of estimated time
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

# Import from centralized constants
from ...constants import (
    DIFFICULTY_ORDER,
    NOTE_TYPE_TO_ID,
    NOTE_TYPES,
    PAD_TOKEN_ID,
)
from ...constants import (
    STAR_RANGES_BY_NAME as STAR_RANGES,
)


@dataclass
class EventToken:
    """A single event token representing a note or event in the chart.

    V2 changes:
    - Removed gogo (loosely defined contextual signal)
    - Added measure_index for exact measure-based windowing
    - Added local_density for explicit density feature
    """

    timestamp: float  # Absolute time in seconds
    beat_position: float  # Position within the measure (0-1)
    note_type: int  # ID from NOTE_TYPE_TO_ID
    duration: float  # Duration for rolls/balloons (0 for regular notes)
    bpm: float  # Current BPM at this event
    scroll: float  # Scroll speed multiplier
    measure_index: int  # Exact measure number from chart data
    local_density: float  # Notes per second in local window

    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor representation [type_id, beat_pos, duration, bpm, scroll, local_density].

        Returns 6-element tensor (same shape as v1, but local_density replaces gogo).
        """
        return torch.tensor(
            [
                self.note_type,
                self.beat_position,
                self.duration,
                self.bpm,
                self.scroll,
                self.local_density,
            ],
            dtype=torch.float32,
        )


class EventTokenizer:
    """
    Tokenizes Taiko chart data into event token sequences.

    V2 Features:
    - Extracts note events from segments with exact measure indices
    - Computes local note density as explicit feature
    - Creates measure-aligned windows (not time-estimated)
    - Normalizes continuous features (BPM, scroll, density)
    """

    def __init__(
        self,
        bpm_mean: float = 150.0,
        bpm_std: float = 50.0,
        scroll_mean: float = 1.0,
        scroll_std: float = 0.5,
        density_mean: float = 5.0,
        density_std: float = 3.0,
        max_duration: float = 4.0,  # Max roll/balloon duration in beats
        density_window_sec: float = 1.0,  # Window for density calculation
    ):
        self.bpm_mean = bpm_mean
        self.bpm_std = bpm_std
        self.scroll_mean = scroll_mean
        self.scroll_std = scroll_std
        self.density_mean = density_mean
        self.density_std = density_std
        self.max_duration = max_duration
        self.density_window_sec = density_window_sec

    def tokenize_chart(self, segments: list[dict]) -> list[EventToken]:
        """
        Convert chart segments to a list of EventTokens with measure indices.

        Args:
            segments: List of segment dicts from the dataset.
                      Each segment should have 'measure_index' or will be inferred.

        Returns:
            List of EventToken objects, sorted by timestamp
        """
        tokens = []
        measure_counter = 0

        for segment in segments:
            segment_start = segment["timestamp"]
            measure_num = segment.get("measure_num", 4)
            measure_den = segment.get("measure_den", 4)
            notes = segment.get("notes", [])

            # Use explicit measure_index if available, otherwise use counter
            segment_measure_index = segment.get("measure_index", measure_counter)

            for note in notes:
                note_type_str = note.get("note_type", "Don")
                if note_type_str not in NOTE_TYPE_TO_ID:
                    continue  # Skip unknown note types

                # Calculate beat position within measure
                note_time = note.get("timestamp", segment_start)

                # Estimate beat position (simplified - assuming 4/4)
                beat_in_measure = (
                    (note_time - segment_start) * note.get("bpm", 120) / 60
                ) % measure_num
                beat_position = (
                    beat_in_measure / measure_num if measure_num > 0 else 0.0
                )

                # Calculate duration for long notes
                duration = 0.0
                if note_type_str in ["Roll", "RollBig", "Balloon", "BalloonAlt"]:
                    duration = note.get("delay", 0.0)  # Use delay as duration hint

                token = EventToken(
                    timestamp=note_time,
                    beat_position=beat_position,
                    note_type=NOTE_TYPE_TO_ID[note_type_str],
                    duration=min(duration, self.max_duration),
                    bpm=note.get("bpm", 120.0),
                    scroll=note.get("scroll", 1.0),
                    measure_index=segment_measure_index,
                    local_density=0.0,  # Will be computed after all tokens are collected
                )
                tokens.append(token)

            measure_counter += 1

        # Sort by timestamp
        tokens.sort(key=lambda t: t.timestamp)

        # Compute local density for each token
        self._compute_and_set_density(tokens)

        return tokens

    def _compute_and_set_density(self, tokens: list[EventToken]) -> None:
        """Compute and set local_density for each token in place."""
        if not tokens:
            return

        timestamps = np.array([t.timestamp for t in tokens])

        for i, token in enumerate(tokens):
            window_start = token.timestamp - self.density_window_sec / 2
            window_end = token.timestamp + self.density_window_sec / 2
            count = np.sum((timestamps >= window_start) & (timestamps <= window_end))
            # Modify in place using object attribute assignment
            object.__setattr__(token, "local_density", count / self.density_window_sec)

    def create_windows_by_measure(
        self,
        tokens: list[EventToken],
        window_measures: int = 4,
        hop_measures: int = 2,
    ) -> list[list[EventToken]]:
        """
        Create measure-aligned windows from token sequence.

        Unlike v1, this uses exact measure indices from chart data,
        ensuring perfect alignment with musical structure regardless
        of BPM changes or time signature variations.

        Args:
            tokens: List of EventTokens with measure_index set
            window_measures: Window size in number of measures
            hop_measures: Hop size in measures

        Returns:
            List of token subsequences (windows)
        """
        if not tokens:
            return []

        # Get measure range
        min_measure = min(t.measure_index for t in tokens)
        max_measure = max(t.measure_index for t in tokens)

        windows = []
        start_measure = min_measure

        while start_measure <= max_measure:
            end_measure = start_measure + window_measures

            # Get tokens in this measure range
            window_tokens = [
                t for t in tokens if start_measure <= t.measure_index < end_measure
            ]

            if window_tokens:  # Only add non-empty windows
                windows.append(window_tokens)

            start_measure += hop_measures

        return windows

    def create_windows(
        self,
        tokens: list[EventToken],
        window_measures: int = 4,
        hop_measures: int = 2,
        default_bpm: float = 120.0,
    ) -> list[list[EventToken]]:
        """
        Create windows from token sequence.

        V2 uses measure-based windowing by default.
        Falls back to time-based if measure indices aren't available.

        Args:
            tokens: List of EventTokens
            window_measures: Window size in measures
            hop_measures: Hop size in measures
            default_bpm: Default BPM (unused in measure-based mode)

        Returns:
            List of token subsequences (windows)
        """
        if not tokens:
            return []

        # Check if measure indices are meaningful (not all same)
        measure_indices = [t.measure_index for t in tokens]
        if len(set(measure_indices)) > 1:
            # Use measure-based windowing
            return self.create_windows_by_measure(tokens, window_measures, hop_measures)
        else:
            # Fallback to time-based windowing (similar to v1)
            return self._create_windows_by_time(
                tokens, window_measures, hop_measures, default_bpm
            )

    def _create_windows_by_time(
        self,
        tokens: list[EventToken],
        window_measures: int = 4,
        hop_measures: int = 2,
        default_bpm: float = 120.0,
    ) -> list[list[EventToken]]:
        """Fallback time-based windowing (similar to v1)."""
        if not tokens:
            return []

        # Split tokens by BPM changes
        segments = self._split_by_bpm(tokens, threshold=5.0)

        all_windows = []
        for segment_tokens in segments:
            if not segment_tokens:
                continue

            segment_bpm = (
                segment_tokens[0].bpm if segment_tokens[0].bpm > 0 else default_bpm
            )
            beats_per_measure = 4  # Assuming 4/4 time
            measure_duration = (beats_per_measure * 60) / segment_bpm

            window_duration = window_measures * measure_duration
            hop_duration = hop_measures * measure_duration

            start_time = segment_tokens[0].timestamp
            end_time = segment_tokens[-1].timestamp
            current_start = start_time

            while current_start < end_time:
                window_end = current_start + window_duration

                window_tokens = [
                    t
                    for t in segment_tokens
                    if current_start <= t.timestamp < window_end
                ]

                if window_tokens:
                    all_windows.append(window_tokens)

                current_start += hop_duration

        return all_windows

    def _split_by_bpm(
        self,
        tokens: list[EventToken],
        threshold: float = 5.0,
    ) -> list[list[EventToken]]:
        """Split token list into segments with consistent BPM."""
        if not tokens:
            return []

        segments = []
        current_segment = [tokens[0]]
        current_bpm = tokens[0].bpm

        for token in tokens[1:]:
            if abs(token.bpm - current_bpm) > threshold:
                if current_segment:
                    segments.append(current_segment)
                current_segment = [token]
                current_bpm = token.bpm
            else:
                current_segment.append(token)

        if current_segment:
            segments.append(current_segment)

        return segments

    def tokens_to_tensor(
        self,
        tokens: list[EventToken],
        max_length: Optional[int] = None,
        normalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert token list to padded tensor batch.

        Args:
            tokens: List of EventTokens
            max_length: Maximum sequence length (None = no limit)
            normalize: Whether to normalize continuous features

        Returns:
            Tuple of (token_tensor, attention_mask)
            token_tensor: [seq_len, 6] - [type, beat_pos, duration, bpm, scroll, local_density]
            attention_mask: [seq_len] - 1 for real tokens, 0 for padding
        """
        if not tokens:
            return torch.zeros(1, 6), torch.zeros(1)

        # Truncate if needed
        if max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length]

        # Stack token tensors
        tensor = torch.stack([t.to_tensor() for t in tokens])

        if normalize:
            # Normalize BPM (column 3)
            tensor[:, 3] = (tensor[:, 3] - self.bpm_mean) / self.bpm_std
            # Normalize scroll (column 4)
            tensor[:, 4] = (tensor[:, 4] - self.scroll_mean) / self.scroll_std
            # Normalize local_density (column 5)
            tensor[:, 5] = (tensor[:, 5] - self.density_mean) / self.density_std

        # Create attention mask (all 1s for real tokens)
        mask = torch.ones(len(tokens))

        return tensor, mask

    def pad_sequence(
        self,
        tensor: torch.Tensor,
        mask: torch.Tensor,
        target_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pad tensor and mask to target length.

        Args:
            tensor: [seq_len, 6] token tensor
            mask: [seq_len] attention mask
            target_length: Target sequence length

        Returns:
            Padded tensor and mask
        """
        current_length = tensor.size(0)

        if current_length >= target_length:
            return tensor[:target_length], mask[:target_length]

        # Pad tensor
        pad_length = target_length - current_length
        pad_tensor = torch.zeros(pad_length, tensor.size(1))
        pad_tensor[:, 0] = PAD_TOKEN_ID  # Set type to PAD

        padded_tensor = torch.cat([tensor, pad_tensor], dim=0)
        padded_mask = torch.cat([mask, torch.zeros(pad_length)], dim=0)

        return padded_tensor, padded_mask
