"""
Event Tokenizer for Taiko Chart Notes

Converts raw chart note data into event tokens suitable for sequence modeling.
Handles 9 note types with continuous features (BPM, scroll, timestamp, duration).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

# Import from centralized constants
from ..constants import (
    DIFFICULTY_ORDER,
    NOTE_TYPE_TO_ID,
    NOTE_TYPES,
    PAD_TOKEN_ID,
)
from ..constants import (
    STAR_RANGES_BY_NAME as STAR_RANGES,
)


@dataclass
class EventToken:
    """A single event token representing a note or event in the chart."""

    timestamp: float  # Absolute time in seconds
    beat_position: float  # Position within the measure (0-1)
    note_type: int  # ID from NOTE_TYPE_TO_ID
    duration: float  # Duration for rolls/balloons (0 for regular notes)
    bpm: float  # Current BPM at this event
    scroll: float  # Scroll speed multiplier
    gogo: bool  # Whether in GOGO time (increased scoring)

    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor representation [type_id, beat_pos, duration, bpm, scroll, gogo]."""
        return torch.tensor(
            [
                self.note_type,
                self.beat_position,
                self.duration,
                self.bpm,
                self.scroll,
                float(self.gogo),
            ],
            dtype=torch.float32,
        )


class EventTokenizer:
    """
    Tokenizes Taiko chart data into event token sequences.

    Features:
    - Extracts note events from segments
    - Computes beat-relative positions
    - Normalizes continuous features (BPM, scroll)
    - Creates beat-aligned windows for MIL instances
    """

    def __init__(
        self,
        bpm_mean: float = 150.0,
        bpm_std: float = 50.0,
        scroll_mean: float = 1.0,
        scroll_std: float = 0.5,
        max_duration: float = 4.0,  # Max roll/balloon duration in beats
    ):
        self.bpm_mean = bpm_mean
        self.bpm_std = bpm_std
        self.scroll_mean = scroll_mean
        self.scroll_std = scroll_std
        self.max_duration = max_duration

    def tokenize_chart(self, segments: list[dict]) -> list[EventToken]:
        """
        Convert chart segments to a list of EventTokens.

        Args:
            segments: List of segment dicts from the dataset

        Returns:
            List of EventToken objects, sorted by timestamp
        """
        tokens = []

        for segment in segments:
            segment_start = segment["timestamp"]
            measure_num = segment.get("measure_num", 4)
            measure_den = segment.get("measure_den", 4)
            notes = segment.get("notes", [])

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
                    # Duration will be until EndOf, but we estimate from context
                    duration = note.get("delay", 0.0)  # Use delay as duration hint

                token = EventToken(
                    timestamp=note_time,
                    beat_position=beat_position,
                    note_type=NOTE_TYPE_TO_ID[note_type_str],
                    duration=min(duration, self.max_duration),
                    bpm=note.get("bpm", 120.0),
                    scroll=note.get("scroll", 1.0),
                    gogo=note.get("gogo", False),
                )
                tokens.append(token)

        # Sort by timestamp
        tokens.sort(key=lambda t: t.timestamp)
        return tokens

    def compute_note_density(
        self, tokens: list[EventToken], window_sec: float = 1.0
    ) -> list[float]:
        """
        Compute local note density for each token (notes per second in window).

        Args:
            tokens: List of EventTokens
            window_sec: Window size in seconds for density calculation

        Returns:
            List of density values, one per token
        """
        if not tokens:
            return []

        timestamps = np.array([t.timestamp for t in tokens])
        densities = []

        for i, t in enumerate(tokens):
            # Count notes in window centered on this note
            window_start = t.timestamp - window_sec / 2
            window_end = t.timestamp + window_sec / 2
            count = np.sum((timestamps >= window_start) & (timestamps <= window_end))
            density = count / window_sec
            densities.append(density)

        return densities

    def create_windows(
        self,
        tokens: list[EventToken],
        window_measures: int = 4,
        hop_measures: int = 2,
        default_bpm: float = 120.0,
    ) -> list[list[EventToken]]:
        """
        Create beat-aligned windows from token sequence, respecting BPM changes.

        Windows are created within BPM-consistent segments to ensure proper
        beat alignment. This prevents window boundaries from falling on
        off-beats when BPM changes occur.

        Args:
            tokens: List of EventTokens
            window_measures: Window size in measures
            hop_measures: Hop size in measures
            default_bpm: Default BPM if not available

        Returns:
            List of token subsequences (windows)
        """
        if not tokens:
            return []

        # Split tokens by BPM changes
        segments = self._split_by_bpm(tokens, threshold=5.0)

        all_windows = []
        for segment_tokens in segments:
            if not segment_tokens:
                continue

            # Use this segment's BPM for window calculation
            segment_bpm = (
                segment_tokens[0].bpm if segment_tokens[0].bpm > 0 else default_bpm
            )
            beats_per_measure = 4  # Assuming 4/4 time
            measure_duration = (beats_per_measure * 60) / segment_bpm

            window_duration = window_measures * measure_duration
            hop_duration = hop_measures * measure_duration

            # Create windows within this segment
            start_time = segment_tokens[0].timestamp
            end_time = segment_tokens[-1].timestamp
            current_start = start_time

            while current_start < end_time:
                window_end = current_start + window_duration

                # Get tokens in this window
                window_tokens = [
                    t
                    for t in segment_tokens
                    if current_start <= t.timestamp < window_end
                ]

                if window_tokens:  # Only add non-empty windows
                    all_windows.append(window_tokens)

                current_start += hop_duration

        return all_windows

    def _split_by_bpm(
        self,
        tokens: list[EventToken],
        threshold: float = 5.0,
    ) -> list[list[EventToken]]:
        """
        Split token list into segments with consistent BPM.

        Args:
            tokens: List of EventTokens sorted by timestamp
            threshold: BPM difference threshold to trigger a new segment

        Returns:
            List of token lists, one per BPM segment
        """
        if not tokens:
            return []

        segments = []
        current_segment = [tokens[0]]
        current_bpm = tokens[0].bpm

        for token in tokens[1:]:
            if abs(token.bpm - current_bpm) > threshold:
                # BPM changed significantly, start new segment
                if current_segment:
                    segments.append(current_segment)
                current_segment = [token]
                current_bpm = token.bpm
            else:
                current_segment.append(token)

        # Don't forget the last segment
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
            token_tensor: [seq_len, 6] - [type, beat_pos, duration, bpm, scroll, gogo]
            attention_mask: [seq_len] - 1 for real tokens, 0 for padding
        """
        if not tokens:
            # Return empty tensors
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
