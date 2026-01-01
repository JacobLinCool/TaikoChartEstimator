"""
Audio Processing for Taiko Chart Estimation

Handles mel spectrogram extraction and alignment with chart events.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T


class AudioProcessor:
    """
    Processes audio waveforms into mel spectrograms for model input.

    Features:
    - Mel spectrogram extraction with configurable parameters
    - Window extraction aligned with chart timing
    - Optional augmentation (time stretch, pitch shift)
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        f_min: float = 20.0,
        f_max: float = 8000.0,
        normalize: bool = True,
    ):
        """
        Initialize audio processor.

        Args:
            sample_rate: Target sample rate for audio
            n_mels: Number of mel frequency bins
            n_fft: FFT window size
            hop_length: Hop length for STFT
            f_min: Minimum frequency for mel filterbank
            f_max: Maximum frequency for mel filterbank
            normalize: Whether to normalize spectrograms
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.normalize = normalize

        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )

        # Amplitude to dB
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

        # Resampler cache
        self._resamplers: dict[int, T.Resample] = {}

    def _get_resampler(self, orig_sr: int) -> T.Resample:
        """Get or create a resampler for the given source sample rate."""
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = T.Resample(orig_sr, self.sample_rate)
        return self._resamplers[orig_sr]

    def process_audio(
        self,
        waveform: np.ndarray | torch.Tensor,
        orig_sample_rate: int,
    ) -> torch.Tensor:
        """
        Process raw audio waveform to mel spectrogram.

        Args:
            waveform: Audio waveform array [samples] or [channels, samples]
            orig_sample_rate: Original sample rate of the audio

        Returns:
            Mel spectrogram tensor [n_mels, time_frames]
        """
        # Convert to tensor if needed
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()

        # Ensure 2D [channels, samples]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Convert stereo to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if orig_sample_rate != self.sample_rate:
            resampler = self._get_resampler(orig_sample_rate)
            waveform = resampler(waveform)

        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)

        # Convert to dB scale
        mel_spec_db = self.amplitude_to_db(mel_spec)

        # Remove channel dimension
        mel_spec_db = mel_spec_db.squeeze(0)

        # Normalize if requested
        if self.normalize:
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (
                mel_spec_db.std() + 1e-8
            )

        return mel_spec_db

    def time_to_frame(self, time_sec: float) -> int:
        """Convert time in seconds to frame index."""
        return int(time_sec * self.sample_rate / self.hop_length)

    def frame_to_time(self, frame_idx: int) -> float:
        """Convert frame index to time in seconds."""
        return frame_idx * self.hop_length / self.sample_rate

    def extract_window(
        self,
        mel_spec: torch.Tensor,
        start_time: float,
        end_time: float,
        pad_value: float = 0.0,
    ) -> torch.Tensor:
        """
        Extract a time window from mel spectrogram.

        Args:
            mel_spec: Full mel spectrogram [n_mels, time_frames]
            start_time: Window start time in seconds
            end_time: Window end time in seconds
            pad_value: Value for padding if window extends beyond spectrogram

        Returns:
            Window tensor [n_mels, window_frames]
        """
        start_frame = self.time_to_frame(start_time)
        end_frame = self.time_to_frame(end_time)

        # Clamp to valid range
        start_frame = max(0, start_frame)
        end_frame = min(mel_spec.size(1), end_frame)

        window = mel_spec[:, start_frame:end_frame]

        # Pad if window is shorter than expected
        expected_frames = self.time_to_frame(end_time - start_time)
        if window.size(1) < expected_frames:
            pad_size = expected_frames - window.size(1)
            window = F.pad(window, (0, pad_size), value=pad_value)

        return window

    def extract_windows_for_instances(
        self,
        mel_spec: torch.Tensor,
        instance_times: list[tuple[float, float]],
        fixed_frames: Optional[int] = None,
    ) -> list[torch.Tensor]:
        """
        Extract mel spectrogram windows aligned with chart instances.

        Args:
            mel_spec: Full mel spectrogram [n_mels, time_frames]
            instance_times: List of (start_time, end_time) for each instance
            fixed_frames: If provided, resize all windows to this frame count

        Returns:
            List of window tensors
        """
        windows = []

        for start_time, end_time in instance_times:
            window = self.extract_window(mel_spec, start_time, end_time)

            if fixed_frames is not None and window.size(1) != fixed_frames:
                # Resize to fixed frame count
                window = F.interpolate(
                    window.unsqueeze(0),
                    size=fixed_frames,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0)

            windows.append(window)

        return windows

    def compute_onset_strength(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Compute onset strength envelope from mel spectrogram.

        Useful for beat tracking and rhythm analysis.

        Args:
            mel_spec: Mel spectrogram [n_mels, time_frames]

        Returns:
            Onset strength envelope [time_frames]
        """
        # Compute first-order difference
        diff = torch.diff(mel_spec, dim=1)

        # Half-wave rectify (keep only positive changes)
        diff = F.relu(diff)

        # Sum across frequency bins
        onset_env = diff.sum(dim=0)

        # Pad to match original length
        onset_env = F.pad(onset_env, (1, 0))

        return onset_env
