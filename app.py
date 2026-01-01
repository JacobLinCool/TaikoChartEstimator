import glob
import os
from dataclasses import dataclass
from typing import Any, Optional

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import ruptures as rpt
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Dynamic imports to avoid loading v1 by default
# from TaikoChartEstimator.data.v1.tokenizer import EventTokenizer
# from TaikoChartEstimator.model.v1.model import TaikoChartEstimator


@dataclass
class ParsedCourse:
    name: str
    level: Optional[int]
    segments: list[dict]
    difficulty_hint: Optional[str]


@dataclass
class ParsedTJA:
    meta: dict[str, Any]
    courses: dict[str, ParsedCourse]


NOTE_DIGIT_TO_TYPE = {
    "1": "Don",
    "2": "Ka",
    "3": "DonBig",
    "4": "KaBig",
    "5": "Roll",
    "6": "RollBig",
    "7": "Balloon",
    "8": "EndOf",
    "9": "BalloonAlt",
}


def _strip_comment(line: str) -> str:
    if "//" in line:
        line = line.split("//", 1)[0]
    return line.strip()


def parse_tja(text: str) -> ParsedTJA:
    """Parse a (single-song) TJA into dataset-like `segments` per course.

    Supported (best-effort): COURSE/LEVEL, BPM, OFFSET, #START/#END,
    #BPMCHANGE, #MEASURE, #SCROLL, #DELAY, #GOGOSTART/#GOGOEND.

    Branching commands are ignored.
    """

    if not text or not text.strip():
        raise ValueError("Empty TJA input")

    text = text.replace("\ufeff", "")
    lines = [_strip_comment(l) for l in text.replace("\r\n", "\n").split("\n")]
    lines = [l for l in lines if l]

    meta: dict[str, Any] = {}
    courses: dict[str, dict[str, Any]] = {}

    current_course: Optional[dict[str, Any]] = None
    in_chart = False

    bpm = 120.0
    offset = 0.0
    measure_num = 4
    measure_den = 4
    scroll = 1.0
    gogo = False

    current_time = 0.0
    measure_start_time = 0.0
    measure_digits: list[str] = []

    def beats_per_measure() -> float:
        # TJA: #MEASURE a/b means measure length = 4 * a / b quarter-note beats
        return 4.0 * float(measure_num) / float(measure_den)

    def measure_duration_sec(local_bpm: float) -> float:
        return beats_per_measure() * 60.0 / max(local_bpm, 1e-6)

    def flush_measure_if_any() -> None:
        nonlocal current_time, measure_start_time, measure_digits
        if current_course is None:
            return
        digits = "".join(measure_digits).strip()
        if not digits:
            return

        dur = measure_duration_sec(bpm)
        step = dur / max(len(digits), 1)
        notes: list[dict] = []
        for i, ch in enumerate(digits):
            if ch == "0":
                continue
            note_type = NOTE_DIGIT_TO_TYPE.get(ch)
            if not note_type:
                continue
            t = measure_start_time + i * step
            notes.append(
                {
                    "note_type": note_type,
                    "timestamp": float(t),
                    "bpm": float(bpm),
                    "scroll": float(scroll),
                    "gogo": bool(gogo),
                }
            )

        current_course["segments"].append(
            {
                "timestamp": float(measure_start_time),
                "measure_num": int(measure_num),
                "measure_den": int(measure_den),
                "notes": notes,
            }
        )

        # Advance time by exactly one measure
        current_time = measure_start_time + dur
        measure_start_time = current_time
        measure_digits = []

    def finalize_long_note_durations() -> None:
        if current_course is None:
            return
        # Flatten notes
        flat: list[dict] = []
        for seg in current_course["segments"]:
            for n in seg.get("notes", []):
                flat.append(n)
        flat.sort(key=lambda n: n.get("timestamp", 0.0))

        open_idx: list[int] = []
        for i, n in enumerate(flat):
            nt = n.get("note_type")
            if nt in {"Roll", "RollBig", "Balloon", "BalloonAlt"}:
                open_idx.append(i)
            elif nt == "EndOf" and open_idx:
                start_i = open_idx.pop()
                start = flat[start_i]
                start_bpm = float(start.get("bpm", 120.0))
                dt = float(n.get("timestamp", 0.0)) - float(start.get("timestamp", 0.0))
                dur_beats = max(0.0, dt * start_bpm / 60.0)
                start["delay"] = float(dur_beats)

    def ensure_course(name: str) -> dict[str, Any]:
        nonlocal courses
        if name not in courses:
            courses[name] = {
                "name": name,
                "level": None,
                "segments": [],
                "difficulty_hint": None,
            }
        return courses[name]

    for raw in lines:
        line = raw.strip()

        if not in_chart and ":" in line and not line.startswith("#"):
            k, v = [p.strip() for p in line.split(":", 1)]
            ku = k.upper()
            meta[ku] = v
            if ku == "BPM":
                try:
                    bpm = float(v)
                except ValueError:
                    pass
            elif ku == "OFFSET":
                try:
                    offset = float(v)
                except ValueError:
                    pass
            elif ku == "COURSE":
                current_course = ensure_course(v)
                # Reset per-course chart state
                in_chart = False
            elif ku == "LEVEL" and current_course is not None:
                try:
                    current_course["level"] = int(float(v))
                except ValueError:
                    current_course["level"] = None
            continue

        if line.startswith("#START"):
            if current_course is None:
                current_course = ensure_course("(default)")
            # Reset chart state at start
            in_chart = True
            bpm = float(meta.get("BPM", bpm) or bpm)
            try:
                offset = float(meta.get("OFFSET", offset) or offset)
            except ValueError:
                offset = offset
            measure_num, measure_den = 4, 4
            scroll = 1.0
            gogo = False
            current_time = 0.0
            measure_start_time = 0.0
            measure_digits = []
            # Apply offset as a global shift (best-effort)
            current_time += float(offset)
            measure_start_time = current_time
            continue

        if not in_chart:
            continue

        if line.startswith("#END"):
            flush_measure_if_any()
            finalize_long_note_durations()
            in_chart = False
            continue

        if line.startswith("#"):
            cmd = line[1:].strip()
            cmd_u = cmd.upper()
            if cmd_u.startswith("BPMCHANGE"):
                flush_measure_if_any()
                try:
                    bpm = float(cmd.split(maxsplit=1)[1])
                except Exception:
                    pass
            elif cmd_u.startswith("MEASURE"):
                flush_measure_if_any()
                try:
                    frac = cmd.split(maxsplit=1)[1].strip()
                    a, b = frac.split("/", 1)
                    measure_num = int(a)
                    measure_den = int(b)
                except Exception:
                    pass
            elif cmd_u.startswith("SCROLL"):
                flush_measure_if_any()
                try:
                    scroll = float(cmd.split(maxsplit=1)[1])
                except Exception:
                    pass
            elif cmd_u.startswith("DELAY"):
                flush_measure_if_any()
                try:
                    current_time += float(cmd.split(maxsplit=1)[1])
                except Exception:
                    pass
                measure_start_time = current_time
            elif cmd_u.startswith("GOGOSTART"):
                flush_measure_if_any()
                gogo = True
            elif cmd_u.startswith("GOGOEND"):
                flush_measure_if_any()
                gogo = False
            else:
                # Ignore other commands (branching etc.)
                pass
            continue

        # Note data: may contain multiple commas
        for ch in line:
            if ch.isdigit():
                measure_digits.append(ch)
            elif ch == ",":
                flush_measure_if_any()

    # Build ParsedTJA
    parsed_courses: dict[str, ParsedCourse] = {}
    difficulty_map = {
        "0": "easy",
        "easy": "easy",
        "1": "normal",
        "normal": "normal",
        "2": "hard",
        "hard": "hard",
        "3": "oni",
        "oni": "oni",
        "4": "oni",
        "ura": "oni",
        "edit": "oni",
    }
    for name, c in courses.items():
        name_l = name.strip().lower()
        hint = difficulty_map.get(name_l)
        parsed_courses[name] = ParsedCourse(
            name=name,
            level=c.get("level"),
            segments=c.get("segments", []),
            difficulty_hint=hint,
        )

    return ParsedTJA(meta=meta, courses=parsed_courses)


def _discover_checkpoints() -> list[str]:
    # Prefer local trained outputs
    paths = []
    for p in glob.glob("outputs/*/pretrained/*"):
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json")):
            paths.append(p)
    # Also accept HF / user-provided paths via manual input
    if not paths:
        return [
            "JacobLinCool/TaikoChartEstimator-20251228",
            "JacobLinCool/TaikoChartEstimator-20251229",
        ]
    return sorted(paths)


_MODEL_CACHE: dict[str, Any] = {}


def _resolve_device(device: str) -> str:
    device = (device or "cpu").lower()
    if device == "cuda" and torch.cuda.is_available():
        return "cuda"
    if (
        device == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def get_model_class(version: str = "v1"):
    if version == "v2":
        from TaikoChartEstimator.model.v2.model import TaikoChartEstimator

        return TaikoChartEstimator
    else:
        from TaikoChartEstimator.model.v1.model import TaikoChartEstimator

        return TaikoChartEstimator


def get_tokenizer_class(version: str = "v1"):
    if version == "v2":
        from TaikoChartEstimator.data.v2.tokenizer import EventTokenizer

        return EventTokenizer
    else:
        from TaikoChartEstimator.data.v1.tokenizer import EventTokenizer

        return EventTokenizer


def _load_model(
    checkpoint_path: str, device: str, version: str = "v1"
) -> Any:  # Returns TaikoChartEstimator
    device = _resolve_device(device)
    key = f"{checkpoint_path}::{device}::{version}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    ModelClass = get_model_class(version)
    model = ModelClass.from_pretrained(checkpoint_path)
    model.eval()
    model.to(torch.device(device))
    _MODEL_CACHE[key] = model
    return model


def _build_instances_from_segments(
    segments: list[dict],
    max_tokens_per_instance: int,
    window_measures: list[int],
    hop_measures: int,
    max_instances_per_chart: int,
    version: str = "v1",
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[float, float]], list[int]
]:
    TokenizerClass = get_tokenizer_class(version)
    tokenizer = TokenizerClass()
    tokens = tokenizer.tokenize_chart(segments)

    all_instances: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    all_times: list[tuple[float, float]] = []
    all_token_counts: list[int] = []

    for window_size in window_measures:
        windows = tokenizer.create_windows(
            tokens, window_measures=window_size, hop_measures=hop_measures
        )
        for window_tokens in windows:
            if not window_tokens:
                continue
            tensor, mask = tokenizer.tokens_to_tensor(
                window_tokens, max_length=max_tokens_per_instance
            )
            all_token_counts.append(int(mask.sum().item()))
            tensor, mask = tokenizer.pad_sequence(tensor, mask, max_tokens_per_instance)
            all_instances.append(tensor)
            all_masks.append(mask)
            all_times.append(
                (float(window_tokens[0].timestamp), float(window_tokens[-1].timestamp))
            )

    if not all_instances:
        raise ValueError("No note events parsed (empty chart or unsupported format)")

    if len(all_instances) > max_instances_per_chart:
        idx = np.linspace(
            0, len(all_instances) - 1, max_instances_per_chart, dtype=int
        ).tolist()
        all_instances = [all_instances[i] for i in idx]
        all_masks = [all_masks[i] for i in idx]
        all_times = [all_times[i] for i in idx]
        all_token_counts = [all_token_counts[i] for i in idx]

    instances = torch.stack(all_instances).unsqueeze(0)  # [1, N, L, 6]
    masks = torch.stack(all_masks).unsqueeze(0)  # [1, N, L]
    counts = torch.tensor([len(all_instances)], dtype=torch.long)  # [1]
    return instances, masks, counts, all_times, all_token_counts


def _plot_attention(
    times: list[tuple[float, float]],
    avg_attention: np.ndarray,
    topk_mask: Optional[np.ndarray],
    title: str,
):
    # Sort by time to avoid misleading zig-zag lines when windows are generated in mixed order.
    t0 = np.array([a for a, _ in times], dtype=np.float64)
    t1 = np.array([b for _, b in times], dtype=np.float64)
    mids = (t0 + t1) / 2.0
    order = np.argsort(mids)

    mids_s = mids[order]
    attn_s = avg_attention[order]
    topk_s = topk_mask[order] if topk_mask is not None else None

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.scatter(mids_s, attn_s, s=14, alpha=0.8, label="Instance")
    ax.plot(mids_s, attn_s, linewidth=1.5, alpha=0.6)

    if topk_s is not None:
        sel = topk_s.astype(bool)
        ax.scatter(
            mids_s[sel],
            attn_s[sel],
            s=40,
            marker="o",
            edgecolors="black",
            linewidths=0.4,
            label="Top-k",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Avg attention (weight)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _plot_branch_heatmap(branch_attn: np.ndarray, title: str):
    # branch_attn: [n_branches, n_instances]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    im = ax.imshow(branch_attn, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Instance (time-sorted)")
    ax.set_ylabel("Branch")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Attention weight")
    fig.tight_layout()
    return fig


def _plot_density_and_attention(
    times: list[tuple[float, float]],
    token_counts: list[int],
    avg_attention: np.ndarray,
    topk_mask: Optional[np.ndarray],
    title: str,
):
    t0 = np.array([a for a, _ in times], dtype=np.float64)
    t1 = np.array([b for _, b in times], dtype=np.float64)
    mids = (t0 + t1) / 2.0
    durations = np.maximum(t1 - t0, 1e-6)
    token_counts_np = np.array(token_counts[: len(times)], dtype=np.float64)
    density = token_counts_np / durations
    order = np.argsort(mids)

    mids_s = mids[order]
    dens_s = density[order]
    attn_s = avg_attention[order]
    topk_s = topk_mask[order] if topk_mask is not None else None

    fig, ax1 = plt.subplots(figsize=(10, 3.2))
    ax1.plot(mids_s, dens_s, linewidth=1.8, color="tab:blue", label="Token density")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Tokens / sec", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.scatter(
        mids_s, attn_s, s=14, color="tab:orange", alpha=0.75, label="Avg attention"
    )
    if topk_s is not None:
        sel = topk_s.astype(bool)
        ax2.scatter(
            mids_s[sel],
            attn_s[sel],
            s=40,
            marker="o",
            edgecolors="black",
            linewidths=0.4,
            color="tab:orange",
            label="Top-k attention",
        )
    ax2.set_ylabel("Avg attention", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.set_title(title)
    # Merge legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    fig.tight_layout()
    return fig


def _plot_local_difficulty(
    times: list[tuple[float, float]],
    local_stars: np.ndarray,
    token_counts: list[int],
    title: str,
):
    """Plot estimated local difficulty (star rating) over time."""
    t0 = np.array([a for a, _ in times], dtype=np.float64)
    t1 = np.array([b for _, b in times], dtype=np.float64)
    mids = (t0 + t1) / 2.0
    durations = np.maximum(t1 - t0, 1e-6)
    token_counts_np = np.array(token_counts[: len(times)], dtype=np.float64)
    density = token_counts_np / durations

    order = np.argsort(mids)
    mids_s = mids[order]
    stars_s = local_stars[order]
    dens_s = density[order]

    # EMA Smoothing
    # Alpha = 2 / (span + 1), for span=4 (approx 8-16s depending on window) -> alpha=0.4
    alpha = 0.3
    if len(stars_s) > 0:
        stars_smooth = np.zeros_like(stars_s)
        stars_smooth[0] = stars_s[0]
        for i in range(1, len(stars_s)):
            stars_smooth[i] = alpha * stars_s[i] + (1 - alpha) * stars_smooth[i - 1]
    else:
        stars_smooth = stars_s

    fig, ax1 = plt.subplots(figsize=(10, 3.5))

    # Plot difficulty curve
    color = "tab:red"
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Estimated Local Stars", color=color)

    # Plot raw faint
    ax1.plot(mids_s, stars_s, color=color, linewidth=1, alpha=0.3, label="Raw")
    # Plot smoothed main
    ax1.plot(mids_s, stars_smooth, color=color, linewidth=2.5, label="Smoothed (EMA)")

    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, alpha=0.25)

    # Fill area under smoothed curve
    ax1.fill_between(mids_s, stars_smooth, alpha=0.1, color=color)

    # Plot density on secondary axis for context
    ax2 = ax1.twinx()
    color2 = "tab:blue"
    ax2.set_ylabel("Density (notes/s)", color=color2)
    ax2.plot(
        mids_s,
        dens_s,
        color=color2,
        linewidth=1,
        linestyle="--",
        alpha=0.5,
        label="Note Density",
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title(title)

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    return fig


def _smooth_embeddings(embeddings: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Apply temporal smoothing (moving average) to embeddings."""
    if len(embeddings) < window_size:
        return embeddings

    # Kernel for simple moving average
    kernel = np.ones(window_size) / window_size

    # Apply to each dimension independenty
    # We can use scipy.ndimage.convolve1d or simplified numpy for dependency-free
    smoothed = np.zeros_like(embeddings)
    for dim in range(embeddings.shape[1]):
        # Padding: 'edge' mode equivalent
        x = embeddings[:, dim]
        pad_width = window_size // 2
        padded = np.pad(x, pad_width, mode="edge")

        # Convolve
        s = np.convolve(padded, kernel, mode="valid")

        # Handle shape mismatch due to even/odd window
        if len(s) > len(x):
            s = s[: len(x)]
        elif len(s) < len(x):
            # Should not happen with padded='edge' widely enough but just in case
            s = np.pad(s, (0, len(x) - len(s)), mode="edge")

        smoothed[:, dim] = s

    return smoothed


def _smooth_labels(labels: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Apply mode filter to labels to enforce temporal continuity."""
    if len(labels) < window_size:
        return labels

    n = len(labels)
    smoothed = labels.copy()
    pad = window_size // 2

    # Simple sliding window mode
    for i in range(n):
        start = max(0, i - pad)
        end = min(n, i + pad + 1)
        window = labels[start:end]

        # Find mode
        counts = np.bincount(window)
        smoothed[i] = np.argmax(counts)

    return smoothed


def _perform_clustering(
    embeddings: np.ndarray,
    min_k: int = 3,
    max_k: int = 8,
    smoothing_window: int = 3,
    label_smoothing_window: int = 3,
    random_state: int = 42,
) -> tuple[np.ndarray, int, dict]:
    """
    Perform K-Means clustering with automatic K selection using Silhouette Score.
    Applying temporal smoothing to stabilize clusters.

    Args:
        embeddings: [N, D] data points
        min_k: Minimum number of clusters
        max_k: Maximum number of clusters

    Returns:
        labels: [N] cluster labels
        best_k: Selected number of clusters
        stats: Info about clustering quality
    """
    # Simply if N is too small
    N = embeddings.shape[0]
    if N < min_k:
        return np.zeros(N, dtype=int), 1, {"score": 0.0}

    # 1. Temporal Smoothing
    if smoothing_window > 1:
        # print(f"Smoothing embeddings with window={smoothing_window}")
        work_embeddings = _smooth_embeddings(embeddings, window_size=smoothing_window)
    else:
        work_embeddings = embeddings

    best_score = -1.0
    best_k = min_k
    best_model = None

    print(f"Clustering {N} instances...")

    effective_max_k = min(max_k, N - 1)
    if effective_max_k < min_k:
        effective_max_k = min_k

    for k in range(min_k, effective_max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(work_embeddings)
        try:
            score = silhouette_score(work_embeddings, labels)
            # print(f"K={k}, Silhouette={score:.4f}")
            if score > best_score:
                best_score = score
                best_k = k
                best_model = kmeans
        except Exception:
            pass

    if best_model is None:
        # Fallback
        kmeans = KMeans(n_clusters=min_k, random_state=random_state, n_init=10)
        kmeans.fit(work_embeddings)
        best_model = kmeans
        best_k = min_k

    labels = best_model.labels_

    # 2. Label Smoothing (Post-processing)
    if label_smoothing_window > 1:
        labels = _smooth_labels(labels, window_size=label_smoothing_window)

    return labels, best_k, {"silhouette": best_score}


def _analyze_clusters(
    cluster_labels: np.ndarray,
    local_stars: np.ndarray,
    note_density: np.ndarray,
    avg_attention: Optional[np.ndarray] = None,
) -> list[dict]:
    """
    Analyze properties of each cluster to create a profile.

    Returns list of dicts: [{id, count, avg_stars, avg_density, avg_attn, desc}]
    """
    unique_labels = np.unique(cluster_labels)
    profiles = []

    for label in unique_labels:
        mask = cluster_labels == label
        count = mask.sum()

        avg_s = local_stars[mask].mean() if len(local_stars) > 0 else 0
        avg_d = note_density[mask].mean() if len(note_density) > 0 else 0
        avg_a = avg_attention[mask].mean() if avg_attention is not None else 0

        profiles.append(
            {
                "Cluster ID": int(label),
                "Count": int(count),
                "Avg Stars": float(f"{avg_s:.2f}"),
                "Avg Density": float(f"{avg_d:.2f}"),
                "Avg Attention": float(f"{avg_a:.4f}"),
            }
        )

    # Sort by Avg Stars to make it intuitive (Cluster 0 = Easiest or Hardest?)
    # Let's keep ID but maybe we can add a rank?
    # Sorting purely by ID is safer for consistency with plot colors.
    profiles.sort(key=lambda x: x["Cluster ID"])
    return profiles


def _plot_clusters(
    times: list[tuple[float, float]],
    cluster_labels: np.ndarray,
    local_stars: np.ndarray,
    title: str,
):
    """Plot timeline colored by cluster ID."""
    t0 = np.array([a for a, _ in times], dtype=np.float64)
    t1 = np.array([b for _, b in times], dtype=np.float64)
    mids = (t0 + t1) / 2.0

    # Sort
    order = np.argsort(mids)
    mids_s = mids[order]
    stars_s = local_stars[order]
    labels_s = cluster_labels[order]

    unique_labels = np.unique(labels_s)
    n_clusters = len(unique_labels)

    # Use a distinct colormap
    cmap = plt.get_cmap("tab10" if n_clusters <= 10 else "tab20")

    fig, ax = plt.subplots(figsize=(10, 3.5))

    # We want to plot segments. Since they are time-sorted, we can just scatter or valid-bar plot.
    # A step plot or bar plot might be good.
    # Let's use a scatter plot for simplicity but heavy markers.

    for i, label in enumerate(unique_labels):
        mask = labels_s == label
        ax.scatter(
            mids_s[mask],
            stars_s[mask],
            color=cmap(i),
            label=f"Cluster {label}",
            s=20,
            alpha=0.8,
        )

    # Also plot a faint line to show connectivity
    ax.plot(mids_s, stars_s, color="gray", alpha=0.2, linewidth=1)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Local Stars")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    return fig


def _detect_segments(
    local_stars: np.ndarray,
    times: list[tuple[float, float]],
    min_segment_size: int = 3,
    penalty_scale: float = 1.0,
) -> list[dict]:
    """
    Detect segments using Change Point Detection.

    IMPORTANT: Windows may not be in temporal order (e.g., mixed window sizes).
    We sort by midpoint time first to ensure temporal coherence.
    """
    n = len(local_stars)
    if n < min_segment_size * 2:
        return [
            {
                "start_time": times[0][0],
                "end_time": times[-1][1],
                "avg_stars": float(local_stars.mean()),
                "n_windows": n,
            }
        ]

    # Calculate window midpoints
    mids = np.array([(t0 + t1) / 2 for t0, t1 in times])

    # SORT by midpoint time (critical for temporal coherence!)
    order = np.argsort(mids)
    mids_sorted = mids[order]
    stars_sorted = local_stars[order]
    times_sorted = [times[i] for i in order]

    # Build cell boundaries (1D Voronoi on SORTED windows)
    cell_bounds = [times_sorted[0][0]]  # Song start
    for i in range(len(mids_sorted) - 1):
        cell_bounds.append((mids_sorted[i] + mids_sorted[i + 1]) / 2)
    cell_bounds.append(times_sorted[-1][1])  # Song end

    # Ruptures detection (on SORTED data)
    signal = stars_sorted.reshape(-1, 1)
    penalty = np.var(stars_sorted) * penalty_scale
    algo = rpt.Pelt(model="l2", min_size=min_segment_size).fit(signal)
    change_points = algo.predict(pen=penalty)

    # Build segments
    segments = []
    prev_idx = 0

    for cp in change_points:
        seg_stars = stars_sorted[prev_idx:cp]

        start_t = cell_bounds[prev_idx]
        end_t = cell_bounds[cp]

        segments.append(
            {
                "start_time": float(start_t),
                "end_time": float(end_t),
                "avg_stars": float(seg_stars.mean()),
                "n_windows": cp - prev_idx,
            }
        )
        prev_idx = cp

    return segments


def _plot_segments(
    times: list[tuple[float, float]],
    local_stars: np.ndarray,
    segments: list[dict],
    title: str,
):
    """
    Plot local difficulty with segment backgrounds (non-overlapping).
    """
    t0 = np.array([a for a, _ in times], dtype=np.float64)
    t1 = np.array([b for _, b in times], dtype=np.float64)
    mids = (t0 + t1) / 2.0

    order = np.argsort(mids)
    mids_s = mids[order]
    stars_s = local_stars[order]

    # Colormap: Red=Hard, Green=Easy
    cmap = plt.get_cmap("RdYlGn_r")

    fig, ax = plt.subplots(figsize=(12, 4))

    # Normalize colors
    max_star = max(s["avg_stars"] for s in segments) if segments else 10
    min_star = min(s["avg_stars"] for s in segments) if segments else 0
    star_range = max(max_star - min_star, 1)

    # Draw segment backgrounds (should NOT overlap now)
    for seg in segments:
        color = cmap((seg["avg_stars"] - min_star) / star_range)
        ax.axvspan(
            seg["start_time"], seg["end_time"], alpha=0.3, color=color, linewidth=0
        )

        # Horizontal line at segment average
        ax.hlines(
            y=seg["avg_stars"],
            xmin=seg["start_time"],
            xmax=seg["end_time"],
            colors=color,
            linewidth=3,
            alpha=0.9,
        )

        # Label (only if segment is wide enough)
        duration = seg["end_time"] - seg["start_time"]
        if duration > 4:  # Only label if > 4 seconds
            mid_x = (seg["start_time"] + seg["end_time"]) / 2
            ax.text(
                mid_x,
                seg["avg_stars"] + 0.02,
                f"{seg['avg_stars']:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="black",
                alpha=0.8,
            )

    # Raw data on top
    ax.plot(mids_s, stars_s, color="gray", alpha=0.4, linewidth=1)

    # Boundary lines
    for seg in segments[1:]:
        ax.axvline(
            x=seg["start_time"], color="black", linewidth=1, linestyle="--", alpha=0.5
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Raw Score")
    ax.set_title(title)
    ax.set_ylim(bottom=0, top=max_star + 2)
    ax.grid(True, alpha=0.15, axis="y")

    fig.tight_layout()
    return fig


def _plot_attention_concentration(
    avg_attention: np.ndarray,
    title: str,
):
    # Cumulative mass of attention sorted by weight (how concentrated the model is)
    attn = np.clip(avg_attention.astype(np.float64), 0.0, None)
    if attn.sum() > 0:
        attn = attn / attn.sum()
    attn_sorted = np.sort(attn)[::-1]
    cum = np.cumsum(attn_sorted)
    k = np.arange(1, len(attn_sorted) + 1)

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(k, cum, linewidth=2)
    ax.set_xlabel("Top-k instances (sorted by attention)")
    ax.set_ylabel("Cumulative attention mass")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def run_inference(
    file_obj,
    text_content,
    course_name,
    checkpoint_path,
    device,
    window_measures_str,
    hop_measures,
    max_instances,
    version="v1",
):
    if file_obj:
        with open(file_obj.name, "r", encoding="utf-8", errors="ignore") as f:
            text_content = f.read()

    parsed = parse_tja(text_content)
    if not parsed.courses:
        raise gr.Error("No COURSE found and no chart parsed.")

    if course_name not in parsed.courses:
        # Fallback to first
        course_name = next(iter(parsed.courses.keys()))

    course = parsed.courses[course_name]

    try:
        window_measures = [
            int(x.strip()) for x in window_measures_str.split(",") if x.strip()
        ]
    except ValueError:
        raise gr.Error(
            "window_measures must be a comma-separated list of integers, e.g. 2,4"
        )
    if not window_measures:
        window_measures = [2, 4]

    device = _resolve_device(device)
    model = _load_model(checkpoint_path, device, version=version)
    max_tokens = int(getattr(model.config, "max_seq_len", 128))

    instances, masks, counts, times, token_counts = _build_instances_from_segments(
        course.segments,
        max_tokens_per_instance=max_tokens,
        window_measures=window_measures,
        hop_measures=int(hop_measures),
        max_instances_per_chart=int(max_instances),
        version=version,
    )

    instances = instances.to(torch.device(device))
    masks = masks.to(torch.device(device))
    counts = counts.to(torch.device(device))

    difficulty_hint = None
    if course.difficulty_hint is not None:
        mapping = {"easy": 0, "normal": 1, "hard": 2, "oni": 3, "ura": 4}
        difficulty_hint = torch.tensor(
            [mapping[course.difficulty_hint]], device=torch.device(device)
        )

    with torch.no_grad():
        out = model.forward(
            instances,
            masks,
            counts,
            difficulty_hint=difficulty_hint,
            return_attention=True,
        )

    # Scalars
    difficulty_names = ["easy", "normal", "hard", "oni", "ura"]
    pred_class_id = int(out.difficulty_logits.argmax(dim=-1).item())
    pred_class = difficulty_names[pred_class_id]
    raw_score = float(out.raw_score.item())
    raw_star = float(out.raw_star.item())
    display_star = float(out.display_star.item())

    # Attention details
    attn = out.attention_info
    avg_attn = attn.get("average_attention")
    branch_attn = attn.get("branch_attentions")
    topk_mask = attn.get("topk_mask")

    # Local Difficulty Estimation (Probe)
    # Use the predicted class ID if no hint was provided
    calib_diff_id = difficulty_hint
    if calib_diff_id is None:
        calib_diff_id = out.difficulty_logits.argmax(dim=-1, keepdim=True)  # [1, 1]

    local_raw, local_stars = model.get_instance_scores(
        out.instance_embeddings, difficulty_class_id=calib_diff_id.view(-1)
    )

    avg_attn_np = (
        avg_attn[0, : counts.item()].detach().cpu().numpy()
        if avg_attn is not None
        else None
    )
    topk_np = (
        topk_mask[0, : counts.item()].detach().cpu().numpy()
        if topk_mask is not None
        else None
    )
    branch_np = (
        branch_attn[0, :, : counts.item()].detach().cpu().numpy()
        if branch_attn is not None
        else None
    )
    local_stars_np = local_stars[0, : counts.item()].detach().cpu().numpy()
    local_raw_np = local_raw[0, : counts.item()].detach().cpu().numpy()

    # Plots
    fig_attn = None
    fig_heat = None
    fig_density = None
    fig_conc = None
    if avg_attn_np is not None:
        fig_attn = _plot_attention(
            times, avg_attn_np, topk_np, title="MIL average attention over time"
        )
    if avg_attn_np is not None:
        fig_density = _plot_density_and_attention(
            times,
            token_counts,
            avg_attn_np,
            topk_np,
            title="Token density vs attention (time-sorted)",
        )
        fig_conc = _plot_attention_concentration(
            avg_attn_np,
            title="Attention concentration (how many windows dominate)",
        )

    fig_local_diff = None
    if local_stars_np is not None:
        fig_local_diff = _plot_local_difficulty(
            times,
            local_stars_np,
            token_counts,
            title=f"Estimated Local Difficulty Curve (Assuming {pred_class} calibration)",
        )

    # Segment Detection (Piecewise Constant Change Point Detection)
    fig_segments = None
    segment_table_df = None

    if local_raw_np is not None and len(times) > 0:
        segments = _detect_segments(
            local_raw_np,  # Use raw score instead of stars
            times,
            min_segment_size=3,
            penalty_scale=0.5,
        )

        # Create table rows
        seg_rows = []
        for i, seg in enumerate(segments):
            seg_rows.append(
                [
                    i + 1,
                    f"{seg['start_time']:.1f}",
                    f"{seg['end_time']:.1f}",
                    f"{seg['end_time'] - seg['start_time']:.1f}",
                    f"{seg['avg_stars']:.1f}",  # This is now avg_raw
                    seg["n_windows"],
                ]
            )
        segment_table_df = seg_rows

        fig_segments = _plot_segments(
            times,
            local_raw_np,  # Use raw score
            segments,
            title=f"Chart Structure: {len(segments)} Segments Detected",
        )

    # Meta/details
    if branch_np is not None:
        mids = np.array([(a + b) / 2.0 for a, b in times], dtype=np.float64)
        order = np.argsort(mids)
        branch_sorted = branch_np[:, order]
        fig_heat = _plot_branch_heatmap(
            branch_sorted, title="MIL attention (branches x instances)"
        )
        # Add a few time tick labels
        ax = fig_heat.axes[0]
        if len(order) > 1:
            n_ticks = 6
            tick_pos = np.linspace(0, len(order) - 1, n_ticks, dtype=int)
            tick_labels = [f"{mids[order[p]]:.0f}s" for p in tick_pos]
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels)

    # Table
    rows = []
    for i, (t0, t1) in enumerate(times):
        rows.append(
            [
                i,
                float(t0),
                float(t1),
                float((t0 + t1) / 2.0),
                int(token_counts[i]) if i < len(token_counts) else None,
                float(avg_attn_np[i]) if avg_attn_np is not None else None,
                int(topk_np[i]) if topk_np is not None else None,
                float(local_stars_np[i]) if i < len(local_stars_np) else None,
            ]
        )

    # More intuitive summary: show top attention windows
    top_md = ""
    if avg_attn_np is not None:
        t0 = np.array([a for a, _ in times], dtype=np.float64)
        t1 = np.array([b for _, b in times], dtype=np.float64)
        mids = (t0 + t1) / 2.0
        durations = np.maximum(t1 - t0, 1e-6)
        token_counts_np = np.array(token_counts[: len(times)], dtype=np.float64)
        density = token_counts_np / durations

        top_n = min(8, len(avg_attn_np))
        top_idx = np.argsort(avg_attn_np)[::-1][:top_n]

        lines = ["### Top segments (by attention)"]
        for rank, idx in enumerate(top_idx, start=1):
            is_topk = int(topk_np[idx]) if topk_np is not None else 0
            lines.append(
                f"{rank}. `[{t0[idx]:.1f}s - {t1[idx]:.1f}s]` "
                f"attn={avg_attn_np[idx]:.4f}, dens={density[idx]:.1f} tok/s, topk={is_topk}"
            )
        top_md = "\n".join(lines)

    # Meta/details
    meta_out = {
        "TITLE": parsed.meta.get("TITLE"),
        "BPM": parsed.meta.get("BPM"),
        "OFFSET": parsed.meta.get("OFFSET"),
        "COURSE": course.name,
        "LEVEL": course.level,
        "difficulty_hint": course.difficulty_hint,
        "n_instances": int(counts.item()),
        "max_tokens_per_instance": int(max_tokens),
        "window_measures": window_measures,
        "hop_measures": int(hop_measures),
        "attention_entropy": (
            float(attn.get("entropy")[0].item())
            if attn.get("entropy") is not None
            else None
        ),
        "attention_effective_n": (
            float(attn.get("effective_n")[0].item())
            if attn.get("effective_n") is not None
            else None
        ),
        "attention_top5_mass": (
            float(attn.get("top5_mass")[0].item())
            if attn.get("top5_mass") is not None
            else None
        ),
    }

    summary_md = (
        f"### Prediction\n"
        f"- predicted difficulty: `{pred_class}`\n"
        f"- raw_score: `{raw_score:.4f}`\n"
        f"- raw_star: `{raw_star:.4f}`\n"
        f"- display_star: `{display_star:.4f}`\n"
    )

    return (
        summary_md,
        meta_out,
        fig_attn,
        fig_density,
        fig_heat,
        fig_conc,
        top_md,
        rows,
        fig_local_diff,
        fig_segments,
        segment_table_df,
    )


def _update_course_dropdown(tja_file, tja_text: str):
    if tja_file:
        with open(tja_file, "r", encoding="utf-8", errors="ignore") as f:
            tja_text = f.read()
    try:
        parsed = parse_tja(tja_text)
        choices = list(parsed.courses.keys())
        value = choices[0] if choices else None
        return gr.Dropdown(choices=choices, value=value)
    except Exception:
        return gr.Dropdown(choices=[], value=None)


def build_app() -> gr.Blocks:
    checkpoints = _discover_checkpoints()

    with gr.Blocks(title="Taiko Estimator") as demo:
        # State for version (CLI override or UI default)
        version_state = gr.State(value="v1")

        gr.Markdown("# Taiko Difficulty Estimator")

        with gr.Row():
            # Left: Input (Upload/Paste with tabs)
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Upload"):
                        tja_file = gr.File(label="Upload TJA file")
                    with gr.TabItem("Paste"):
                        tja_text = gr.Textbox(label="Paste TJA content", lines=12)

                course = gr.Dropdown(label="COURSE", choices=[], value=None)
                btn = gr.Button("Run Inference", variant="primary", size="lg")

            # Right: Options
            with gr.Column(scale=1):
                gr.Markdown("### Options")
                checkpoint = gr.Dropdown(
                    label="Checkpoint",
                    choices=checkpoints,
                    value=checkpoints[-1] if checkpoints else None,
                    allow_custom_value=True,
                )
                device = gr.Dropdown(
                    label="Device", choices=["cpu", "mps", "cuda"], value="cpu"
                )

                with gr.Accordion("Advanced", open=False):
                    window_measures = gr.Textbox(
                        label="window_measures (comma-separated)", value="2,4"
                    )
                    hop_measures = gr.Slider(
                        label="hop_measures", minimum=1, maximum=8, value=2, step=1
                    )
                    max_instances = gr.Slider(
                        label="max_instances", minimum=1, maximum=512, value=128, step=1
                    )
                    model_version = gr.Dropdown(
                        choices=["v1", "v2"], value="v1", label="Model Version"
                    )

        with gr.Row():
            with gr.Column(scale=1):
                summary = gr.Markdown()
                top_segments = gr.Markdown()
            with gr.Column(scale=1):
                meta_json = gr.JSON(label="Metadata")

        with gr.Tabs():
            with gr.TabItem("Chart Structure"):
                gr.Markdown("### Automatic Segment Detection")
                gr.Markdown(
                    "Detects distinct sections based on difficulty changes (Piecewise Constant Model)."
                )
                plot_segments = gr.Plot(label="Detected Segments")
                segment_table = gr.Dataframe(
                    headers=[
                        "#",
                        "Start (s)",
                        "End (s)",
                        "Duration",
                        "Avg Raw",
                        "Windows",
                    ],
                    datatype=["number", "str", "str", "str", "str", "number"],
                    label="Segment Details",
                )
            with gr.TabItem("Local Difficulty"):
                plot_local_diff = gr.Plot(label="Local Difficulty Curve")
            with gr.TabItem("Attention & Density"):
                plot_density = gr.Plot(label="Density vs Attention")
            with gr.TabItem("Attention Details"):
                plot_attn = gr.Plot(label="Raw Attention")
            with gr.TabItem("Heatmap"):
                plot_heat = gr.Plot(label="Branch Heatmap")
            with gr.TabItem("Concentration"):
                plot_conc = gr.Plot(label="Concentration")
            with gr.TabItem("Raw Data"):
                # headers needs to match rows
                df = gr.Dataframe(
                    headers=[
                        "id",
                        "start",
                        "end",
                        "mid",
                        "tokens",
                        "attention",
                        "is_topk",
                        "local_stars",
                    ],
                    datatype=[
                        "number",
                        "number",
                        "number",
                        "number",
                        "number",
                        "number",
                        "number",
                        "number",
                    ],
                )

        # Auto-refresh COURSE choices when input changes
        tja_file.change(
            _update_course_dropdown, inputs=[tja_file, tja_text], outputs=[course]
        )
        tja_text.change(
            _update_course_dropdown, inputs=[tja_file, tja_text], outputs=[course]
        )

        btn.click(
            run_inference,
            inputs=[
                tja_file,
                tja_text,
                course,
                checkpoint,
                device,
                window_measures,
                hop_measures,
                max_instances,
                model_version,
            ],
            outputs=[
                summary,
                meta_json,
                plot_attn,
                plot_density,
                plot_heat,
                plot_conc,
                top_segments,
                df,
                plot_local_diff,
                plot_segments,
                segment_table,
            ],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
