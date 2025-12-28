import glob
import os
from dataclasses import dataclass
from typing import Any, Optional

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch

from TaikoChartEstimator.data.tokenizer import EventTokenizer
from TaikoChartEstimator.model.model import TaikoChartEstimator


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
    return sorted(paths)


_MODEL_CACHE: dict[str, TaikoChartEstimator] = {}


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


def _load_model(checkpoint_path: str, device: str) -> TaikoChartEstimator:
    device = _resolve_device(device)
    key = f"{checkpoint_path}::{device}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    model = TaikoChartEstimator.from_pretrained(checkpoint_path)
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
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[float, float]], list[int]
]:
    tokenizer = EventTokenizer()
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
    tja_file,
    tja_text: str,
    course_name: str,
    checkpoint_path: str,
    device: str,
    window_measures_text: str,
    hop_measures: int,
    max_instances: int,
):
    if tja_file:
        with open(tja_file, "r", encoding="utf-8", errors="ignore") as f:
            tja_text = f.read()

    parsed = parse_tja(tja_text)
    if not parsed.courses:
        raise gr.Error("No COURSE found and no chart parsed.")

    if course_name not in parsed.courses:
        # Fallback to first
        course_name = next(iter(parsed.courses.keys()))

    course = parsed.courses[course_name]

    try:
        window_measures = [
            int(x.strip()) for x in window_measures_text.split(",") if x.strip()
        ]
    except ValueError:
        raise gr.Error(
            "window_measures must be a comma-separated list of integers, e.g. 2,4"
        )
    if not window_measures:
        window_measures = [2, 4]

    device = _resolve_device(device)
    model = _load_model(checkpoint_path, device=device)
    max_tokens = int(getattr(model.config, "max_seq_len", 128))

    instances, masks, counts, times, token_counts = _build_instances_from_segments(
        course.segments,
        max_tokens_per_instance=max_tokens,
        window_measures=window_measures,
        hop_measures=int(hop_measures),
        max_instances_per_chart=int(max_instances),
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

    # Heatmap: sort instances by time for interpretability
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
        "attention_entropy": float(attn.get("entropy")[0].item())
        if attn.get("entropy") is not None
        else None,
        "attention_effective_n": float(attn.get("effective_n")[0].item())
        if attn.get("effective_n") is not None
        else None,
        "attention_top5_mass": float(attn.get("top5_mass")[0].item())
        if attn.get("top5_mass") is not None
        else None,
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

    with gr.Blocks(title="TaikoChartEstimator Inference") as demo:
        gr.Markdown("# TaikoChartEstimator - Inference")
        gr.Markdown(
            """
## How to Read Visualizations

- The model splits the chart into multiple **windows (instances)** and aggregates them using MIL (Multiple Instance Learning) for a prediction.
- `Avg attention` is the importance weight of this window for the final judgment; it is typically normalized by softmax within a single chart, so the values are usually small.
- `Top-k` is another Top-K pooling branch that selects windows that "look most like peak difficulty points"; they do not necessarily overlap perfectly with attention peaks.

Recommended combinations:
- `Token density vs attention`: Check if high-density segments are simultaneously emphasized.
- `Attention concentration`: Check if the model relies on only a few windows (closer to 1 means more concentrated).
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                tja_file = gr.File(
                    label="Upload .tja", file_types=[".tja"], type="filepath"
                )
                tja_text = gr.Textbox(label="Or paste TJA content", lines=16)

                course = gr.Dropdown(label="COURSE", choices=[], value=None)

                checkpoint = gr.Dropdown(
                    label="Checkpoint",
                    choices=checkpoints,
                    value=checkpoints[-1] if checkpoints else None,
                    allow_custom_value=True,
                )

                device = gr.Dropdown(
                    label="Device", choices=["cpu", "mps", "cuda"], value="cpu"
                )

                window_measures = gr.Textbox(
                    label="window_measures (comma-separated)", value="2,4"
                )
                hop_measures = gr.Slider(
                    label="hop_measures", minimum=1, maximum=8, value=2, step=1
                )
                max_instances = gr.Slider(
                    label="max_instances", minimum=8, maximum=256, value=64, step=1
                )

                run_btn = gr.Button("Run inference", variant="primary")

            with gr.Column(scale=2):
                summary = gr.Markdown()
                meta_json = gr.JSON(label="Details")
                attn_plot = gr.Plot(label="Attention (time-sorted)")
                density_plot = gr.Plot(label="Token density vs attention")
                heat_plot = gr.Plot(label="Branch attention heatmap")
                conc_plot = gr.Plot(label="Attention concentration")
                top_segments = gr.Markdown()
                table = gr.Dataframe(
                    headers=[
                        "instance_idx",
                        "t_start",
                        "t_end",
                        "t_mid",
                        "token_count",
                        "avg_attention",
                        "topk_selected",
                    ],
                    datatype=[
                        "number",
                        "number",
                        "number",
                        "number",
                        "number",
                        "number",
                        "number",
                    ],
                    label="Per-instance details",
                    wrap=True,
                )

        # Auto-refresh COURSE choices when input changes
        tja_file.change(
            _update_course_dropdown, inputs=[tja_file, tja_text], outputs=[course]
        )
        tja_text.change(
            _update_course_dropdown, inputs=[tja_file, tja_text], outputs=[course]
        )

        run_btn.click(
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
            ],
            outputs=[
                summary,
                meta_json,
                attn_plot,
                density_plot,
                heat_plot,
                conc_plot,
                top_segments,
                table,
            ],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
