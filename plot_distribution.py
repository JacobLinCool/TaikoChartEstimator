#!/usr/bin/env python3
"""
Script to generate distribution plots of Raw Score from report.jsonl
Supports overall distribution and breakdown by difficulty level
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Difficulty colors - vibrant and distinguishable
DIFFICULTY_COLORS = {
    "easy": "#4ade80",  # Green
    "normal": "#facc15",  # Yellow
    "hard": "#f87171",  # Red
    "oni": "#a855f7",  # Purple
    "ura": "#ec4899",  # Pink (for Edit/Ura)
}

DIFFICULTY_ORDER = ["easy", "normal", "hard", "oni", "ura"]


def load_data(
    jsonl_path: str,
) -> tuple[list[float], dict[str, list[float]], dict[str, dict[int, list[float]]]]:
    """Load raw_score values from JSONL file, grouped by difficulty and level."""
    all_scores = []
    scores_by_difficulty = defaultdict(list)
    scores_by_difficulty_level = defaultdict(lambda: defaultdict(list))

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if "raw_score" in data:
                    score = data["raw_score"]
                    level = data.get("level", 0)
                    all_scores.append(score)

                    # Get difficulty from 'hint' or 'predicted' field
                    difficulty = data.get(
                        "hint", data.get("predicted", "unknown")
                    ).lower()

                    # Normalize difficulty names
                    if difficulty in ["edit", "ura", "4"]:
                        difficulty = "ura"
                    elif difficulty in ["oni", "3"]:
                        difficulty = "oni"
                    elif difficulty in ["hard", "2"]:
                        difficulty = "hard"
                    elif difficulty in ["normal", "1"]:
                        difficulty = "normal"
                    elif difficulty in ["easy", "0"]:
                        difficulty = "easy"

                    scores_by_difficulty[difficulty].append(score)
                    scores_by_difficulty_level[difficulty][level].append(score)

    return (
        all_scores,
        dict(scores_by_difficulty),
        {k: dict(v) for k, v in scores_by_difficulty_level.items()},
    )


def plot_distribution(
    scores: list[float], output_path: str, title: str = "Distribution of Raw Score"
):
    """Generate a distribution plot for raw scores."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 7))

    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    mean_score = np.mean(scores)
    median_score = np.median(scores)
    std_score = np.std(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)

    n_bins = 50
    counts, bins, patches = ax.hist(
        scores, bins=n_bins, edgecolor="white", linewidth=0.5, alpha=0.8
    )

    cm = plt.colormaps["plasma"]
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    norm = plt.Normalize(min_score, max_score)
    for c, p in zip(bin_centers, patches):
        plt.setp(p, "facecolor", cm(norm(c)))

    kde = stats.gaussian_kde(scores)
    x_range = np.linspace(min_score - 0.5, max_score + 0.5, 200)
    kde_values = kde(x_range)
    kde_scaled = kde_values * len(scores) * (bins[1] - bins[0])
    ax.plot(x_range, kde_scaled, color="#00fff5", linewidth=2.5, label="Density Curve")

    ax.axvline(
        mean_score,
        color="#ff6b6b",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_score:.2f}",
    )
    ax.axvline(
        median_score,
        color="#4ecdc4",
        linestyle="-.",
        linewidth=2,
        label=f"Median: {median_score:.2f}",
    )
    ax.axvspan(
        mean_score - std_score,
        mean_score + std_score,
        alpha=0.15,
        color="#ff6b6b",
        label=f"Std Dev: ±{std_score:.2f}",
    )

    ax.set_xlabel("Raw Score", fontsize=14, fontweight="bold", color="white")
    ax.set_ylabel("Frequency", fontsize=14, fontweight="bold", color="white")
    ax.set_title(title, fontsize=18, fontweight="bold", color="white", pad=20)
    ax.grid(True, alpha=0.2, linestyle="--")

    legend = ax.legend(
        loc="upper right", fontsize=10, facecolor="#1a1a2e", edgecolor="#4a4a6a"
    )
    for text in legend.get_texts():
        text.set_color("white")

    stats_text = f"N = {len(scores):,}\nMin = {min_score:.2f}\nMax = {max_score:.2f}\nRange = {max_score - min_score:.2f}"
    props = dict(
        boxstyle="round,pad=0.5", facecolor="#16213e", edgecolor="#4a4a6a", alpha=0.9
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
        color="white",
        fontfamily="monospace",
    )

    for spine in ax.spines.values():
        spine.set_color("#4a4a6a")
        spine.set_linewidth(1.5)

    ax.tick_params(colors="white", labelsize=11)

    plt.tight_layout()
    plt.savefig(
        output_path, dpi=150, facecolor="#1a1a2e", edgecolor="none", bbox_inches="tight"
    )
    plt.close()
    print(f"Plot saved to: {output_path}")

    return {
        "count": len(scores),
        "mean": mean_score,
        "median": median_score,
        "std": std_score,
        "min": min_score,
        "max": max_score,
    }


def plot_by_difficulty_subplots(scores_by_difficulty: dict, output_path: str):
    """Generate subplots for each difficulty level."""
    plt.style.use("dark_background")

    # Filter to only include difficulties with data
    available_difficulties = [
        d
        for d in DIFFICULTY_ORDER
        if d in scores_by_difficulty and len(scores_by_difficulty[d]) > 0
    ]
    n_difficulties = len(available_difficulties)

    if n_difficulties == 0:
        print("No difficulty data available")
        return

    # Create subplot grid
    cols = min(3, n_difficulties)
    rows = (n_difficulties + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    fig.patch.set_facecolor("#1a1a2e")

    # Flatten axes for easy iteration
    if n_difficulties == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, difficulty in enumerate(available_difficulties):
        ax = axes[idx]
        ax.set_facecolor("#16213e")

        scores = scores_by_difficulty[difficulty]
        color = DIFFICULTY_COLORS.get(difficulty, "#ffffff")

        # Histogram
        n_bins = 30
        counts, bins, patches = ax.hist(
            scores,
            bins=n_bins,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.7,
        )

        # KDE curve
        if len(scores) > 1:
            kde = stats.gaussian_kde(scores)
            x_range = np.linspace(min(scores) - 0.5, max(scores) + 0.5, 100)
            kde_values = kde(x_range)
            kde_scaled = kde_values * len(scores) * (bins[1] - bins[0])
            ax.plot(x_range, kde_scaled, color="white", linewidth=2, alpha=0.8)

        # Statistics
        mean_val = np.mean(scores)
        median_val = np.median(scores)

        ax.axvline(mean_val, color="#ff6b6b", linestyle="--", linewidth=2, alpha=0.8)
        ax.axvline(median_val, color="#4ecdc4", linestyle="-.", linewidth=2, alpha=0.8)

        # Labels
        ax.set_xlabel("Raw Score", fontsize=11, color="white")
        ax.set_ylabel("Frequency", fontsize=11, color="white")
        ax.set_title(
            f"{difficulty.upper()}", fontsize=14, fontweight="bold", color=color, pad=10
        )
        ax.grid(True, alpha=0.2, linestyle="--")

        # Stats text
        stats_text = f"N={len(scores):,}\nμ={mean_val:.2f}\nσ={np.std(scores):.2f}"
        props = dict(
            boxstyle="round,pad=0.3", facecolor="#16213e", edgecolor=color, alpha=0.9
        )
        ax.text(
            0.97,
            0.97,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=props,
            color="white",
            fontfamily="monospace",
        )

        for spine in ax.spines.values():
            spine.set_color("#4a4a6a")
        ax.tick_params(colors="white", labelsize=9)

    # Hide extra axes
    for idx in range(n_difficulties, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Raw Score Distribution by Difficulty",
        fontsize=20,
        fontweight="bold",
        color="white",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        output_path, dpi=150, facecolor="#1a1a2e", edgecolor="none", bbox_inches="tight"
    )
    plt.close()
    print(f"Subplot plot saved to: {output_path}")


def plot_by_difficulty_overlay(scores_by_difficulty: dict, output_path: str):
    """Generate an overlay plot comparing all difficulty levels."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 8))

    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    # Get global min/max for consistent x-axis
    all_scores = []
    for scores in scores_by_difficulty.values():
        all_scores.extend(scores)
    global_min = min(all_scores)
    global_max = max(all_scores)
    x_range = np.linspace(global_min - 0.3, global_max + 0.3, 200)

    legend_handles = []

    for difficulty in DIFFICULTY_ORDER:
        if (
            difficulty not in scores_by_difficulty
            or len(scores_by_difficulty[difficulty]) < 2
        ):
            continue

        scores = scores_by_difficulty[difficulty]
        color = DIFFICULTY_COLORS.get(difficulty, "#ffffff")

        # KDE curve only (for overlay clarity)
        kde = stats.gaussian_kde(scores)
        kde_values = kde(x_range)

        (line,) = ax.plot(
            x_range,
            kde_values,
            color=color,
            linewidth=3,
            alpha=0.9,
            label=f"{difficulty.upper()} (n={len(scores):,}, μ={np.mean(scores):.2f})",
        )
        legend_handles.append(line)

        # Fill under curve
        ax.fill_between(x_range, kde_values, alpha=0.15, color=color)

        # Add mean marker
        mean_val = np.mean(scores)
        kde_at_mean = kde(mean_val)[0]
        ax.scatter(
            [mean_val],
            [kde_at_mean],
            color=color,
            s=100,
            zorder=5,
            edgecolor="white",
            linewidth=2,
        )

    ax.set_xlabel("Raw Score", fontsize=14, fontweight="bold", color="white")
    ax.set_ylabel("Density", fontsize=14, fontweight="bold", color="white")
    ax.set_title(
        "Raw Score Distribution Comparison by Difficulty",
        fontsize=18,
        fontweight="bold",
        color="white",
        pad=20,
    )
    ax.grid(True, alpha=0.2, linestyle="--")

    legend = ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=11,
        facecolor="#1a1a2e",
        edgecolor="#4a4a6a",
        title="Difficulty Level",
    )
    legend.get_title().set_color("white")
    legend.get_title().set_fontweight("bold")
    for text in legend.get_texts():
        text.set_color("white")

    for spine in ax.spines.values():
        spine.set_color("#4a4a6a")
        spine.set_linewidth(1.5)

    ax.tick_params(colors="white", labelsize=11)

    plt.tight_layout()
    plt.savefig(
        output_path, dpi=150, facecolor="#1a1a2e", edgecolor="none", bbox_inches="tight"
    )
    plt.close()
    print(f"Overlay plot saved to: {output_path}")


def plot_violin_by_difficulty(scores_by_difficulty: dict, output_path: str):
    """Generate a violin plot comparing difficulty levels."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 7))

    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    # Prepare data
    available_difficulties = [
        d
        for d in DIFFICULTY_ORDER
        if d in scores_by_difficulty and len(scores_by_difficulty[d]) > 0
    ]
    data = [scores_by_difficulty[d] for d in available_difficulties]
    colors = [DIFFICULTY_COLORS.get(d, "#ffffff") for d in available_difficulties]

    # Create violin plot
    parts = ax.violinplot(
        data, positions=range(len(data)), showmeans=True, showmedians=True
    )

    # Color the violins
    for idx, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[idx])
        pc.set_edgecolor("white")
        pc.set_alpha(0.7)

    # Style the lines
    for partname in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
        if partname in parts:
            parts[partname].set_edgecolor("white")
            parts[partname].set_linewidth(1.5)

    # Add box plot overlay
    bp = ax.boxplot(
        data,
        positions=range(len(data)),
        widths=0.15,
        patch_artist=True,
        showfliers=False,
        showmeans=False,
    )

    for idx, box in enumerate(bp["boxes"]):
        box.set_facecolor(colors[idx])
        box.set_alpha(0.9)
        box.set_edgecolor("white")

    for element in ["whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set_color("white")
            item.set_linewidth(1.5)

    # Labels
    ax.set_xticks(range(len(available_difficulties)))
    ax.set_xticklabels(
        [d.upper() for d in available_difficulties], fontsize=12, fontweight="bold"
    )

    # Color x-tick labels
    for idx, label in enumerate(ax.get_xticklabels()):
        label.set_color(colors[idx])

    ax.set_xlabel("Difficulty", fontsize=14, fontweight="bold", color="white")
    ax.set_ylabel("Raw Score", fontsize=14, fontweight="bold", color="white")
    ax.set_title(
        "Raw Score Distribution by Difficulty (Violin Plot)",
        fontsize=18,
        fontweight="bold",
        color="white",
        pad=20,
    )
    ax.grid(True, alpha=0.2, linestyle="--", axis="y")

    # Add count annotations
    for idx, d in enumerate(available_difficulties):
        count = len(scores_by_difficulty[d])
        mean = np.mean(scores_by_difficulty[d])
        ax.annotate(
            f"n={count:,}\nμ={mean:.2f}",
            xy=(idx, min(scores_by_difficulty[d]) - 0.3),
            ha="center",
            va="top",
            fontsize=9,
            color="white",
            fontfamily="monospace",
        )

    for spine in ax.spines.values():
        spine.set_color("#4a4a6a")
        spine.set_linewidth(1.5)

    ax.tick_params(colors="white", labelsize=11)

    plt.tight_layout()
    plt.savefig(
        output_path, dpi=150, facecolor="#1a1a2e", edgecolor="none", bbox_inches="tight"
    )
    plt.close()
    print(f"Violin plot saved to: {output_path}")


def plot_violin_per_difficulty_by_level(
    scores_by_difficulty_level: dict, output_dir: str
):
    """Generate a violin plot per difficulty, showing distribution by star level."""

    for difficulty in DIFFICULTY_ORDER:
        if difficulty not in scores_by_difficulty_level:
            continue

        level_data = scores_by_difficulty_level[difficulty]
        if not level_data:
            continue

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(14, 8))

        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        color = DIFFICULTY_COLORS.get(difficulty, "#ffffff")

        # Get available levels and sort them
        available_levels = sorted(level_data.keys())

        # Prepare data for violin plot
        data = []
        positions = []
        level_labels = []

        for level in available_levels:
            scores = level_data[level]
            if len(scores) >= 2:  # Need at least 2 points for violin
                data.append(scores)
                positions.append(level)
                level_labels.append(str(level))

        if not data:
            plt.close()
            continue

        # Create violin plot
        parts = ax.violinplot(
            data, positions=positions, showmeans=True, showmedians=True, widths=0.8
        )

        # Color the violins with gradient based on level
        cm = plt.colormaps["viridis"]
        level_norm = plt.Normalize(min(positions), max(positions))

        for idx, pc in enumerate(parts["bodies"]):
            level_color = cm(level_norm(positions[idx]))
            pc.set_facecolor(level_color)
            pc.set_edgecolor("white")
            pc.set_alpha(0.7)
            pc.set_linewidth(1)

        # Style the lines
        for partname in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
            if partname in parts:
                parts[partname].set_edgecolor("white")
                parts[partname].set_linewidth(1.5)

        # Add box plot overlay
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.2,
            patch_artist=True,
            showfliers=False,
            showmeans=False,
        )

        for idx, box in enumerate(bp["boxes"]):
            level_color = cm(level_norm(positions[idx]))
            box.set_facecolor(level_color)
            box.set_alpha(0.9)
            box.set_edgecolor("white")

        for element in ["whiskers", "caps", "medians"]:
            for item in bp[element]:
                item.set_color("white")
                item.set_linewidth(1.5)

        # Add scatter points for individual data points
        for idx, (level, scores) in enumerate(zip(positions, data)):
            # Add jitter to x positions
            jitter = np.random.uniform(-0.15, 0.15, len(scores))
            ax.scatter(
                [level + j for j in jitter],
                scores,
                alpha=0.3,
                s=10,
                color="white",
                edgecolors="none",
            )

        # Add count and mean annotations at top
        y_max = max(max(d) for d in data)
        for idx, level in enumerate(positions):
            scores = data[idx]
            count = len(scores)
            mean = np.mean(scores)
            ax.annotate(
                f"n={count}\nμ={mean:.2f}",
                xy=(level, y_max + 0.3),
                ha="center",
                va="bottom",
                fontsize=8,
                color="white",
                fontfamily="monospace",
                alpha=0.8,
            )

        # Labels and title
        ax.set_xticks(positions)
        ax.set_xticklabels([f"★{l}" for l in positions], fontsize=11, fontweight="bold")

        ax.set_xlabel("Star Level", fontsize=14, fontweight="bold", color="white")
        ax.set_ylabel("Raw Score", fontsize=14, fontweight="bold", color="white")
        ax.set_title(
            f"Raw Score Distribution by Level - {difficulty.upper()}",
            fontsize=18,
            fontweight="bold",
            color=color,
            pad=20,
        )
        ax.grid(True, alpha=0.2, linestyle="--", axis="y")

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=level_norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30)
        cbar.set_label("Star Level", fontsize=12, color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        cbar.outline.set_edgecolor("white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

        # Add summary stats box
        all_scores = [s for d in data for s in d]
        stats_text = (
            f"Total: {len(all_scores):,}\n"
            f"Mean: {np.mean(all_scores):.2f}\n"
            f"Std: {np.std(all_scores):.2f}\n"
            f"Levels: {len(positions)}"
        )
        props = dict(
            boxstyle="round,pad=0.5", facecolor="#16213e", edgecolor=color, alpha=0.9
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
            color="white",
            fontfamily="monospace",
        )

        for spine in ax.spines.values():
            spine.set_color("#4a4a6a")
            spine.set_linewidth(1.5)

        ax.tick_params(colors="white", labelsize=11)

        # Adjust y-axis to make room for annotations
        ax.set_ylim(bottom=ax.get_ylim()[0], top=y_max + 1.0)

        plt.tight_layout()
        output_path = Path(output_dir) / f"raw_score_{difficulty}_by_level.png"
        plt.savefig(
            output_path,
            dpi=150,
            facecolor="#1a1a2e",
            edgecolor="none",
            bbox_inches="tight",
        )
        plt.close()
        print(f"  Saved: {output_path}")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    jsonl_path = script_dir / "report.jsonl"

    print("Loading data from report.jsonl...")
    all_scores, scores_by_difficulty, scores_by_difficulty_level = load_data(jsonl_path)
    print(f"Loaded {len(all_scores)} total raw_score values")
    print(f"Difficulties found: {list(scores_by_difficulty.keys())}")
    for d, scores in sorted(scores_by_difficulty.items()):
        print(f"  {d}: {len(scores)} entries")

    # Print level breakdown
    print("\nLevel breakdown per difficulty:")
    for difficulty in DIFFICULTY_ORDER:
        if difficulty in scores_by_difficulty_level:
            levels = scores_by_difficulty_level[difficulty]
            level_counts = {k: len(v) for k, v in sorted(levels.items())}
            print(f"  {difficulty.upper()}: {level_counts}")

    # Generate overall distribution
    print("\n1. Generating overall distribution plot...")
    output_path = script_dir / "raw_score_distribution.png"
    stats_result = plot_distribution(all_scores, str(output_path))

    print("\n=== Overall Statistics ===")
    print(f"  Count:  {stats_result['count']:,}")
    print(f"  Mean:   {stats_result['mean']:.4f}")
    print(f"  Median: {stats_result['median']:.4f}")
    print(f"  Std:    {stats_result['std']:.4f}")
    print(f"  Min:    {stats_result['min']:.4f}")
    print(f"  Max:    {stats_result['max']:.4f}")

    # Generate by-difficulty subplots
    print("\n2. Generating difficulty subplots...")
    output_path_subplots = script_dir / "raw_score_by_difficulty.png"
    plot_by_difficulty_subplots(scores_by_difficulty, str(output_path_subplots))

    # Generate overlay comparison
    print("\n3. Generating difficulty overlay comparison...")
    output_path_overlay = script_dir / "raw_score_difficulty_overlay.png"
    plot_by_difficulty_overlay(scores_by_difficulty, str(output_path_overlay))

    # Generate violin plot
    print("\n4. Generating violin plot comparison...")
    output_path_violin = script_dir / "raw_score_difficulty_violin.png"
    plot_violin_by_difficulty(scores_by_difficulty, str(output_path_violin))

    # Generate per-difficulty violin plots by level
    print("\n5. Generating per-difficulty violin plots by star level...")
    plot_violin_per_difficulty_by_level(scores_by_difficulty_level, str(script_dir))

    # Print statistics by difficulty
    print("\n=== Statistics by Difficulty ===")
    for difficulty in DIFFICULTY_ORDER:
        if (
            difficulty in scores_by_difficulty
            and len(scores_by_difficulty[difficulty]) > 0
        ):
            scores = scores_by_difficulty[difficulty]
            print(f"\n  {difficulty.upper()}:")
            print(f"    Count:  {len(scores):,}")
            print(f"    Mean:   {np.mean(scores):.4f}")
            print(f"    Median: {np.median(scores):.4f}")
            print(f"    Std:    {np.std(scores):.4f}")
            print(f"    Min:    {np.min(scores):.4f}")
            print(f"    Max:    {np.max(scores):.4f}")
