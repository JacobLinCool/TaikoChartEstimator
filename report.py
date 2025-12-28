#!/usr/bin/env python3
"""
Batch TJA report generator using TaikoChartEstimator.

Scans a directory (recursively) for .tja files (UTF-8 or Shift-JIS encoded),
runs the trained model on each, and generates a markdown report with results.

Can also generate markdown reports from pre-computed JSONL files.

Usage:
    python report.py <tja_directory> [options]
    python report.py --input-jsonl <jsonl_file> [options]

Example:
    # Process TJA files and generate report
    python report.py ./songs --checkpoint outputs/20251228_062253/pretrained --output report.md
    python report.py ./songs --sort raw_star,title --exclude "*.backup.tja"

    # Generate report from pre-computed JSONL
    python report.py --input-jsonl report.jsonl --output new_report.md
    python report.py --input-jsonl report.jsonl --sort -raw_star --output sorted.md
"""

import argparse
import fnmatch
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import torch

# Import from the existing app.py
from app import (
    ParsedCourse,
    ParsedTJA,
    _build_instances_from_segments,
    _discover_checkpoints,
    _load_model,
    _resolve_device,
    parse_tja,
)

# Supported sort keys
SORT_KEYS = [
    "title",
    "file",
    "course",
    "level",
    "hint",
    "predicted",
    "raw_star",
    "raw_score",
    "instances",
]
REPORT_COLUMNS = [
    "file",
    "title",
    "course",
    "level",
    "hint",
    "predicted",
    "raw_star",
    "raw_score",
    "instances",
]


@dataclass
class CourseResult:
    """Result for a single course within a TJA file."""

    course_name: str
    level: Optional[int]
    difficulty_hint: Optional[str]
    predicted_difficulty: str
    raw_score: float
    raw_star: float
    n_instances: int


@dataclass
class TJAResult:
    """Result for a single TJA file."""

    file_path: str
    relative_path: str
    title: Optional[str]
    courses: list[CourseResult]
    error: Optional[str] = None


def read_tja_file(file_path: str) -> str:
    """Read a TJA file with automatic encoding detection (UTF-8 or Shift-JIS)."""
    # Try UTF-8 first (with BOM handling)
    encodings = ["utf-8-sig", "utf-8", "shift_jis", "cp932"]

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
                # Validate by checking if we can parse some basic content
                if any(
                    kw in content.upper()
                    for kw in ["TITLE:", "COURSE:", "BPM:", "#START"]
                ):
                    return content
        except (UnicodeDecodeError, UnicodeError):
            continue

    # Fallback: read with error handling
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def run_inference_for_course(
    parsed: ParsedTJA,
    course: ParsedCourse,
    model,
    device: str,
    window_measures: list[int],
    hop_measures: int,
    max_instances: int,
) -> CourseResult:
    """Run inference on a single course.

    Note: difficulty_hint is passed to the model to help predict star ratings.
    The hint maps course names (easy/normal/hard/oni/ura) to indices 0-4.
    """
    max_tokens = int(getattr(model.config, "max_seq_len", 128))

    instances, masks, counts, times, token_counts = _build_instances_from_segments(
        course.segments,
        max_tokens_per_instance=max_tokens,
        window_measures=window_measures,
        hop_measures=hop_measures,
        max_instances_per_chart=max_instances,
    )

    instances = instances.to(torch.device(device))
    masks = masks.to(torch.device(device))
    counts = counts.to(torch.device(device))

    # Use difficulty_hint for prediction - maps course difficulty name to index
    difficulty_hint = None
    if course.difficulty_hint is not None:
        mapping = {"easy": 0, "normal": 1, "hard": 2, "oni": 3, "ura": 4}
        if course.difficulty_hint in mapping:
            difficulty_hint = torch.tensor(
                [mapping[course.difficulty_hint]], device=torch.device(device)
            )

    with torch.no_grad():
        out = model.forward(
            instances,
            masks,
            counts,
            difficulty_hint=difficulty_hint,  # This hint is used for star prediction
            return_attention=True,
        )

    difficulty_names = ["easy", "normal", "hard", "oni", "ura"]
    pred_class_id = int(out.difficulty_logits.argmax(dim=-1).item())
    pred_class = difficulty_names[pred_class_id]

    return CourseResult(
        course_name=course.name,
        level=course.level,
        difficulty_hint=course.difficulty_hint,
        predicted_difficulty=pred_class,
        raw_score=float(out.raw_score.item()),
        raw_star=float(out.raw_star.item()),
        n_instances=int(counts.item()),
    )


def process_tja_file(
    file_path: str,
    base_dir: str,
    model,
    device: str,
    window_measures: list[int],
    hop_measures: int,
    max_instances: int,
    target_courses: Optional[list[str]] = None,
) -> TJAResult:
    """Process a single TJA file and return results for all courses."""
    relative_path = os.path.relpath(file_path, base_dir)

    try:
        tja_text = read_tja_file(file_path)

        # Skip charts with unsupported features
        tja_upper = tja_text.upper()
        if "#BRANCH" in tja_upper:
            return TJAResult(
                file_path=file_path,
                relative_path=relative_path,
                title=None,
                courses=[],
                error="Branching (#BRANCH) is not supported by this naive simple parser",
            )
        if "#START P2" in tja_upper:
            return TJAResult(
                file_path=file_path,
                relative_path=relative_path,
                title=None,
                courses=[],
                error="Double player (#START P2) is not supported by this naive simple parser",
            )

        parsed = parse_tja(tja_text)

        if not parsed.courses:
            return TJAResult(
                file_path=file_path,
                relative_path=relative_path,
                title=parsed.meta.get("TITLE"),
                courses=[],
                error="No courses found in TJA file",
            )

        course_results = []
        for course_name, course in parsed.courses.items():
            # Filter courses if specified
            if target_courses and course_name.lower() not in [
                c.lower() for c in target_courses
            ]:
                continue

            # Skip empty courses
            if not course.segments:
                continue

            try:
                result = run_inference_for_course(
                    parsed=parsed,
                    course=course,
                    model=model,
                    device=device,
                    window_measures=window_measures,
                    hop_measures=hop_measures,
                    max_instances=max_instances,
                )
                course_results.append(result)
            except Exception as e:
                # Create an error result for this course
                course_results.append(
                    CourseResult(
                        course_name=course_name,
                        level=course.level,
                        difficulty_hint=course.difficulty_hint,
                        predicted_difficulty="error",
                        raw_score=0.0,
                        raw_star=0.0,
                        n_instances=0,
                    )
                )

        return TJAResult(
            file_path=file_path,
            relative_path=relative_path,
            title=parsed.meta.get("TITLE"),
            courses=course_results,
        )

    except Exception as e:
        return TJAResult(
            file_path=file_path,
            relative_path=relative_path,
            title=None,
            courses=[],
            error=str(e),
        )


def find_tja_files(
    directory: str, exclude_patterns: Optional[list[str]] = None
) -> list[str]:
    """Recursively find all .tja files in a directory, with optional glob exclusions."""
    tja_files = []
    exclude_patterns = exclude_patterns or []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".tja"):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, directory)

                # Check if file matches any exclude pattern
                excluded = False
                for pattern in exclude_patterns:
                    if fnmatch.fnmatch(relative_path, pattern) or fnmatch.fnmatch(
                        file, pattern
                    ):
                        excluded = True
                        break

                if not excluded:
                    tja_files.append(full_path)

    return sorted(tja_files)


@dataclass
class FlatRow:
    """Flattened row for sorting and display."""

    file_name: str
    title: str
    course_name: str
    level: Optional[int]
    hint: Optional[str]
    predicted: str
    raw_star: float
    raw_score: float
    instances: int


def sort_rows(rows: list[FlatRow], sort_keys: list[str]) -> list[FlatRow]:
    """Sort rows by the given keys (in order). Prefix with '-' for descending."""
    if not sort_keys:
        return rows

    def get_sort_key(row: FlatRow):
        keys = []
        for key in sort_keys:
            descending = key.startswith("-")
            key_name = key.lstrip("-")

            if key_name == "title":
                val = row.title.lower()
            elif key_name == "file":
                val = row.file_name.lower()
            elif key_name == "course":
                val = row.course_name.lower()
            elif key_name == "level":
                val = row.level if row.level is not None else -1
            elif key_name == "hint":
                val = row.hint or ""
            elif key_name == "predicted":
                val = row.predicted
            elif key_name == "raw_star":
                val = row.raw_star
            elif key_name == "raw_score":
                val = row.raw_score
            elif key_name == "instances":
                val = row.instances
            else:
                val = 0

            # For descending, negate numeric values or reverse strings
            if descending:
                if isinstance(val, (int, float)):
                    val = -val
                elif isinstance(val, str):
                    # Reverse string comparison by inverting character codes
                    val = [-ord(c) for c in val]

            keys.append(val)
        return tuple(keys)

    return sorted(rows, key=get_sort_key)


def generate_markdown_report(
    results: list[TJAResult],
    checkpoint_path: str,
    base_dir: str,
    sort_keys: Optional[list[str]] = None,
    hide_columns: Optional[list[str]] = None,
) -> str:
    """Generate a markdown report from the results."""
    lines = []

    # Header
    lines.append("# TaikoChartEstimator Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Source Directory:** `{base_dir}`")
    lines.append(f"**Checkpoint:** `{checkpoint_path}`")
    lines.append("")

    # Summary statistics
    total_files = len(results)
    successful_files = sum(1 for r in results if r.error is None and r.courses)
    failed_files = sum(1 for r in results if r.error is not None)
    empty_files = sum(1 for r in results if r.error is None and not r.courses)
    total_courses = sum(len(r.courses) for r in results)

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total TJA Files:** {total_files}")
    lines.append(f"- **Successfully Processed:** {successful_files}")
    lines.append(f"- **Failed:** {failed_files}")
    lines.append(f"- **Empty (no courses):** {empty_files}")
    lines.append(f"- **Total Courses Analyzed:** {total_courses}")
    lines.append("")

    # Build flat rows for sorting
    rows: list[FlatRow] = []
    for result in results:
        if result.error:
            continue
        for course in result.courses:
            title = result.title or "-"
            if len(title) > 30:
                title = title[:27] + "..."
            file_name = os.path.basename(result.relative_path)
            if len(file_name) > 25:
                file_name = file_name[:22] + "..."
            rows.append(
                FlatRow(
                    file_name=file_name,
                    title=title,
                    course_name=course.course_name,
                    level=course.level,
                    hint=course.difficulty_hint,
                    predicted=course.predicted_difficulty,
                    raw_star=course.raw_star,
                    raw_score=course.raw_score,
                    instances=course.n_instances,
                )
            )

    # Sort rows
    if sort_keys:
        rows = sort_rows(rows, sort_keys)

    # Results table
    if rows:
        lines.append("## Results")
        lines.append("")

        # Build columns dynamically based on hide_columns
        hidden = set(hide_columns or [])
        col_defs = [
            ("file", "File", lambda r: r.file_name),
            ("title", "Title", lambda r: r.title),
            ("course", "Course", lambda r: r.course_name),
            ("level", "Level", lambda r: str(r.level) if r.level is not None else "-"),
            ("hint", "Hint", lambda r: r.hint or "-"),
            ("predicted", "Predicted", lambda r: r.predicted),
            ("raw_star", "Raw ★", lambda r: f"{r.raw_star:.2f}"),
            ("raw_score", "Raw Score", lambda r: f"{r.raw_score:.4f}"),
            ("instances", "Instances", lambda r: str(r.instances)),
        ]
        visible_cols = [
            (key, header, getter)
            for key, header, getter in col_defs
            if key not in hidden
        ]

        if visible_cols:
            header_row = "| " + " | ".join(h for _, h, _ in visible_cols) + " |"
            sep_row = "|" + "|".join("------" for _ in visible_cols) + "|"
            lines.append(header_row)
            lines.append(sep_row)

            for row in rows:
                row_values = [getter(row) for _, _, getter in visible_cols]
                lines.append("| " + " | ".join(row_values) + " |")
        lines.append("")

    # Error summary
    errors = [r for r in results if r.error]
    if errors:
        lines.append("## Errors")
        lines.append("")
        for result in errors:
            lines.append(f"- `{result.relative_path}`: {result.error}")
        lines.append("")

    return "\n".join(lines)


def generate_markdown_from_jsonl(
    jsonl_path: str,
    sort_keys: Optional[list[str]] = None,
    hide_columns: Optional[list[str]] = None,
) -> str:
    """Generate a markdown report from a pre-computed JSONL file."""
    lines = []

    # Read JSONL file
    rows: list[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # Header
    lines.append("# TaikoChartEstimator Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Source JSONL:** `{jsonl_path}`")
    lines.append("")

    # Summary
    unique_files = len(set(r.get("file", "") for r in rows))
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total Entries:** {len(rows)}")
    lines.append(f"- **Unique Files:** {unique_files}")
    lines.append("")

    # Build flat rows for sorting
    flat_rows: list[FlatRow] = []
    for r in rows:
        title = r.get("title", "-")
        if len(title) > 30:
            title = title[:27] + "..."
        file_name = os.path.basename(r.get("file", "-"))
        if len(file_name) > 25:
            file_name = file_name[:22] + "..."
        flat_rows.append(
            FlatRow(
                file_name=file_name,
                title=title,
                course_name=r.get("course", "-"),
                level=r.get("level"),
                hint=r.get("hint"),
                predicted=r.get("predicted", "-"),
                raw_star=r.get("raw_star", 0.0),
                raw_score=r.get("raw_score", 0.0),
                instances=r.get("instances", 0),
            )
        )

    # Sort rows
    if sort_keys:
        flat_rows = sort_rows(flat_rows, sort_keys)

    # Results table
    if flat_rows:
        lines.append("## Results")
        lines.append("")

        # Build columns dynamically based on hide_columns
        hidden = set(hide_columns or [])
        col_defs = [
            ("file", "File", lambda r: r.file_name),
            ("title", "Title", lambda r: r.title),
            ("course", "Course", lambda r: r.course_name),
            ("level", "Level", lambda r: str(r.level) if r.level is not None else "-"),
            ("hint", "Hint", lambda r: r.hint or "-"),
            ("predicted", "Predicted", lambda r: r.predicted),
            ("raw_star", "Raw ★", lambda r: f"{r.raw_star:.2f}"),
            ("raw_score", "Raw Score", lambda r: f"{r.raw_score:.4f}"),
            ("instances", "Instances", lambda r: str(r.instances)),
        ]
        visible_cols = [
            (key, header, getter)
            for key, header, getter in col_defs
            if key not in hidden
        ]

        if visible_cols:
            header_row = "| " + " | ".join(h for _, h, _ in visible_cols) + " |"
            sep_row = "|" + "|".join("------" for _ in visible_cols) + "|"
            lines.append(header_row)
            lines.append(sep_row)

            for row in flat_rows:
                row_values = [getter(row) for _, _, getter in visible_cols]
                lines.append("| " + " | ".join(row_values) + " |")
        lines.append("")

    return "\n".join(lines)


def generate_jsonl_report(
    results: list[TJAResult],
    sort_keys: Optional[list[str]] = None,
) -> str:
    """Generate a JSONL report from the results (one JSON object per line)."""
    # Build flat rows
    rows: list[dict] = []
    for result in results:
        if result.error:
            continue
        for course in result.courses:
            rows.append(
                {
                    "file": result.relative_path,
                    "title": result.title or "",
                    "course": course.course_name,
                    "level": course.level,
                    "hint": course.difficulty_hint,
                    "predicted": course.predicted_difficulty,
                    "raw_star": round(course.raw_star, 4),
                    "raw_score": round(course.raw_score, 4),
                    "instances": course.n_instances,
                }
            )

    # Sort if needed (reuse same logic)
    if sort_keys:
        flat_rows = [
            FlatRow(
                file_name=r["file"],
                title=r["title"],
                course_name=r["course"],
                level=r["level"],
                hint=r["hint"],
                predicted=r["predicted"],
                raw_star=r["raw_star"],
                raw_score=r["raw_score"],
                instances=r["instances"],
            )
            for r in rows
        ]
        sorted_flat = sort_rows(flat_rows, sort_keys)
        rows = [
            {
                "file": fr.file_name,
                "title": fr.title,
                "course": fr.course_name,
                "level": fr.level,
                "hint": fr.hint,
                "predicted": fr.predicted,
                "raw_star": fr.raw_star,
                "raw_score": fr.raw_score,
                "instances": fr.instances,
            }
            for fr in sorted_flat
        ]

    return "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a markdown report for TJA files using TaikoChartEstimator"
    )
    parser.add_argument(
        "directory",
        type=str,
        nargs="?",
        default=None,
        help="Directory containing .tja files (searched recursively). Not required if --input-jsonl is used.",
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        default=None,
        help="Path to pre-computed JSONL file. If provided, generates markdown from JSONL instead of processing TJA files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: latest in outputs/)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="report.md",
        help="Output markdown file path (default: report.md)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run inference on (default: cpu)",
    )
    parser.add_argument(
        "--window-measures",
        type=str,
        default="2,4",
        help="Comma-separated window measure sizes (default: 2,4)",
    )
    parser.add_argument(
        "--hop-measures",
        type=int,
        default=2,
        help="Hop size in measures (default: 2)",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=64,
        help="Maximum instances per chart (default: 64)",
    )
    parser.add_argument(
        "--courses",
        type=str,
        default=None,
        help="Comma-separated course names to process (default: all)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        action="append",
        default=[],
        help="Glob pattern to exclude TJA files (can be specified multiple times)",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default=None,
        help=f"Comma-separated sort keys (prefix with '-' for descending). Available: {', '.join(SORT_KEYS)}",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Deduplicate TJA files by title (keeps first occurrence)",
    )
    parser.add_argument(
        "--hide-columns",
        type=str,
        default=None,
        help=f"Comma-separated columns to hide in the report. Available: {', '.join(REPORT_COLUMNS)}",
    )

    args = parser.parse_args()

    # Handle JSONL input mode
    if args.input_jsonl:
        if not os.path.isfile(args.input_jsonl):
            print(f"Error: '{args.input_jsonl}' is not a valid file", file=sys.stderr)
            sys.exit(1)

        # Parse sort keys
        sort_keys = None
        if args.sort:
            sort_keys = [k.strip() for k in args.sort.split(",") if k.strip()]
            for key in sort_keys:
                key_name = key.lstrip("-")
                if key_name not in SORT_KEYS:
                    print(
                        f"Error: Invalid sort key '{key_name}'. Available: {', '.join(SORT_KEYS)}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

        # Parse hide columns
        hide_columns = None
        if args.hide_columns:
            hide_columns = [
                c.strip() for c in args.hide_columns.split(",") if c.strip()
            ]
            for col in hide_columns:
                if col not in REPORT_COLUMNS:
                    print(
                        f"Error: Invalid column '{col}'. Available: {', '.join(REPORT_COLUMNS)}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

        print(f"Generating report from JSONL: {args.input_jsonl}")
        report = generate_markdown_from_jsonl(
            args.input_jsonl, sort_keys=sort_keys, hide_columns=hide_columns
        )

        output_path = args.output
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"Markdown report saved to: {output_path}")
        return

    # Validate directory for TJA processing mode
    if not args.directory:
        print(
            "Error: directory is required when not using --input-jsonl", file=sys.stderr
        )
        sys.exit(1)

    if not os.path.isdir(args.directory):
        print(f"Error: '{args.directory}' is not a valid directory", file=sys.stderr)
        sys.exit(1)

    # Find checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoints = _discover_checkpoints()
        if not checkpoints:
            print(
                "Error: No checkpoints found. Please specify --checkpoint",
                file=sys.stderr,
            )
            sys.exit(1)
        checkpoint_path = checkpoints[-1]
        print(f"Using checkpoint: {checkpoint_path}")

    # Parse window measures
    try:
        window_measures = [
            int(x.strip()) for x in args.window_measures.split(",") if x.strip()
        ]
    except ValueError:
        print(
            "Error: --window-measures must be comma-separated integers", file=sys.stderr
        )
        sys.exit(1)

    # Parse target courses
    target_courses = None
    if args.courses:
        target_courses = [c.strip() for c in args.courses.split(",") if c.strip()]

    # Parse sort keys
    sort_keys = None
    if args.sort:
        sort_keys = [k.strip() for k in args.sort.split(",") if k.strip()]
        # Validate sort keys
        for key in sort_keys:
            key_name = key.lstrip("-")
            if key_name not in SORT_KEYS:
                print(
                    f"Error: Invalid sort key '{key_name}'. Available: {', '.join(SORT_KEYS)}",
                    file=sys.stderr,
                )
                sys.exit(1)

    # Parse hide columns
    hide_columns = None
    if args.hide_columns:
        hide_columns = [c.strip() for c in args.hide_columns.split(",") if c.strip()]
        for col in hide_columns:
            if col not in REPORT_COLUMNS:
                print(
                    f"Error: Invalid column '{col}'. Available: {', '.join(REPORT_COLUMNS)}",
                    file=sys.stderr,
                )
                sys.exit(1)

    # Find TJA files
    print(f"Scanning for TJA files in: {args.directory}")
    if args.exclude:
        print(f"Excluding patterns: {args.exclude}")
    tja_files = find_tja_files(args.directory, exclude_patterns=args.exclude)

    if not tja_files:
        print("No .tja files found", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(tja_files)} TJA file(s)")

    # Load model
    print(f"Loading model from: {checkpoint_path}")
    device = _resolve_device(args.device)
    model = _load_model(checkpoint_path, device)
    print(f"Model loaded on device: {device}")

    # Process files
    results = []
    seen_titles: set[str] = set()
    skipped_duplicates = 0

    for i, file_path in enumerate(tja_files, 1):
        relative = os.path.relpath(file_path, args.directory)
        print(f"[{i}/{len(tja_files)}] Processing: {relative}")

        result = process_tja_file(
            file_path=file_path,
            base_dir=args.directory,
            model=model,
            device=device,
            window_measures=window_measures,
            hop_measures=args.hop_measures,
            max_instances=args.max_instances,
            target_courses=target_courses,
        )

        # Deduplication by title
        if args.dedup and result.title:
            if result.title in seen_titles:
                print(f"  ⏭️ Skipped (duplicate title: {result.title})")
                skipped_duplicates += 1
                continue
            seen_titles.add(result.title)

        results.append(result)

        if result.error:
            print(f"  ⚠️ Error: {result.error}")
        else:
            for course in result.courses:
                print(
                    f"  ✓ {course.course_name}: {course.predicted_difficulty} ({course.raw_star:.2f}★)"
                )

    if args.dedup and skipped_duplicates > 0:
        print(f"\nSkipped {skipped_duplicates} duplicate(s) by title")

    # Generate reports
    print(f"\nGenerating reports...")
    report = generate_markdown_report(
        results,
        checkpoint_path,
        args.directory,
        sort_keys=sort_keys,
        hide_columns=hide_columns,
    )
    jsonl_report = generate_jsonl_report(results, sort_keys=sort_keys)

    # Write markdown output
    output_path = args.output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Markdown report saved to: {output_path}")

    # Write JSONL output
    jsonl_path = (
        output_path.rsplit(".", 1)[0] + ".jsonl"
        if "." in output_path
        else output_path + ".jsonl"
    )
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(jsonl_report)
    print(f"JSONL report saved to: {jsonl_path}")

    # Print summary
    successful = sum(1 for r in results if r.error is None and r.courses)
    failed = sum(1 for r in results if r.error is not None)
    print(
        f"\nSummary: {successful} successful, {failed} failed out of {len(results)} files"
    )


if __name__ == "__main__":
    main()
