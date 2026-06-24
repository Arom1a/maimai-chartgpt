"""CLI entry point for converting processed.json to simai maidata.txt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.simai_converter import SimaiConverter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert processed.json (ChartGPT format) to simai maidata.txt",
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to processed.json",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output path for maidata.txt (default: same directory as input)",
    )
    parser.add_argument(
        "--difficulty", "-d", type=int, default=None,
        help="Chart difficulty index (0-based, default: 0). "
             "0=easy, 1=basic, 2=advanced, 3=expert, 4=master, 5=remaster",
    )
    parser.add_argument(
        "--level-num", type=int, default=None,
        help="Simai level number for &inote_N (overrides auto-detection)",
    )
    parser.add_argument(
        "--denominator", type=int, default=1,
        help="Base time signature denominator (default: 1)",
    )
    args = parser.parse_args()

    # ── Load input ────────────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)

    # ── Select chart ──────────────────────────────────────────────────────
    charts = data.get("charts", [])
    if not charts:
        print("Error: no charts found in input", file=sys.stderr)
        sys.exit(1)

    diff_idx = args.difficulty or 0
    if diff_idx >= len(charts):
        print(
            f"Error: difficulty index {diff_idx} out of range "
            f"(input has {len(charts)} charts)",
            file=sys.stderr,
        )
        sys.exit(1)

    chart = charts[diff_idx]

    # ── Build converter ───────────────────────────────────────────────────
    converter = SimaiConverter(
        title=data.get("title", "Untitled"),
        cabinet=data.get("cabinet", "DX"),
        version=data.get("version", ""),
        bpm10_list=chart["bpm10_list"],
        constant=tuple(chart["constant"]),
        designer=chart.get("designer", ""),
        notes=chart.get("notes", []),
        base_denominator=args.denominator,
    )

    # Override level number if provided
    if args.level_num is not None:
        converter._infer_level_num = lambda: args.level_num

    # ── Convert ───────────────────────────────────────────────────────────
    output_text = converter.convert()

    # ── Write output ──────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = input_path.parent / "maidata.txt"

    out_path.write_text(output_text, encoding="utf-8")
    print(f"Wrote simai chart to {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    notes = chart.get("notes", [])
    kind_counts: dict[str, int] = {}
    for n in notes:
        k = n["kind"]
        kind_counts[k] = kind_counts.get(k, 0) + 1
    print(f"Chart: {data.get('title', 'Untitled')} [{chart['constant'][0]}.{chart['constant'][1]}]")
    print(f"Notes: {len(notes)} total")
    for kind in ("Tap", "Hold", "Slide", "Touch", "TouchHold"):
        if kind in kind_counts:
            print(f"  {kind}: {kind_counts[kind]}")


if __name__ == "__main__":
    main()
