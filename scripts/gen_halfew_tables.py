#!/usr/bin/env python3
"""
gen_halfew_tables.py — regenerate Tables 2 and 3 for
correspondence/2026-11-26-halfew-out-of-sample.tex
directly from results/novel_dgp_halfew_results.json.

Usage:
    python3 scripts/gen_halfew_tables.py

Writes two files:
    scripts/out/table2_reversed.tex
    scripts/out/table3_twopeak.tex

Also asserts that every cell matches the JSON within tolerance 0.0015.
Run from the repo root.
"""

import json
import sys
import os
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
JSON_PATH = REPO_ROOT / "results" / "novel_dgp_halfew_results.json"
OUT_DIR   = pathlib.Path(__file__).resolve().parent / "out"

# Tolerance for the final assertion (match to 3 decimal places after rounding)
TOLERANCE = 0.0015


def fmt_lw(x: float) -> str:
    """Format mean log-wealth to 3 decimal places."""
    return f"{x:.3f}"


def fmt_pow(x: float) -> str:
    """Format power to 3 decimal places."""
    return f"{x:.3f}"


def fmt_cliff(x: float) -> str:
    """Format Cliff's delta with sign, 3 decimal places."""
    if x >= 0:
        return f"$+{x:.3f}$"
    else:
        return f"${x:.3f}$"


def fmt_worst(x: float) -> str:
    """Format worst shortfall with sign, 3 decimal places."""
    if x >= 0:
        return f"$+{x:.3f}$"
    else:
        return f"${x:.3f}$"


def build_table_body(cells: list) -> str:
    """Build the midrule rows for a table from a list of JSON cells."""
    rows = []
    for c in cells:
        g      = c["gamma"]
        mean_ew = c["mean_ew"]
        mean_he = c["mean_he"]
        mean_mw = c["mean_mw"]
        pow_ew  = c["pow_ew"]
        pow_he  = c["pow_he"]
        cliff   = c["he_cliff"]
        worst   = c["he_worst"]

        g_str = f"{g:.2f}"

        row = (
            f"{g_str} & "
            f"{fmt_lw(mean_ew)} & "
            f"{fmt_lw(mean_he)} & "
            f"{fmt_lw(mean_mw)} & "
            f"{fmt_pow(pow_ew)} & "
            f"{fmt_pow(pow_he)} & "
            f"{fmt_cliff(cliff)} & "
            f"{fmt_worst(worst)} \\\\"
        )
        rows.append(row)
    return "\n".join(rows)


def verify_table(cells: list, table_body: str, label: str) -> int:
    """
    Parse back the generated table body and compare with JSON.
    Returns the number of mismatches (should be 0).
    """
    lines = [l.strip() for l in table_body.strip().splitlines() if l.strip()]
    mismatches = 0
    if len(lines) != len(cells):
        print(f"ERROR [{label}]: expected {len(cells)} rows, got {len(lines)}")
        return 999

    for cell, line in zip(cells, lines):
        # strip trailing \\
        line = line.rstrip("\\").strip()
        parts = [p.strip() for p in line.split("&")]
        # parts: gamma, mean_ew, mean_he, mean_mw, pow_ew, pow_he, cliff, worst
        def parse_num(s):
            s = s.replace("$", "").replace("+", "").strip()
            return float(s)

        json_vals = [
            cell["mean_ew"], cell["mean_he"], cell["mean_mw"],
            cell["pow_ew"], cell["pow_he"],
            cell["he_cliff"], cell["he_worst"],
        ]
        # Round JSON vals to 3 dp for comparison (same rounding as fmt_ functions)
        json_rounded = [round(v, 3) for v in json_vals]
        tex_vals = [parse_num(parts[i+1]) for i in range(7)]

        for i, (jv, tv) in enumerate(zip(json_rounded, tex_vals)):
            if abs(jv - tv) > TOLERANCE:
                cols = ["mean_ew", "mean_he", "mean_mw",
                        "pow_ew", "pow_he", "cliff", "worst"]
                g = cell["gamma"]
                print(f"MISMATCH [{label}] gamma={g} {cols[i]}: "
                      f"JSON_rounded={jv:.4f} TEX={tv:.4f} "
                      f"diff={jv-tv:.4f}")
                mismatches += 1

    return mismatches


def main():
    if not JSON_PATH.exists():
        print(f"ERROR: JSON not found at {JSON_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(JSON_PATH) as f:
        data = json.load(f)

    rev_cells = data["reversed_gradient"]
    two_cells = data["two_peak"]

    body2 = build_table_body(rev_cells)
    body3 = build_table_body(two_cells)

    # Assert correctness before writing
    mis2 = verify_table(rev_cells, body2, "Table2-REVERSED")
    mis3 = verify_table(two_cells, body3, "Table3-TWOPEAK")
    total_mis = mis2 + mis3

    if total_mis > 0:
        print(f"\nASSERTION FAILED: {total_mis} cell(s) do not match JSON.",
              file=sys.stderr)
        sys.exit(1)
    else:
        print(f"ASSERTION PASSED: 0 mismatches across "
              f"{len(rev_cells)*7 + len(two_cells)*7} data cells "
              f"({len(rev_cells)} + {len(two_cells)} rows × 7 columns).")

    # Write outputs
    OUT_DIR.mkdir(exist_ok=True)
    out2 = OUT_DIR / "table2_reversed.tex"
    out3 = OUT_DIR / "table3_twopeak.tex"
    out2.write_text(body2 + "\n")
    out3.write_text(body3 + "\n")
    print(f"Wrote {out2}")
    print(f"Wrote {out3}")

    # Also print to stdout for review
    print("\n=== TABLE 2 (REVERSED) BODY ===")
    print(body2)
    print("\n=== TABLE 3 (TWO-PEAK) BODY ===")
    print(body3)


if __name__ == "__main__":
    main()
