#!/usr/bin/env python3
"""Independent verification: Pearson correlation matrix statistics for 32 LLM judges
on alphaNLI (baseline arm).

Data format (from reading data-loading code only, NOT from any existing analysis scripts):
  - /repo/datasets/chaosnli-alphanli-1000/items/uids.txt: one UID per line, 1000 items
  - /repo/datasets/chaosnli-alphanli-1000/votes/baseline/*.jsonl: one file per judge
    Each line is JSON: {"uid": "<uid>", "label": "1" or "2", "parse_fail": false/true}

Encoding: label "1" -> 1, label "2" -> 0.
Complete-case: drop items where ANY judge has parse_fail=True or label not in {"1","2"}.

Computes:
  - 32x32 Pearson correlation matrix R
  - off-diagonal summary stats
  - eigenvalues (lambda_max, lambda_min, sum of squares)
  - participation ratio = (sum lambda)^2 / sum(lambda^2)
  - Kish plug-in = 32 / (1 + 31 * rho_bar)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent
REPO = PROJECT / "repo"
VOTES_DIR = REPO / "datasets" / "chaosnli-alphanli-1000" / "votes" / "baseline"
UIDS_FILE = REPO / "datasets" / "chaosnli-alphanli-1000" / "items" / "uids.txt"

# ---------------------------------------------------------------------------
# Load UIDs (preserving order)
# ---------------------------------------------------------------------------
uids_ordered = UIDS_FILE.read_text(encoding="utf-8").split()
n_items_total = len(uids_ordered)
print(f"Total items in uids.txt: {n_items_total}")

# ---------------------------------------------------------------------------
# Load votes for all 32 judges
# ---------------------------------------------------------------------------
judge_files = sorted(VOTES_DIR.glob("*.jsonl"))
n_judges = len(judge_files)
print(f"Judge files found: {n_judges}")
assert n_judges == 32, f"Expected 32 judges, got {n_judges}"

# For each judge, build a dict: uid -> vote (1 or 0) or None if parse failure / bad label
judge_votes: list[dict[str, int | None]] = []
judge_names: list[str] = []

for jpath in judge_files:
    judge_names.append(jpath.stem)
    uid_to_vote: dict[str, int | None] = {}
    with open(jpath, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            uid = str(r["uid"])
            if r.get("parse_fail", False):
                uid_to_vote[uid] = None
            else:
                lbl = str(r["label"])
                if lbl == "1":
                    uid_to_vote[uid] = 1
                elif lbl == "2":
                    uid_to_vote[uid] = 0
                else:
                    uid_to_vote[uid] = None  # unexpected label
    judge_votes.append(uid_to_vote)

# ---------------------------------------------------------------------------
# Complete-case filtering: keep only items where ALL 32 judges have valid votes
# ---------------------------------------------------------------------------
valid_uids = []
for uid in uids_ordered:
    if all(jv.get(uid) is not None for jv in judge_votes):
        valid_uids.append(uid)

n_kept = len(valid_uids)
n_dropped = n_items_total - n_kept
print(f"Items kept (all 32 judges valid): {n_kept}")
print(f"Items dropped (>=1 parse failure or bad label): {n_dropped}")

# ---------------------------------------------------------------------------
# Build 32 x n_kept binary matrix
# ---------------------------------------------------------------------------
# X[j, i] = vote of judge j on item i (1 if label "1", 0 if label "2")
X = np.zeros((32, n_kept), dtype=np.float64)
for j, jv in enumerate(judge_votes):
    for i, uid in enumerate(valid_uids):
        X[j, i] = jv[uid]  # type: ignore[arg-type]

print(f"\nVote matrix shape: {X.shape}  (judges x items)")
print(f"Mean vote per judge (fraction voting '1'): min={X.mean(axis=1).min():.4f}  max={X.mean(axis=1).max():.4f}")

# ---------------------------------------------------------------------------
# Pearson correlation matrix R (32 x 32)
# ---------------------------------------------------------------------------
# np.corrcoef(X) treats each row as a variable — exactly what we want
R = np.corrcoef(X)  # shape (32, 32)
print(f"\nR shape: {R.shape}")

# Check diagonal
diag_vals = np.diag(R)
print(f"Diagonal: min={diag_vals.min():.6f}  max={diag_vals.max():.6f}  (should all be 1.0)")

# ---------------------------------------------------------------------------
# Off-diagonal statistics
# ---------------------------------------------------------------------------
mask = ~np.eye(32, dtype=bool)
off_diag = R[mask]  # 32*31 = 992 values

rho_bar = float(np.mean(off_diag))
rho_std = float(np.std(off_diag, ddof=0))  # population std over all 992 off-diag entries
rho_min = float(np.min(off_diag))
rho_max = float(np.max(off_diag))

print(f"\n--- Off-diagonal Pearson correlations ({len(off_diag)} entries) ---")
print(f"rho_bar (mean):  {rho_bar:.4f}")
print(f"std:             {rho_std:.4f}")
print(f"min:             {rho_min:.4f}")
print(f"max:             {rho_max:.4f}")

# ---------------------------------------------------------------------------
# Eigenvalues of R
# ---------------------------------------------------------------------------
# np.linalg.eigh is appropriate for real symmetric matrices (numerically stable)
eigenvalues = np.linalg.eigh(R)[0]  # returns sorted ascending real eigenvalues

lambda_min = float(eigenvalues[0])
lambda_max = float(eigenvalues[-1])
trace_R = float(np.sum(eigenvalues))
sum_sq_lambda = float(np.sum(eigenvalues ** 2))

print(f"\n--- Eigenvalues of R ---")
print(f"lambda_max:       {lambda_max:.4f}")
print(f"lambda_min:       {lambda_min:.4f}")
print(f"trace(R) = Σλ:    {trace_R:.4f}  (should be 32)")
print(f"Σ(λ²):            {sum_sq_lambda:.4f}")

# ---------------------------------------------------------------------------
# Participation ratio = (Σλ)² / Σ(λ²)
# ---------------------------------------------------------------------------
participation_ratio = (trace_R ** 2) / sum_sq_lambda
print(f"\n--- Participation ratio ---")
print(f"PR = (Σλ)² / Σ(λ²): {participation_ratio:.4f}")

# ---------------------------------------------------------------------------
# Kish plug-in
# ---------------------------------------------------------------------------
kish = 32.0 / (1.0 + 31.0 * rho_bar)
print(f"\n--- Kish plug-in n_eff ---")
print(f"Kish = 32 / (1 + 31*rho_bar): {kish:.4f}")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("SUMMARY (all to 4 dp)")
print("="*60)
print(f"Items total:          {n_items_total}")
print(f"Items kept:           {n_kept}")
print(f"Items dropped:        {n_dropped}")
print(f"Judges:               {n_judges}")
print(f"")
print(f"rho_bar:              {rho_bar:.4f}")
print(f"rho_std:              {rho_std:.4f}")
print(f"rho_min:              {rho_min:.4f}")
print(f"rho_max:              {rho_max:.4f}")
print(f"")
print(f"lambda_max:           {lambda_max:.4f}")
print(f"lambda_min:           {lambda_min:.4f}")
print(f"trace(R):             {trace_R:.4f}")
print(f"sum(lambda^2):        {sum_sq_lambda:.4f}")
print(f"")
print(f"Participation ratio:  {participation_ratio:.4f}")
print(f"Kish plug-in:         {kish:.4f}")
