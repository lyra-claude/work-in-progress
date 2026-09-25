"""Blind independent verification of error-correlation n_eff on alphaNLI.

Written from scratch. Does NOT read the other agent's script or results.
Builds the 32x32 Pearson correlation of per-judge error-vs-gold indicators,
computes Kish and participation-ratio n_eff, and a compound-symmetry control.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/lyra/projects/chaosnli-neff")
VOTES_DIR = ROOT / "repo/datasets/chaosnli-alphanli-1000/votes/baseline"
UIDS_FILE = ROOT / "repo/datasets/chaosnli-alphanli-1000/items/uids.txt"
RAW_HUMAN = ROOT / "chaosNLI_v1.0/chaosNLI_alphanli.jsonl"
LABELS = ("1", "2")  # alphaNLI binary

# --- item roster (defines column order) ---
uids = UIDS_FILE.read_text().split()
assert len(uids) == len(set(uids))
uid_index = {u: i for i, u in enumerate(uids)}
n_items = len(uids)

# --- gold from majority_label in the raw ChaosNLI file ---
gold_by_uid = {}
with RAW_HUMAN.open() as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        u = str(row["uid"])
        if u in uid_index:
            gold_by_uid[u] = str(row["majority_label"])
missing_gold = [u for u in uids if u not in gold_by_uid]
assert not missing_gold, f"missing gold for {len(missing_gold)} uids"
gold = np.array([LABELS.index(gold_by_uid[u]) for u in uids], dtype=int)

# --- load votes, judges sorted alphabetically by filename ---
vote_files = sorted(VOTES_DIR.glob("*.jsonl"))
judges = [p.stem for p in vote_files]
k = len(judges)
assert k == 32, f"expected 32 judges, got {k}"

lab_idx = {v: i for i, v in enumerate(LABELS)}
votes = np.full((k, n_items), -1, dtype=int)
parse_fail = np.zeros((k, n_items), dtype=bool)

for j, path in enumerate(vote_files):
    seen = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            u = str(r["uid"])
            if u not in uid_index:
                continue
            seen.add(u)
            lab = str(r.get("label", ""))
            pf = r.get("parse_fail", False)
            if lab not in lab_idx or pf:
                parse_fail[j, uid_index[u]] = True
                votes[j, uid_index[u]] = -1 if lab not in lab_idx else lab_idx[lab]
            else:
                votes[j, uid_index[u]] = lab_idx[lab]
    # any uid never seen counts as missing
    for u in uids:
        if u not in seen:
            parse_fail[j, uid_index[u]] = True
            votes[j, uid_index[u]] = -1

# --- complete-case: drop items where ANY judge missing/parse-fail ---
missing_or_bad = parse_fail | (votes < 0)
affected = missing_or_bad.any(axis=0)
keep = ~affected
n = int(keep.sum())

votes_kept = votes[:, keep]
gold_kept = gold[keep]

# --- error-vs-gold indicator: e_ij = 1 if vote != gold ---
E = (votes_kept != gold_kept[None, :]).astype(float)  # (k, n)

# --- Pearson correlation across items, unit diagonal ---
C = np.corrcoef(E)  # 32x32
np.fill_diagonal(C, 1.0)

off = C[np.triu_indices(k, 1)]
rho_bar = float(off.mean())
off_std = float(off.std(ddof=0))

eig = np.linalg.eigvalsh(C)
sum_eig = float(eig.sum())
sum_eig_sq = float((eig ** 2).sum())

# --- 2. Kish n_eff ---
n_eff_Kish = 32 / (1 + 31 * rho_bar)

# --- 3. PR n_eff ---
n_eff_PR = sum_eig ** 2 / sum_eig_sq

# --- 4. Compound-symmetry control ---
n_eff_PR_CS = 32 ** 2 / ((1 + 31 * rho_bar) ** 2 + 31 * (1 - rho_bar) ** 2)

# --- 5. Gap decomposition ---
total = n_eff_PR - n_eff_Kish
formula = n_eff_PR_CS - n_eff_Kish
cs_dep = n_eff_PR - n_eff_PR_CS

def pct(x):
    return 100.0 * x / total if total != 0 else float("nan")

print(f"n (complete-case items)      = {n}")
print(f"parse_fail/missing items     = {int(affected.sum())}")
print(f"rho_bar (mean off-diag)      = {rho_bar:.4f}")
print(f"off-diag std                 = {off_std:.4f}")
print(f"sum eigenvalues              = {sum_eig:.4f}")
print(f"sum squared eigenvalues      = {sum_eig_sq:.4f}")
print(f"n_eff_Kish                   = {n_eff_Kish:.4f}")
print(f"n_eff_PR                     = {n_eff_PR:.4f}")
print(f"n_eff_PR_CS                  = {n_eff_PR_CS:.4f}")
print("--- gap decomposition ---")
print(f"total   = n_eff_PR - n_eff_Kish      = {total:.4f}  ({pct(total):.1f}%)")
print(f"formula = n_eff_PR_CS - n_eff_Kish   = {formula:.4f}  ({pct(formula):.1f}%)")
print(f"cs_dep  = n_eff_PR - n_eff_PR_CS     = {cs_dep:.4f}  ({pct(cs_dep):.1f}%)")

# Cross-check: paper reported values
rv = json.loads((ROOT / "repo/reference/reported_values.json").read_text())
alpha = rv["datasets"]["alphanli"]
print("--- cross-check vs reported_values.json (alphanli) ---")
print(f"paper n_eff                  = {alpha['n_eff']:.4f}")
print(f"paper nu_H                   = {alpha['nu_H']:.4f}")
print(f"paper PR                     = {alpha['PR']:.4f}")
print(f"my n_eff_Kish                = {n_eff_Kish:.4f}")
print(f"my n_eff_PR (error-vs-gold)  = {n_eff_PR:.4f}")
