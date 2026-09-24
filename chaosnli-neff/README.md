# ChaosNLI N-eff Distributional Diversity Pipeline

This pipeline measures per-item distributional diversity (inverse-Simpson D) across 3000 NLI items from ChaosNLI, comparing a crowd of ~100 human annotators to a fixed 32-LLM judge panel from Li, Yu, Li ("How Many Humans Is a Judge Panel Worth?", arXiv 2609.21277). D = (Σc_l)²/Σc_l² is the effective number of active labels per item — an apples-to-apples within-item spread measure computed identically from both sources. The analysis carefully separates this from two other panel metrics (n_eff_panel and ν_H) that measure inter-judge correlation structure, not label spread. Human data is ChaosNLI v1.0 (Nie, Zhou, Bansal, EMNLP 2020; CC BY-NC 4.0); LLM vote data is from Chao1208/32judges-votes (CC BY 4.0).

## Reproduce

```bash
# 1. Clone the vote repo (already done if you have this directory)
git clone https://github.com/Chao1208/32judges-votes repo/

# 2. Download ChaosNLI v1.0 (CC BY-NC 4.0 — not redistributed here)
#    from https://github.com/easonnie/ChaosNLI and place the three JSONL files at:
#    chaosNLI_v1.0/chaosNLI_mnli_m.jsonl
#    chaosNLI_v1.0/chaosNLI_snli.jsonl
#    chaosNLI_v1.0/chaosNLI_alphanli.jsonl

# 3. Set up venv
python3 -m venv .venv
.venv/bin/pip install numpy pandas matplotlib

# 4. Run
.venv/bin/python3 compute.py
```

Outputs: `results.md` (tables + estimand definitions), `figure.png` (violin plots).

Vendor-stratified analysis (within-vendor vs cross-vendor inter-judge agreement) lives in `compute_vendor.py` / `results-vendor.md` / `figure-vendor.png`.
