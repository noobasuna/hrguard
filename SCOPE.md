# Release scope — Relationship Manipulation Gating (HrGuard)

This GitHub repo should contain **only** artifacts needed to reproduce the
HrGuard paper contribution (relationship-manipulation gating in agentic
multi-turn conversation).

## Keep (public)

### Core gate / judge / generate
- `apply_cumulative_relationship_gate.py`
- `romance_scam_judge.py`
- `openai_batch_judge.py`
- `make_turn_level_judge_inputs.py`
- `openclaw_generate_sequential.py`
- `openclaw_generate_real_sequential_pregated.py`
- `openclaw_generate_real.py` (optional helper)

### Benchmark prompts
- `datasets/dataset/openclaw_multiturn_level4plus_current_advpara.jsonl`
- `datasets/dataset/openclaw_structured_1100*.jsonl` (if cited)
- `datasets/dataset/benign_relationship_control_30.jsonl` (if cited)
- `datasets/dataset/benign_ambiguous_controls_12.jsonl` (if cited)
- `datasets/dataset/multiturn_40_prompts.jsonl` (optional stress set)

### Paper-facing helpers
- `datasets/apply_relationship_gate.py`
- `datasets/apply_llamaguard_baseline.py`
- `datasets/apply_shieldgemma_baseline.py`
- `datasets/apply_qwen3guard_baseline.py`
- `datasets/romance_scam_judge.py`
- `datasets/requirement.txt`
- `datasets/make_openclaw_multiturn_level4plus.py`
- `datasets/make_openclaw_adversarial_paraphrase_benchmark.py`
- figure/table builders under `results/figures/make_*.py` that produce paper figures
  (optional; prefer a slim `paper/` export)

### Meta
- `README.md`
- `LICENSE`
- `SCOPE.md` (this file)
- teaser image used in the paper (if any)

## Do NOT publish

- `.env`, API keys, `gateaway.txt`, credentials
- `logs/`, `__pycache__/`, `openclaw-tsubame-home/`
- `results/judge-pipeline/` (full judged JSONL; host on Zenodo if needed)
- cluster-only `submit_*.sh` with personal `/work/9/...` paths (or sanitize first)
- Fraud-R1 / phishing poster assets and scripts:
  - `datasets/make_fraudr1_*`
  - `datasets/dataset/FP-base-full/`
  - `datasets/dataset/FP-levelup-full/`
  - `datasets/asset/` fraud posters
  - `datasets/attacks/` (generic attack toolkit unrelated to HrGuard gating)
- large generation dumps (`*.jsonl` model outputs at repo root)
- Excel scratch workbooks, `vllmlog.txt`, `Untitled`

## Suggested prune commands (before push)

```bash
# stop tracking / remove off-paper paths (example)
git rm -r --cached \
  datasets/attacks \
  datasets/asset \
  datasets/dataset/FP-base-full \
  datasets/dataset/FP-levelup-full \
  datasets/make_fraudr1_robustness_grouped_bar.py \
  datasets/make_fraudr1_robustness_motivation_figure.py \
  datasets/make_fraudr1_robustness_side_by_side.py \
  2>/dev/null || true

# never add secrets
# ensure .gitignore covers .env logs results/judge-pipeline
```

Then add only the keep-list files, commit, and `git push origin main`.
