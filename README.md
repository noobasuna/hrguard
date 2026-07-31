# HrGuard: Relationship Manipulation Gating Policy in Agentic AI Conversation

![HrGuard teaser](teaser_page-0001.jpg)

Code and benchmark artifacts for **HrGuard** — a transcript-aware **relationship-manipulation gating policy** for multi-turn agentic conversations.

HrGuard monitors evolving attacker/victim dialogs and intervenes with a shared cumulative risk score via:

- **Pregate** (online, before generation)
- **Postgate** (offline, after generation; stop-after-trigger)

Primary evaluation uses role-conditioned metrics under a GPT-4o-mini outcome judge:
attacker harmful compliance ↓ and victim protective intervention ↑.

> This repository is scoped to the **Relationship Manipulation Gating** paper contribution only.
> Unrelated fraud-benchmark / attack-toolkit material is not part of this release.

## Repository contents

| Path | Role |
|---|---|
| `apply_cumulative_relationship_gate.py` | Core Postgate / cumulative relationship gate |
| `main_judge.py` | Outcome / turn judge (OpenAI / local) |
| `openai_batch_judge.py` | OpenAI Batch API wrapper for the same judge |
| `make_turn_level_judge_inputs.py` | Expand sequential dialogs into per-turn judge rows |
| `openclaw_generate_sequential.py` | Multi-turn Raw / GS generation |
| `openclaw_generate_real_sequential_pregated.py` | Multi-turn Pregate generation |
| `datasets/dataset/openclaw_multiturn_level4plus_current_advpara.jsonl` | Main 1000-dialog benchmark (500 attacker / 500 victim) |
| `datasets/apply_relationship_gate.py` | Lightweight gate helper used in analysis |
| `datasets/apply_*_baseline.py` | Industry-guard baselines (LlamaGuard / ShieldGemma / Qwen3Guard) |
| `datasets/requirement.txt` | Python dependencies |

## Pipeline

```text
benchmark JSONL
    → sequential generation (Raw / GS / Pregate)
    → turn-level judge
    → cumulative relationship gate (Postgate)
    → final outcome judge
    → H / P / R and H_A / P_V
```

### 1. Turn judge inputs

```bash
python make_turn_level_judge_inputs.py \
  --input path/to/sequential.jsonl \
  --output path/to/turn_judge_inputs.jsonl
```

### 2. Judge turns

```bash
python main_judge.py \
  --transport openai \
  --model gpt-4o-mini \
  --input path/to/turn_judge_inputs.jsonl \
  --output path/to/turn_judged.jsonl
```

### 3. Apply Postgate

```bash
python apply_cumulative_relationship_gate.py \
  --raw path/to/sequential.jsonl \
  --judged path/to/turn_judged.jsonl \
  --output path/to/cumulative_gated_stopped.jsonl \
  --require-judged-turns \
  --turn-threshold 5.0 \
  --cumulative-threshold 6.0 \
  --decay 0.85 \
  --stop-after-trigger
```

Default gate policy (paper main setting):

- Turn threshold: τ_turn=5
- Cumulative threshold: τ_cum=6
- Decay factor: λ=0.85

- **oracle-role**: victim-mode dialogs bypass the gate unless `--ignore-mode-label`

### 4. Final judge on gated transcripts

**Important:** for gated rows, judge the gated `final_response` (refusal), not `raw_final_response`.

```bash
python main_judge.py \
  --transport openai \
  --model gpt-4o-mini \
  --input path/to/cumulative_gated_stopped.jsonl \
  --output path/to/final_judged.jsonl
```

## Environment

Python 3.10+ recommended:

```bash
pip install -r datasets/requirement.txt
```

OpenClaw (or another OpenAI-compatible generator backend) is required for multi-turn generation.
Set `OPENAI_API_KEY` when using `--transport openai`.

## Ethics

This project contains **relationship-manipulation evaluation prompts** for research only.
Do not use the benchmark to build or deploy manipulative agents.
Released artifacts separate prompt construction, generation, judgment, and gating so defenses can be audited independently.

## Citation

If you use this repository, please cite the paper:

**Relationship Manipulation Gating Policy in Agentic AI Conversation** (HrGuard).

## License

MIT — see `LICENSE`.
