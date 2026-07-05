# hrguard

Relational manipulation across turns and paraphrases in AI agents.

![hrguard teaser](teaser.jpg)

`hrguard` contains the benchmark, generation scripts, analysis code, and supporting assets for evaluating whether an agent can distinguish harmful relationship-manipulation requests from protective victim-side requests.

## What is included

The repository currently keeps the project content under `datasets/`:

- benchmark construction scripts
- multi-turn prompt generation scripts
- adversarial paraphrase generation scripts
- judge and gate application scripts
- benchmark analysis and plotting scripts
- benchmark datasets in JSONL / JSON format
- figures and assets used in the paper

## Main benchmark sets

- `datasets/dataset/openclaw_structured_1100.jsonl`: main 1100-prompt structured benchmark
- `datasets/dataset/openclaw_structured_1100_current.jsonl`: current-version prompt set
- `datasets/dataset/openclaw_structured_1100_adversarial_paraphrased.jsonl`: adversarially paraphrased variant
- `datasets/dataset/openclaw_multiturn_level4plus_current_advpara.jsonl`: 1000-item multi-turn benchmark
- `datasets/dataset/multiturn_40_prompts.jsonl`: 40-case multi-turn stress test
- `datasets/dataset/vulnerability_profiled_victim_12.jsonl`: vulnerability-profiled victim slice
- `datasets/dataset/benign_relationship_control_30.jsonl`: benign control set

## Key scripts

- `datasets/openclaw_generate_real.py`: run OpenClaw-generated outputs through a gateway or local agent
- `datasets/openclaw_judge_batch.py`: judge batches of outputs
- `datasets/apply_relationship_gate.py`: apply the relationship-specific gate
- `datasets/make_openclaw_multiturn_level4plus.py`: build the level-4-plus multi-turn set
- `datasets/make_openclaw_adversarial_paraphrase_benchmark.py`: build paraphrased benchmark variants
- `datasets/plot_relationship_harm_results.py`: plot the main relationship-harm results
- `datasets/make_fraudr1_robustness_*`: construct Fraud-R1 robustness figures and summaries

## Environment

The code was developed with Python 3.10+. For the benchmark scripts:

```bash
pip install -r datasets/requirement.txt
```

The OpenClaw-based runs also depend on a working OpenClaw installation and a reachable model backend, such as a local Ollama server.

## Running generation

Example gateway-based run:

```bash
python datasets/openclaw_generate_real.py \
  --input datasets/dataset/openclaw_multiturn_level4plus_current_advpara.jsonl \
  --output datasets/results/openclaw_multiturn_level4plus_current_advpara_ollama_gateway.jsonl \
  --transport gateway \
  --model ollama/llama3.2 \
  --condition multiturn-level4plus-current-advpara \
  --batch-size 3 \
  --batch-sleep-seconds 300 \
  --item-sleep-seconds 15 \
  --pre-row-sleep-seconds 15 \
  --resume
```

## Running judgment and analysis

Typical workflow:

1. generate model outputs
2. judge the outputs with the separate judge pipeline
3. apply the relationship gate
4. summarize or plot the results

The exact command depends on the target condition, but the repository includes the scripts needed for each stage.

## Notes

- This repository intentionally separates prompt construction, generation, judgment, and plotting.
- Generated outputs in `datasets/results/` should generally be treated as run artifacts rather than source data.
- The project includes harmful content for research and evaluation purposes only.
