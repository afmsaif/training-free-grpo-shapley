# Training-Free GRPO with Shapley-Guided Experience Refinement

> **Research code** for the paper: *"Shapley-Guided Experience Refinement for Training-Free GRPO"*

This repository extends [Training-Free GRPO](https://github.com/TencentCloudADP/youtu-agent) with cardinality-restricted Shapley values to identify which learned experiences actually improve agent performance — and which ones hurt it.

---

## Overview

Training-Free GRPO generates *experiences* (short heuristics) from agent rollout trajectories. This work adds three mechanisms to improve experience quality:

| Method | What it does | Cost |
|---|---|---|
| **Shapley scoring** | Measures each experience's actual impact on Pass@1 | ~100 rollout evaluations |
| **LLM-as-judge** | Ranks experiences by perceived quality (DeepSeek-R1-32B) | 1 LLM call |
| **Iterative refinement** | Generate → score → keep top-k → re-rollout, repeat | 2–3× rollout cost |

Key finding: LLM judge and Shapley agree only **18.5%** of the time (ρ = −0.23), showing that perceived experience quality does not predict actual performance impact.

---

## Hardware Requirements

- **4× NVIDIA A6000** (48 GB each, 192 GB total)
- GPU 0,1 → Rollout agent (Qwen2.5-7B)
- GPU 2,3 → Experience extractor / judge (DeepSeek-R1-32B)

---

## Installation

```bash
# Clone the repo
git clone https://github.com/afmsaif/training-free-grpo-shapley.git
cd training-free-grpo-shapley

# Install dependencies (same as base youtu-agent)
conda create -n mas2 python=3.10
conda activate mas2
pip install -r requirements.txt

# Patch vLLM tool parser (fixes hermes JSON parse errors)
# See: utu/practice/README_VLLM_PATCH.md
```

---

## Quick Start

### Step 0 — Start both model servers

Open **two separate terminals**:

```bash
# Terminal 1: Rollout agent — Qwen2.5-7B (GPU 0,1)
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8000 \
  --tensor-parallel-size 2 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --gpu-memory-utilization 0.85

# Terminal 2: Experience extractor — DeepSeek-R1-Distill-Qwen-32B (GPU 2,3)
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --port 8001 \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
```

Verify both are running:
```bash
curl http://localhost:8000/health && echo "Port 8000 OK"
curl http://localhost:8001/health && echo "Port 8001 OK"
```

### Step 0b — Set environment variables

```bash
export UTU_LLM_BASE_URL=http://localhost:8000/v1
export UTU_LLM_API_KEY=xxx
export JUDGE_LLM_TYPE=chat.completions
export JUDGE_LLM_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
export JUDGE_LLM_BASE_URL=http://localhost:8001/v1
export JUDGE_LLM_API_KEY=xxx
```

Add these to `~/.bashrc` to avoid setting them every session.

---

## Full Pipeline (Recommended)

Run the complete pipeline for a dataset in one command:

```bash
# SWE-bench
python -m scripts.run_full_pipeline \
    --dataset swebench \
    --extractor_port 8001 \
    --extractor_model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --m 10 --k 10 --n_seeds 3

# AppWorld
python -m scripts.run_full_pipeline \
    --dataset appworld \
    --extractor_port 8001 \
    --extractor_model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --m 10 --k 10 --n_seeds 3

# Math reasoning (original dataset, no external API needed)
python -m scripts.run_full_pipeline \
    --dataset math \
    --extractor_port 8001 \
    --extractor_model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --m 10 --k 10 --n_seeds 3
```

The pipeline runs four steps automatically:

```
Step 1: Download dataset → upload to GRPO database
Step 2: Training-Free GRPO with iterative refinement (Qwen7B + DeepSeek32B)
Step 3: Compute Shapley scores on held-out split (q100..q199)
Step 4: Compare configurations on eval split (q200..q299) — 4-way table
Step 5: LLM judge ranking + Shapley comparison (Spearman ρ)
```

### Pipeline flags

```bash
--skip_prepare        # Skip dataset download/upload (if already done)
--skip_configs        # Skip YAML config creation (if manually edited)
--skip_grpo           # Skip GRPO training (use existing agent YAML)
--skip_shapley        # Skip Shapley computation (use precomputed scores)
--skip_compare        # Skip comparison evaluation
--overwrite_configs   # Overwrite existing YAML configs (default: skip)
```

---

## Step-by-Step Guide

If you want to run individual steps manually:

### Step 1 — Prepare datasets

```bash
# Download and upload SWE-bench Lite (300 samples)
python scripts/prepare_datasets.py --dataset swebench
python -m scripts.data.upload_dataset \
    --file_path data/swebench_lite.jsonl \
    --dataset_name SWEBench

# Download and upload AppWorld
python scripts/prepare_datasets.py --dataset appworld
python -m scripts.data.upload_dataset \
    --file_path data/appworld.jsonl \
    --dataset_name AppWorld
```

### Step 2 — Run standard Training-Free GRPO (baseline)

```bash
python -m scripts.run_training_free_GRPO \
    --config_name math_reasoning \
    --restart_step 0
```

### Step 3 — Run iterative GRPO with LLM judge

Uses DeepSeek-R1-32B to score experiences at each refinement round:

```bash
python -m scripts.run_llm_judge \
    --config_name math_reasoning \
    --iterative \
    --n_candidates 20 \
    --n_keep 10 \
    --n_refinement_rounds 2 \
    --extractor_port 8001 \
    --extractor_model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --restart_step 0
```

### Step 4 — Compute Shapley scores

```bash
python -m scripts.run_shapley \
    --config_name math_reasoning \
    --experiences_path configs/agents/practice/math_practice_agent.yaml \
    --m 10 \
    --k 10 \
    --train_questions 100 \
    --shapley_size 100 \
    --log_dir logs/shapley/math
```

Results saved to `logs/shapley/math/shapley_progress.json`. You can kill this at any time — results are saved after every V(S) call.

### Step 5 — LLM judge ranking + Shapley comparison

```bash
# Standalone: rank existing experiences with DeepSeek + compare to Shapley
python -m scripts.run_llm_judge \
    --config_name math_reasoning \
    --experiences_path configs/agents/practice/math_practice_agent.yaml \
    --shapley_path logs/shapley/math/shapley_progress.json \
    --extractor_port 8001 \
    --extractor_model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
```

### Step 6 — Compare experience configurations

Evaluates 4 configurations on held-out questions (never used in training or Shapley estimation):

```bash
python -m scripts.compare_experience_sets \
    --config_name math_reasoning \
    --experiences_path configs/agents/practice/math_practice_agent.yaml \
    --shapley_path logs/shapley/math/shapley_progress.json \
    --train_questions 100 \
    --shapley_size 100 \
    --eval_size 100 \
    --n_seeds 3
```

Output:
```
Configuration                                    Mean@1    ±SE    vs baseline
No experiences (baseline)                        0.2400  0.0121  —
All experiences (27 total)                       0.2200  0.0088  (-0.0200)
Positive-psi only (7 exp, 20 removed)            0.2700  0.0115  (+0.0300)
Top-7 by psi (Shapley-optimal)                   0.2700  0.0100  (+0.0300)
```

---

## Dataset Split Design

All datasets use a three-way split to prevent leakage:

```
DAPO-Math-17k / SWEBench / AppWorld (shuffle=False):
  q0   .. q99    → TRAINING  (used by GRPO to generate experiences)
  q100 .. q199   → SHAPLEY   (used to compute psi_i scores)
  q200 .. q299   → EVAL      (used only for final comparison)
```

This guarantees:
- Shapley scores are not inflated by training questions
- Final comparison is not inflated by Shapley estimation questions

---

## Repository Structure

```
scripts/
  run_training_free_GRPO.py   # Standard GRPO (baseline)
  run_llm_judge.py            # LLM judge ranking + iterative GRPO
  run_shapley.py              # Shapley score estimation
  compare_experience_sets.py  # 4-way comparison on eval split
  run_full_pipeline.py        # Runs all steps for a dataset
  prepare_datasets.py         # Download SWE-bench / AppWorld

utu/practice/
  experience_shapley_random.py    # Cardinality-restricted Shapley estimator
  experience_llm_judge.py         # LLM-as-judge scorer
  iterative_shapley_grpo.py       # Iterative refinement GRPO
  training_free_grpo_shapley.py   # Static Shapley feedback GRPO
  experience_updater_shapley.py   # Experience updater with psi labels
  verify/
    code_verify.py                # Verification for SWE-bench and AppWorld

configs/
  practice/
    math_reasoning.yaml           # Math dataset config
    swebench_practice.yaml        # SWE-bench config
    appworld_practice.yaml        # AppWorld config
  eval/
    swebench/swebench_eval.yaml
    appworld/appworld_eval.yaml
```

---

## Logs and Outputs

All results are saved to `logs/shapley/{dataset}/`:

| File | Contents |
|---|---|
| `shapley_progress.json` | Ranked experiences with psi values |
| `shapley_progress.csv` | Same as above, CSV format (open in Excel) |
| `shapley_history.jsonl` | Per-call convergence history |
| `comparison_results.json` | 4-way comparison with all seeds |
| `llm_judge/judge_standalone.json` | LLM judge scores and reasoning |
| `llm_judge/judge_standalone.csv` | Judge rankings (CSV) |

---

## Adding a New Dataset

1. **Prepare data** in GRPO format (`scripts/prepare_datasets.py` as template):
```json
{"dataset": "MyDataset", "source": "training_free_grpo",
 "question": "...", "answer": "..."}
```

2. **Upload**:
```bash
python -m scripts.data.upload_dataset \
    --file_path data/my_dataset.jsonl \
    --dataset_name MyDataset
```

3. **Create verification function** at `utu/practice/verify/my_verify.py`:
```python
def my_verify_func(sample, timeout_score=0, **kwargs):
    if sample.correct_answer.lower() in sample.response.lower():
        return {"reward": 1.0, "reasoning": None}
    return {"reward": 0.0, "reasoning": None}
```

4. **Create eval config** at `configs/eval/my_domain/my_eval.yaml` and **practice config** at `configs/practice/my_practice.yaml` — see existing configs as templates.

5. **Run**:
```bash
python -m scripts.run_full_pipeline --dataset my_dataset ...
```

---

## Common Issues

| Error | Cause | Fix |
|---|---|---|
| `SERPER_API_KEY not set` | Web search agent needs Serper | `export SERPER_API_KEY=your_key` or use a non-web dataset |
| `JUDGE_LLM_TYPE not found` | Web search config needs judge env vars | Set all 4 `JUDGE_LLM_*` env vars |
| `IndexError: list index out of range` | All rollouts failed (server down or wrong dataset) | Check `curl localhost:8000/health` |
| `duplicate key rollout_concurrency` | Pipeline regenerated broken YAML | Fix YAML manually or use `--skip_configs` |
| `Mean@32 = 0` | Wrong `Mean@k` key being read | Update to dynamic key lookup (already fixed) |
| `Connection error` | vLLM server not running | Restart both servers |

---

## Citation

If you use this code, please cite:

```bibtex
@misc{training-free-grpo-shapley,
  title  = {Shapley-Guided Experience Refinement for Training-Free GRPO},
  author = {Saif, A.F.M. and others},
  year   = {2026},
  url    = {https://github.com/afmsaif/training-free-grpo-shapley}
}
```

This work builds on:
```bibtex
@misc{youtu-agent,
  title = {Training-Free GRPO},
  author = {TencentCloudADP},
  url   = {https://github.com/TencentCloudADP/youtu-agent}
}

@inproceedings{ghorbani2019,
  title  = {Data Shapley: Equitable Valuation of Data for Machine Learning},
  author = {Ghorbani, Amirata and Zou, James},
  booktitle = {ICML},
  year   = {2019}
}

@article{castro2009,
  title   = {Polynomial calculation of the Shapley value based on sampling},
  author  = {Castro, Javier and G{\'o}mez, Daniel and Tejada, Juan},
  journal = {Computers \& Operations Research},
  year    = {2009}
}
```
