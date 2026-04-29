#!/usr/bin/env python3
"""
Full pipeline: prepare dataset → GRPO training → Shapley scoring → comparison.

Usage:
    # SWE-bench full pipeline
    python scripts/run_full_pipeline.py --dataset swebench \
        --extractor_port 8001 \
        --extractor_model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B

    # AppWorld full pipeline
    python scripts/run_full_pipeline.py --dataset appworld \
        --extractor_port 8001 \
        --extractor_model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B

Env vars required:
    JUDGE_LLM_TYPE=chat.completions
    JUDGE_LLM_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
    JUDGE_LLM_BASE_URL=http://localhost:8001/v1
    JUDGE_LLM_API_KEY=xxx
    UTU_LLM_BASE_URL=http://localhost:8000/v1
    UTU_LLM_API_KEY=xxx
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, ".")

DATASET_CONFIGS = {
    "swebench": {
        "dataset_name": "SWEBench",
        "config_name": "swebench_practice",
        "eval_config": "swebench/swebench_eval",
        "experiences_path": "configs/agents/practice/swebench_practice_agent.yaml",
        "agent_objective": (
            "You are a software engineering agent that resolves GitHub issues by "
            "generating git patches. Given a repository, issue description, and "
            "relevant code context, produce a correct diff patch."
        ),
        "learning_objective": (
            "Extract experiences about effective debugging strategies, common bug "
            "patterns, how to correctly format git patches, and which types of code "
            "changes are most likely to fix specific categories of issues."
        ),
    },
    "appworld": {
        "dataset_name": "AppWorld",
        "config_name": "appworld_practice",
        "eval_config": "appworld/appworld_eval",
        "experiences_path": "configs/agents/practice/appworld_practice_agent.yaml",
        "agent_objective": (
            "You are a digital assistant agent that completes tasks across multiple "
            "apps (Spotify, Amazon, Gmail, Venmo, Google Calendar, Splitwise) by "
            "writing and executing Python API calls."
        ),
        "learning_objective": (
            "Extract experiences about effective multi-app task decomposition, "
            "correct API usage patterns, how to handle errors gracefully, and "
            "strategies for completing complex cross-app workflows."
        ),
    },
}


def create_configs(dataset: str, cfg: dict, overwrite: bool = False):
    """Create eval and practice YAML configs for the dataset.
    
    Skips creation if files already exist unless overwrite=True,
    so manually edited configs are never accidentally overwritten.
    """
    dataset_name = cfg["dataset_name"]
    config_name  = cfg["config_name"]

    # --- Eval config ---
    eval_dir = f"configs/eval/{dataset}"
    os.makedirs(eval_dir, exist_ok=True)
    eval_yaml = f"""\
# @package _global_
defaults:
  - /agents/practice/math_agent@agent   # reuse math agent (has execute_python_code)
  - _self_

exp_id: "{dataset}_eval"

data:
  dataset: "{dataset_name}"
  type: "single"

concurrency: 32
pass_k: 1

verify_filename: "code_verify.py"
verify_func_name: "{dataset}_verify_func"
"""
    eval_path = f"{eval_dir}/{dataset}_eval.yaml"
    if os.path.exists(eval_path) and not overwrite:
        print(f"  Skipped (already exists): {eval_path}")
    else:
        with open(eval_path, "w") as f:
            f.write(eval_yaml)
        print(f"  Created {eval_path}")

    # --- Practice config ---
    os.makedirs("configs/practice", exist_ok=True)
    practice_yaml = f"""\
# @package _global_
defaults:
  - /eval/{dataset}/{dataset}_eval@evaluation
  - _self_

exp_id: "{dataset}_practice"

practice:
  epochs: 1
  batch_size: 20
  grpo_n: 5
  rollout_concurrency: 32
  rollout_temperature: 0.7
  task_timeout: 600
  do_eval: false
  shuffle_data: true
  rollout_data_truncate: 100
  given_ground_truth: true
  num_experiences_per_query: 2

  agent_objective: |
    {cfg["agent_objective"]}

  learning_objective: |
    {cfg["learning_objective"]}

data:
  practice_dataset_name: "{dataset_name}"
"""
    practice_path = f"configs/practice/{config_name}.yaml"
    if os.path.exists(practice_path) and not overwrite:
        print(f"  Skipped (already exists): {practice_path}")
    else:
        with open(practice_path, "w") as f:
            f.write(practice_yaml)
        print(f"  Created {practice_path}")


async def run_step(description: str, cmd: str, env_extra: dict = None):
    """Run a shell command and print status."""
    print(f"\n{'='*65}")
    print(f"STEP: {description}")
    print(f"CMD:  {cmd}")
    print(f"{'='*65}")

    env = {**os.environ, **(env_extra or {})}
    result = subprocess.run(cmd, shell=True, env=env)
    if result.returncode != 0:
        print(f"  ✗ Failed with exit code {result.returncode}")
        raise RuntimeError(f"Step failed: {description}")
    print(f"  ✓ Done")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["swebench", "appworld", "both"],
        default="swebench",
    )
    parser.add_argument("--extractor_port", type=int, default=8001)
    parser.add_argument(
        "--extractor_model",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    )
    parser.add_argument("--skip_prepare",        action="store_true",
                        help="Skip dataset download/upload and config creation")
    parser.add_argument("--skip_configs",         action="store_true",
                        help="Skip YAML config creation even when preparing dataset")
    parser.add_argument("--overwrite_configs",     action="store_true",
                        help="Overwrite existing YAML configs (default: skip if exists)")
    parser.add_argument("--skip_grpo",            action="store_true")
    parser.add_argument("--skip_shapley",         action="store_true")
    parser.add_argument("--skip_compare",         action="store_true")
    parser.add_argument("--m",  type=int, default=10, help="Shapley subsets")
    parser.add_argument("--k",  type=int, default=10, help="Shapley budget")
    parser.add_argument("--n_seeds", type=int, default=3)
    args = parser.parse_args()

    datasets = (
        ["swebench", "appworld"] if args.dataset == "both"
        else [args.dataset]
    )

    base_env = {
        "UTU_LLM_BASE_URL":    "http://localhost:8000/v1",
        "UTU_LLM_API_KEY":     "xxx",
        "JUDGE_LLM_TYPE":      "chat.completions",
        "JUDGE_LLM_MODEL":     args.extractor_model,
        "JUDGE_LLM_BASE_URL":  f"http://localhost:{args.extractor_port}/v1",
        "JUDGE_LLM_API_KEY":   "xxx",
    }

    for dataset in datasets:
        cfg = DATASET_CONFIGS[dataset]
        dataset_name  = cfg["dataset_name"]
        config_name   = cfg["config_name"]
        experiences_p = cfg["experiences_path"]

        log_dir = f"logs/shapley/{dataset}"
        os.makedirs(log_dir, exist_ok=True)
        shapley_path = f"{log_dir}/shapley_progress.json"

        print(f"\n{'#'*65}")
        print(f"# Dataset: {dataset_name}")
        print(f"{'#'*65}")

        # Step 1: Prepare and upload dataset
        if not args.skip_prepare:
            # Step 1a: Download dataset
            await run_step(
                f"Download {dataset_name}",
                f"python scripts/prepare_datasets.py --dataset {dataset} --no_upload",
                base_env,
            )
            # Step 1b: Upload using sys.executable so utu is always on sys.path
            jsonl_file = (
                "data/swebench_lite.jsonl"
                if dataset == "swebench"
                else f"data/{dataset}.jsonl"
            )
            await run_step(
                f"Upload {dataset_name} to GRPO database",
                f"{sys.executable} scripts/data/upload_dataset.py "
                f"--file_path {jsonl_file} "
                f"--dataset_name {dataset_name}",
                base_env,
            )
            # Create YAML configs only if they don't already exist
            if not args.skip_configs:
                print("\nCreating YAML configs...")
                create_configs(dataset, cfg, overwrite=args.overwrite_configs)
            else:
                print("\nSkipping config creation (--skip_configs set)")

        # Step 2: Run Training-Free GRPO (Qwen 7B rollout, DeepSeek extractor)
        if not args.skip_grpo:
            await run_step(
                f"Training-Free GRPO on {dataset_name} "
                f"(rollout=Qwen7B, extractor=DeepSeek32B)",
                f"python -m scripts.run_llm_judge "
                f"--config_name {config_name} "
                f"--iterative "
                f"--n_candidates 20 --n_keep 10 --n_refinement_rounds 2 "
                f"--extractor_port {args.extractor_port} "
                f"--extractor_model {args.extractor_model} "
                f"--restart_step 0",
                base_env,
            )

        # Step 3: Compute Shapley scores on held-out split
        if not args.skip_shapley:
            await run_step(
                f"Compute Shapley scores for {dataset_name}",
                f"python -m scripts.run_shapley "
                f"--config_name {config_name} "
                f"--experiences_path {experiences_p} "
                f"--m {args.m} --k {args.k} "
                f"--train_questions 100 --shapley_size 100 "
                f"--log_dir {log_dir}",
                base_env,
            )

        # Step 4: Compare experience configurations
        if not args.skip_compare:
            await run_step(
                f"Compare experience sets for {dataset_name}",
                f"python -m scripts.compare_experience_sets "
                f"--config_name {config_name} "
                f"--experiences_path {experiences_p} "
                f"--shapley_path {shapley_path} "
                f"--train_questions 100 --shapley_size 100 --eval_size 100 "
                f"--n_seeds {args.n_seeds} "
                f"--log_dir {log_dir}",
                base_env,
            )

        # Step 5: LLM judge ranking + Shapley comparison
        if not args.skip_compare:
            await run_step(
                f"LLM judge ranking for {dataset_name}",
                f"python -m scripts.run_llm_judge "
                f"--config_name {config_name} "
                f"--experiences_path {experiences_p} "
                f"--shapley_path {shapley_path} "
                f"--extractor_port {args.extractor_port} "
                f"--extractor_model {args.extractor_model} "
                f"--log_dir {log_dir}/llm_judge",
                base_env,
            )

        # Print summary
        _print_summary(dataset, log_dir, shapley_path)


def _print_summary(dataset: str, log_dir: str, shapley_path: str):
    print(f"\n{'='*65}")
    print(f"SUMMARY: {dataset}")
    print(f"{'='*65}")

    # Shapley results
    if os.path.exists(shapley_path):
        with open(shapley_path) as f:
            data = json.load(f)
        print(f"\n  Shapley rankings ({dataset}):")
        for e in data["ranked_experiences"][:5]:
            print(f"    #{e['rank']}  [{e['experience_id']}]  "
                  f"psi={e['psi_value']:+.5f}  "
                  f"\"{e['content'][:50]}\"")
        print(f"  ... (saved to {shapley_path})")

    # Comparison results
    compare_path = f"{log_dir}/comparison_results.json"
    if os.path.exists(compare_path):
        with open(compare_path) as f:
            data = json.load(f)
        print(f"\n  Performance comparison ({dataset}):")
        baseline = data["baseline_mean_at_1"]
        for key, r in data["results"].items():
            delta = r["mean_at_1"] - baseline
            delta_str = f"({delta:+.4f})" if key != "no_experiences" else ""
            print(f"    {r['label']:<45}  "
                  f"{r['mean_at_1']:.4f} ± {r.get('stderr', 0):.4f}  {delta_str}")

    # LLM judge results
    judge_path = f"{log_dir}/llm_judge/judge_standalone.json"
    if os.path.exists(judge_path):
        with open(judge_path) as f:
            data = json.load(f)
        rho = data.get("spearman_rho", "N/A")
        print(f"\n  Shapley vs LLM judge agreement ({dataset}):")
        print(f"    Spearman ρ = {rho}")
        print(f"    Saved to {judge_path}")

    print(f"\n  All results in: {log_dir}/")


if __name__ == "__main__":
    asyncio.run(main())