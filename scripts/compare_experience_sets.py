# """
# Step 2: Compare experience configurations on the held-out EVAL split.

# Evaluates four configurations on questions that were NEVER seen during
# Shapley estimation or GRPO training:

#   1. No experiences (baseline)
#   2. All 27 experiences
#   3. Positive-psi only  (harmful experiences removed)
#   4. Top-k by psi       (Shapley-optimal subset)

# Dataset layout:
#     0   .. T-1     : TRAINING  (used by GRPO)
#     T   .. T+S-1   : SHAPLEY   (used by run_shapley.py — NOT used here)
#     T+S .. T+S+E-1 : EVAL      (THIS SCRIPT ONLY)

# Run AFTER run_shapley.py has produced shapley_progress.json.

# Usage:
#     python -m scripts.compare_experience_sets \
#         --config_name math_reasoning \
#         --experiences_path configs/agents/practice/math_practice_agent.yaml \
#         --shapley_path logs/shapley/shapley_progress.json \
#         --train_questions 100 \
#         --shapley_size 100 \
#         --eval_size 100
# """

# import argparse
# import asyncio
# import json
# import logging
# import re
# import sys

# import yaml

# logging.basicConfig(
#     level=logging.WARNING,
#     format="%(asctime)s - %(levelname)s - %(message)s",
# )


# def _load_experiences(path: str) -> dict[str, str]:
#     with open(path, "r", encoding="utf-8") as f:
#         config = yaml.safe_load(f)
#     instructions = (
#         config.get("agent", {}).get("agent", {}).get("instructions", "")
#         or config.get("agent", {}).get("instructions", "")
#     )
#     matches = re.compile(
#         r"^\[([^\]]+)\]\.\s+(.+)$", re.MULTILINE
#     ).findall(instructions)
#     if not matches:
#         raise ValueError(f"No experiences found in {path}.")
#     return {eid: content for eid, content in matches}


# def _load_psi(path: str) -> dict[str, float]:
#     with open(path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     return {e["experience_id"]: e["psi_value"] for e in data["ranked_experiences"]}


# async def main():
#     parser = argparse.ArgumentParser(
#         description="Step 2: Compare experience sets on held-out eval split"
#     )
#     parser.add_argument("--config_name", required=True)
#     parser.add_argument("--experiences_path", required=True)
#     parser.add_argument("--shapley_path", required=True,
#                         help="Path to shapley_progress.json from run_shapley.py")
#     parser.add_argument("--train_questions", type=int, default=100)
#     parser.add_argument("--shapley_size",    type=int, default=100)
#     parser.add_argument("--eval_size",       type=int, default=100,
#                         help="Questions in the held-out eval split (default 100)")
#     parser.add_argument("--log_dir", default="logs/shapley")
#     args = parser.parse_args()

#     sys.path.insert(0, ".")
#     from utu.config.loader import ConfigLoader
#     from utu.practice.experience_shapley_random import make_value_fn

#     config     = ConfigLoader.load_training_free_grpo_config(args.config_name)
#     experiences = _load_experiences(args.experiences_path)
#     psi         = _load_psi(args.shapley_path)

#     # Only keep experiences present in both psi results and yaml
#     psi = {k: v for k, v in psi.items() if k in experiences}

#     eval_start = args.train_questions + args.shapley_size
#     eval_end   = eval_start + args.eval_size

#     print(f"\n{'='*65}")
#     print(f"STEP 2: Final Comparison on Held-Out Eval Split")
#     print(f"{'='*65}")
#     print(f"  Dataset layout:")
#     print(f"    Training : q0   .. q{args.train_questions-1}    (GRPO, not used here)")
#     print(f"    Shapley  : q{args.train_questions} .. q{eval_start-1}   (psi estimation, not used here)")
#     print(f"    Eval     : q{eval_start} .. q{eval_end-1}   <- THIS SCRIPT")
#     print(f"  n experiences total : {len(experiences)}")
#     print(f"  Shapley path        : {args.shapley_path}")
#     print(f"{'='*65}\n")

#     # Eval value function — uses EVAL split only, completely separate from Shapley
#     eval_value_fn = make_value_fn(
#         base_eval_config=config.evaluation,
#         batch_size=args.eval_size,
#         question_start=eval_start,
#         split_name="eval",
#     )

#     # Build the four subsets to compare
#     ranked_ids   = sorted(psi.keys(), key=lambda x: psi[x], reverse=True)
#     positive_ids = [eid for eid in ranked_ids if psi[eid] > 0]
#     negative_ids = [eid for eid in ranked_ids if psi[eid] < 0]

#     # Top-k: prefix of ranked_ids maximising cumulative psi sum
#     cumsum, best_k, best_sum = 0.0, 0, 0.0
#     for k_idx, eid in enumerate(ranked_ids, 1):
#         cumsum += psi[eid]
#         if cumsum > best_sum:
#             best_sum = cumsum
#             best_k = k_idx
#     top_k_ids = ranked_ids[:best_k]

#     subsets = {
#         "no_experiences": (
#             "No experiences (baseline)", {}
#         ),
#         "all_experiences": (
#             f"All {len(experiences)} experiences",
#             experiences
#         ),
#         "positive_psi": (
#             f"Positive-psi only ({len(positive_ids)} exp, {len(negative_ids)} removed)",
#             {eid: experiences[eid] for eid in positive_ids}
#         ),
#         "top_k_psi": (
#             f"Top-{best_k} by psi (Shapley-optimal)",
#             {eid: experiences[eid] for eid in top_k_ids}
#         ),
#     }

#     print("Subsets to evaluate:")
#     for key, (label, subset) in subsets.items():
#         print(f"  {label}")
#     print(f"\nEach evaluated on q{eval_start}..q{eval_end-1} "
#           f"({args.eval_size} held-out questions)\n")

#     results = {}
#     for key, (label, subset) in subsets.items():
#         print(f"\n--- Evaluating: {label} ---", flush=True)
#         score = await eval_value_fn(subset)
#         results[key] = {
#             "label": label,
#             "mean_at_1": score,
#             "n_experiences": len(subset),
#             "experience_ids": list(subset.keys()),
#         }

#     # Print results table
#     baseline = results["no_experiences"]["mean_at_1"]
#     print(f"\n{'='*65}")
#     print(f"RESULTS  (eval split: q{eval_start}..q{eval_end-1})")
#     print(f"{'='*65}")
#     print(f"  {'Configuration':<50}  {'Mean@1':>7}  {'vs baseline':>11}")
#     for key, r in results.items():
#         delta = r["mean_at_1"] - baseline
#         delta_str = f"{delta:+.4f}" if key != "no_experiences" else "—"
#         print(f"  {r['label']:<50}  {r['mean_at_1']:>7.4f}  {delta_str:>11}")

#     # Psi summary
#     print(f"\n  Psi rankings (computed on q{args.train_questions}..q{eval_start-1}):")
#     print(f"  {'Exp':>6}  {'psi':>8}  {'helpful?':>9}")
#     for eid in ranked_ids:
#         helpful = "helpful" if psi[eid] > 0 else ("harmful" if psi[eid] < 0 else "neutral")
#         print(f"  {eid:>6}  {psi[eid]:>8.5f}  {helpful:>9}")

#     # Save
#     output = {
#         "train_questions": args.train_questions,
#         "shapley_size":    args.shapley_size,
#         "eval_size":       args.eval_size,
#         "eval_split":      f"q{eval_start}..q{eval_end-1}",
#         "baseline_mean_at_1": baseline,
#         "psi_rankings":    psi,
#         "negative_psi":    negative_ids,
#         "positive_psi":    positive_ids,
#         "top_k":           top_k_ids,
#         "results":         results,
#     }
#     out_path = f"{args.log_dir}/comparison_results.json"
#     with open(out_path, "w", encoding="utf-8") as f:
#         json.dump(output, f, indent=2, ensure_ascii=False)
#     print(f"\n  Saved: {out_path}")

#     # Paper-ready table
#     print(f"\n{'='*65}")
#     print(f"PAPER TABLE")
#     print(f"{'='*65}")
#     for key, r in results.items():
#         delta = r["mean_at_1"] - baseline
#         delta_str = f"({delta:+.4f})" if key != "no_experiences" else ""
#         print(f"  {r['label']:<50}  {r['mean_at_1']:.4f}  {delta_str}")


# if __name__ == "__main__":
#     asyncio.run(main())

"""
Step 2: Compare experience configurations on the held-out EVAL split.

Runs each configuration n_seeds times and averages to reduce stochasticity.
Optionally sets temperature=0 for fully deterministic decoding.

Dataset layout:
    0   .. T-1     : TRAINING  (used by GRPO)
    T   .. T+S-1   : SHAPLEY   (run_shapley.py — NOT used here)
    T+S .. T+S+E-1 : EVAL      (THIS SCRIPT ONLY)

Usage:
    # 3 seeds, stochastic (recommended for paper)
    python -m scripts.compare_experience_sets \
        --config_name math_reasoning \
        --experiences_path configs/agents/practice/math_practice_agent.yaml \
        --shapley_path logs/shapley/shapley_progress.json \
        --train_questions 100 --shapley_size 100 --eval_size 100 \
        --n_seeds 3

    # Single run, deterministic (quick sanity check)
    python -m scripts.compare_experience_sets \
        ... --n_seeds 1 --temperature 0.0
"""

import argparse
import asyncio
import json
import logging
import math
import os
import re
import sys

import yaml

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def _load_experiences(path: str) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    instructions = (
        config.get("agent", {}).get("agent", {}).get("instructions", "")
        or config.get("agent", {}).get("instructions", "")
    )
    matches = re.compile(
        r"^\[([^\]]+)\]\.\s+(.+)$", re.MULTILINE
    ).findall(instructions)
    if not matches:
        raise ValueError(f"No experiences found in {path}.")
    return {eid: content for eid, content in matches}


def _load_psi(path: str) -> dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {e["experience_id"]: e["psi_value"] for e in data["ranked_experiences"]}


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _stderr(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var / len(xs))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", required=True)
    parser.add_argument("--experiences_path", required=True)
    parser.add_argument("--shapley_path", required=True)
    parser.add_argument("--train_questions", type=int, default=100)
    parser.add_argument("--shapley_size",    type=int, default=100)
    parser.add_argument("--eval_size",       type=int, default=100)
    parser.add_argument(
        "--n_seeds", type=int, default=3,
        help="Seeds to average over (default 3). "
             "Total V(S) calls = 4 * n_seeds.",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Override model temperature. "
             "0.0 = fully deterministic (n_seeds=1 is then sufficient). "
             "Default: use config value.",
    )
    parser.add_argument("--log_dir", default="logs/shapley")
    args = parser.parse_args()

    sys.path.insert(0, ".")
    from utu.config.loader import ConfigLoader
    from utu.practice.experience_shapley_random import make_value_fn

    config      = ConfigLoader.load_training_free_grpo_config(args.config_name)
    experiences = _load_experiences(args.experiences_path)
    psi         = _load_psi(args.shapley_path)
    psi         = {k: v for k, v in psi.items() if k in experiences}

    eval_start = args.train_questions + args.shapley_size
    eval_end   = eval_start + args.eval_size

    # Build subsets
    ranked_ids   = sorted(psi.keys(), key=lambda x: psi[x], reverse=True)
    positive_ids = [eid for eid in ranked_ids if psi[eid] > 0]
    negative_ids = [eid for eid in ranked_ids if psi[eid] < 0]

    # Top-k: greedy prefix maximising cumulative psi
    cumsum, best_k, best_sum = 0.0, 0, 0.0
    for k_idx, eid in enumerate(ranked_ids, 1):
        cumsum += psi[eid]
        if cumsum > best_sum:
            best_sum = cumsum
            best_k = k_idx
    top_k_ids = ranked_ids[:best_k]

    same_set = (set(positive_ids) == set(top_k_ids))

    subsets = {
        "no_experiences": (
            "No experiences (baseline)", {}
        ),
        "all_experiences": (
            f"All {len(experiences)} experiences", experiences
        ),
        "positive_psi": (
            f"Positive-psi only "
            f"({len(positive_ids)} exp, {len(negative_ids)} removed)",
            {eid: experiences[eid] for eid in positive_ids}
        ),
        "top_k_psi": (
            f"Top-{best_k} by psi (Shapley-optimal)",
            {eid: experiences[eid] for eid in top_k_ids}
        ),
    }

    # Determine effective temperature
    try:
        config_temp = config.evaluation.agent.model.model_settings.temperature
    except Exception:
        config_temp = None
    effective_temp = args.temperature if args.temperature is not None else config_temp

    print(f"\n{'='*65}")
    print(f"STEP 2: Multi-Seed Comparison on Held-Out Eval Split")
    print(f"{'='*65}")
    print(f"  Eval split      : q{eval_start}..q{eval_end-1}  ({args.eval_size} questions)")
    print(f"  n_seeds         : {args.n_seeds}  "
          f"(total V(S) calls: {4 * args.n_seeds})")
    print(f"  temperature     : {effective_temp}  "
          f"{'← deterministic' if effective_temp == 0 else '← stochastic'}")
    if same_set:
        print(f"  Note: positive_psi and top_{best_k} are the SAME set "
              f"— expect identical scores after averaging")
    if effective_temp == 0 and args.n_seeds > 1:
        print(f"  Warning: temperature=0 is deterministic; all seeds will give "
              f"identical scores. n_seeds=1 is sufficient.")
    print(f"{'='*65}\n")

    all_scores: dict[str, list[float]] = {key: [] for key in subsets}
    seeds = [42 + i * 17 for i in range(args.n_seeds)]

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'─'*65}")
        print(f"Seed {seed_idx+1}/{args.n_seeds}  (seed={seed})")
        print(f"{'─'*65}")

        # Unique split_name per seed to avoid DB exp_id conflicts
        eval_value_fn = make_value_fn(
            base_eval_config=config.evaluation,
            batch_size=args.eval_size,
            question_start=eval_start,
            split_name=f"eval_s{seed}",
            temperature_override=args.temperature,
        )

        for key, (label, subset) in subsets.items():
            print(f"\n  Evaluating: {label}...", flush=True)
            score = await eval_value_fn(subset)
            all_scores[key].append(score)
            print(f"  → Mean@1 = {score:.4f}")

    # Compute stats
    baseline_mean = _mean(all_scores["no_experiences"])

    print(f"\n{'='*65}")
    print(f"RESULTS  (mean ± SE, n={args.n_seeds} seeds, temp={effective_temp})")
    print(f"{'='*65}")
    print(f"\n  {'Configuration':<52}  "
          f"{'Mean@1':>8}  {'±SE':>6}  {'vs baseline':>11}  {'all scores'}")

    results = {}
    for key, (label, subset) in subsets.items():
        scores = all_scores[key]
        mean   = _mean(scores)
        se     = _stderr(scores)
        delta  = mean - baseline_mean
        delta_str = f"{delta:+.4f}" if key != "no_experiences" else "—"
        scores_str = "  ".join(f"{s:.4f}" for s in scores)
        print(f"  {label:<52}  {mean:>8.4f}  {se:>6.4f}  "
              f"{delta_str:>11}  [{scores_str}]")
        results[key] = {
            "label": label,
            "mean_at_1": mean,
            "stderr": se,
            "delta_vs_baseline": delta if key != "no_experiences" else 0.0,
            "all_scores": scores,
            "n_experiences": len(subset),
            "experience_ids": list(subset.keys()),
        }

    if same_set:
        residual = abs(
            _mean(all_scores["positive_psi"]) - _mean(all_scores["top_k_psi"])
        )
        print(f"\n  positive_psi == top_k (same set). "
              f"Residual gap after averaging: {residual:.4f} "
              f"{'(≈0 as expected with temp=0)' if effective_temp == 0 else '(stochastic noise)'}")

    # Save
    os.makedirs(args.log_dir, exist_ok=True)
    output = {
        "train_questions":    args.train_questions,
        "shapley_size":       args.shapley_size,
        "eval_size":          args.eval_size,
        "eval_split":         f"q{eval_start}..q{eval_end-1}",
        "n_seeds":            args.n_seeds,
        "temperature":        effective_temp,
        "baseline_mean_at_1": baseline_mean,
        "psi_rankings":       psi,
        "negative_psi":       negative_ids,
        "positive_psi":       positive_ids,
        "top_k":              top_k_ids,
        "same_set":           same_set,
        "results":            results,
    }
    out_path = os.path.join(args.log_dir, "comparison_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*65}")
    print(f"PAPER TABLE  (n={args.n_seeds} seeds, temp={effective_temp})")
    print(f"{'='*65}")
    for key, r in results.items():
        delta = r["mean_at_1"] - baseline_mean
        delta_str = f"({delta:+.4f})" if key != "no_experiences" else ""
        print(f"  {r['label']:<52}  "
              f"{r['mean_at_1']:.4f} ± {r['stderr']:.4f}  {delta_str}")

    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())