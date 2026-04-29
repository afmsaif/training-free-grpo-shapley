"""
Quick Shapley experiment — results printed live after each subset.

Time estimates (batch_size=3, n=11):
    m=5   ->  ~33 min
    m=10  ->  ~65 min
    m=20  ->  ~130 min

Usage
-----
# Quickest possible (5 subsets, 3 questions each)
python -m scripts.run_shapley_random \
    --config_name math_reasoning \
    --experiences_path configs/agents/practice/math_practice_agent.yaml

# More subsets for stability check
python -m scripts.run_shapley_random --m 10

# Kill anytime — intermediate results saved after every subset in log_dir/
"""

import argparse
import asyncio
import json
import logging
import re
import sys

import yaml

logging.basicConfig(
    level=logging.WARNING,          # suppress noisy INFO logs from rollout
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def _load_experiences(path: str) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    instructions = (
        config.get("agent", {}).get("agent", {}).get("instructions", "")
        or config.get("agent", {}).get("instructions", "")
    )
    matches = re.compile(r"^\[([^\]]+)\]\.\s+(.+)$", re.MULTILINE).findall(instructions)
    if not matches:
        raise ValueError(f"No experiences found in {path}.")
    experiences = {eid: content for eid, content in matches}
    print(f"Loaded {len(experiences)} experiences from {path}")
    return experiences


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", required=True)
    parser.add_argument("--experiences_path", required=True)
    parser.add_argument(
        "--m", type=int, default=10,
        help="Number of random subsets to sample (default 10). "
             "Each sample gives exactly (n-k) marginals. "
             "Kill anytime — results saved after every V(S) call.",
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="Budget: target number of experiences to select (default 10). "
             "Each subset S has |S|=k-1. "
             "Set to roughly half your total experiences for best coverage.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=50,
        help="Questions per V(S) evaluation (default 50). "
             "Higher = less noise but slower.",
    )
    parser.add_argument(
        "--train_questions", type=int, default=100,
        help="Number of questions used during GRPO training (rollout_data_truncate). "
             "Shapley eval uses the immediately following batch as the test split. "
             "Default 100 matches rollout_data_truncate: 100 in math_reasoning.yaml.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", default="logs/shapley_random")
    args = parser.parse_args()

    sys.path.insert(0, ".")
    from utu.config.loader import ConfigLoader
    from utu.practice.experience_shapley_random import run_random_shapley

    config = ConfigLoader.load_training_free_grpo_config(args.config_name)
    experiences = _load_experiences(args.experiences_path)

    n = len(experiences)
    k = min(args.k, n)
    est_calls = args.m * (1 + n - k)
    from math import comb
    total_subsets_at_k = comb(n - 1, k - 1)

    print(f"\nSettings:")
    print(f"  Estimator     : Cardinality-Restricted Shapley (psi_i at budget k)")
    print(f"  n experiences : {n}")
    print(f"  k budget      : {k}  (select top-k experiences)")
    print(f"  |S| per call  : {k-1}  (fixed — always k-1)")
    print(f"  m samples     : {args.m}  "
          f"({100*args.m/total_subsets_at_k:.3f}% of all C(n-1,k-1)={total_subsets_at_k} subsets)")
    print(f"  Marginals/exp : {args.m}  (exactly m per experience)")
    print(f"  batch_size    : {args.batch_size} questions per V(S) call")
    print(f"  Est. V(S)     : ~{est_calls} calls")
    print(f"  Est. time     : ~{est_calls} min  (kill anytime, saves after every call)")
    print(f"  Output dir    : {args.log_dir}/\n")

    phi = await run_random_shapley(
        experiences=experiences,
        base_eval_config=config.evaluation,
        m=args.m,
        k=k,
        batch_size=args.batch_size,
        train_questions=args.train_questions,
        seed=args.seed,
        log_dir=args.log_dir,
    )

    print(f"\n{'='*60}")
    print(f"FINAL RANKINGS  (m={args.m} subsets)")
    print(f"{'='*60}")
    for rank, (eid, val) in enumerate(phi.items(), 1):
        bar = ("+" if val >= 0 else "-") * min(int(abs(val) * 300), 25)
        preview = experiences[eid][:65].replace("\n", " ")
        print(f"  #{rank:2d} [{eid}]  psi={val:+.5f}  {bar:<25}  \"{preview}\"")

    helpful = [(e, v) for e, v in phi.items() if v > 0]
    harmful  = [(e, v) for e, v in phi.items() if v < 0]
    print(f"\n  Helpful experiences: {len(helpful)}")
    print(f"  Harmful experiences: {len(harmful)}  "
          f"({[e for e,_ in harmful]})")
    print(f"\nAll results in: {args.log_dir}/")
    print("  shapley_progress.json  — latest full rankings")
    print("  shapley_progress.csv   — latest CSV (open in Excel)")
    print("  shapley_history.jsonl  — full history per V(S) call")


if __name__ == "__main__":
    asyncio.run(main())