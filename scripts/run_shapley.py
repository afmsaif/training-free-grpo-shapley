"""
Step 1: Compute cardinality-restricted Shapley scores.

Uses the SHAPLEY split (questions T..T+S-1) to compute psi_i values.
Does NOT touch the EVAL split — that is reserved for compare_experience_sets.py.

Dataset layout:
    0   .. T-1     : TRAINING  (already used by GRPO)
    T   .. T+S-1   : SHAPLEY   (this script)
    T+S .. T+S+E-1 : EVAL      (compare_experience_sets.py)

Usage:
    python -m scripts.run_shapley \
        --config_name math_reasoning \
        --experiences_path configs/agents/practice/math_practice_agent.yaml \
        --m 10 --k 10 \
        --train_questions 100 \
        --shapley_size 100
"""

import argparse
import asyncio
import json
import logging
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


async def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Compute Shapley scores on the Shapley split"
    )
    parser.add_argument("--config_name", required=True)
    parser.add_argument("--experiences_path", required=True)
    parser.add_argument("--m", type=int, default=10,
                        help="Random subsets to sample (default 10)")
    parser.add_argument("--k", type=int, default=10,
                        help="Budget: number of experiences to select (default 10)")
    parser.add_argument("--train_questions", type=int, default=100,
                        help="Questions used during GRPO training (default 100)")
    parser.add_argument("--shapley_size", type=int, default=100,
                        help="Questions in Shapley estimation split (default 100)")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Questions per V(S) call (default 50)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", default="logs/shapley")
    args = parser.parse_args()

    sys.path.insert(0, ".")
    from utu.config.loader import ConfigLoader
    from utu.practice.experience_shapley_random import ExperienceShapley, make_value_fn

    config = ConfigLoader.load_training_free_grpo_config(args.config_name)
    experiences = _load_experiences(args.experiences_path)
    n = len(experiences)
    k = min(args.k, n)

    shapley_start = args.train_questions
    eval_start    = args.train_questions + args.shapley_size

    print(f"\n{'='*65}")
    print(f"STEP 1: Shapley Score Estimation")
    print(f"{'='*65}")
    print(f"  n experiences      : {n}")
    print(f"  k budget           : {k}")
    print(f"  m samples          : {args.m}")
    print(f"  Dataset layout:")
    print(f"    Training  : q0   .. q{args.train_questions-1}")
    print(f"    Shapley   : q{shapley_start} .. q{eval_start-1}   <- THIS SCRIPT")
    print(f"    Eval      : q{eval_start} .. q{eval_start+args.batch_size-1}   <- compare_experience_sets.py")
    print(f"  Est. V(S) calls    : ~{args.m * (1 + n - k)}")
    print(f"  Output             : {args.log_dir}/")
    print(f"{'='*65}\n")

    # Shapley value function — uses SHAPLEY split only
    shapley_value_fn = make_value_fn(
        base_eval_config=config.evaluation,
        batch_size=args.batch_size,
        question_start=shapley_start,
        split_name="shapley",
    )

    estimator = ExperienceShapley(
        experiences=experiences,
        value_fn=shapley_value_fn,
        m=args.m,
        k=k,
        seed=args.seed,
        log_dir=args.log_dir,
    )
    phi = await estimator.run()

    # Print final rankings
    print(f"\n{'='*65}")
    print(f"SHAPLEY RESULTS  (psi_i at budget k={k})")
    print(f"{'='*65}")
    helpful = [(e, v) for e, v in phi.items() if v > 0]
    harmful  = [(e, v) for e, v in phi.items() if v < 0]
    neutral  = [(e, v) for e, v in phi.items() if v == 0]
    for rank, (eid, val) in enumerate(phi.items(), 1):
        bar = ("+" if val >= 0 else "-") * min(int(abs(val) * 300), 25)
        preview = experiences[eid][:60].replace("\n", " ")
        print(f"  #{rank:2d} [{eid}]  psi={val:+.5f}  {bar:<25}  \"{preview}\"")
    print(f"\n  Helpful: {len(helpful)}  Harmful: {len(harmful)}  Neutral: {len(neutral)}")
    print(f"\nNow run compare_experience_sets.py to evaluate on the held-out EVAL split.")
    print(f"  --shapley_path {args.log_dir}/shapley_progress.json")
    print(f"  --train_questions {args.train_questions}")
    print(f"  --shapley_size {args.shapley_size}")


if __name__ == "__main__":
    asyncio.run(main())