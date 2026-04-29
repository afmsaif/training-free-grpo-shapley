"""
Sanity check before running Shapley experiment.

Evaluates V(empty) and V(full experiences) to verify there is measurable
signal. If both return the same score, Shapley values will all be zero
and the experiment is not viable.

Usage:
    python -m scripts.check_shapley_signal \
        --config_name math_reasoning \
        --experiences_path configs/agents/practice/math_practice_agent.yaml \
        --batch_size 50
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
    matches = re.compile(r"^\[([^\]]+)\]\.\s+(.+)$", re.MULTILINE).findall(instructions)
    if not matches:
        raise ValueError(f"No experiences found in {path}.")
    return {eid: content for eid, content in matches}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", required=True)
    parser.add_argument("--experiences_path", required=True)
    parser.add_argument("--batch_size", type=int, default=50)
    args = parser.parse_args()

    sys.path.insert(0, ".")
    from utu.config.loader import ConfigLoader
    from utu.practice.experience_shapley_random import make_value_fn

    config = ConfigLoader.load_training_free_grpo_config(args.config_name)
    experiences = _load_experiences(args.experiences_path)
    value_fn = make_value_fn(
        base_eval_config=config.evaluation,
        batch_size=args.batch_size,
    )

    print(f"\n{'='*60}")
    print(f"Shapley Signal Check")
    print(f"  batch_size = {args.batch_size} questions per call")
    print(f"  n experiences = {len(experiences)}")
    print(f"{'='*60}\n")

    # --- V(empty) ---
    print("Step 1/3: Evaluating V(empty) — no experiences injected...")
    v_empty = await value_fn({})
    print(f"  V(empty) = {v_empty:.4f}\n")

    # --- V(single best) — just first experience as a quick check ---
    first_id = list(experiences.keys())[0]
    print(f"Step 2/3: Evaluating V({{{first_id}}}) — one experience...")
    v_one = await value_fn({first_id: experiences[first_id]})
    print(f"  V({{{first_id}}}) = {v_one:.4f}  "
          f"(marginal = {v_one - v_empty:+.4f})\n")

    # --- V(full) ---
    print(f"Step 3/3: Evaluating V(full) — all {len(experiences)} experiences...")
    v_full = await value_fn(experiences)
    print(f"  V(full)  = {v_full:.4f}  "
          f"(marginal vs empty = {v_full - v_empty:+.4f})\n")

    print("="*60)
    print("VERDICT")
    print("="*60)

    gap = abs(v_full - v_empty)
    if gap < 0.01:
        print(f"\n  V(empty)={v_empty:.4f}  V(full)={v_full:.4f}  gap={gap:.4f}")
        print("\n  ⚠  NO SIGNAL: V(full) ≈ V(empty).")
        print("  Shapley values will all be ~0 — experiment not viable.")
        print("\n  Possible causes:")
        print("  1. batch_size too small → model scores 0% on all sampled questions")
        print("     → Try: --batch_size 100 or 200")
        print("  2. Wrong dataset — check config points to math_reasoning, not AIME")
        print("  3. Model not using experiences — check instructions are injected correctly")
        print("  4. Model performance is 0% regardless of experiences on this dataset")
    elif v_empty == 0.0:
        print(f"\n  V(empty)=0.0  V(full)={v_full:.4f}  gap={gap:.4f}")
        print("\n  ⚠  V(empty)=0: model scores 0% with no experiences.")
        print("  Shapley experiment will work but all values will be positive.")
        print("  Recommendation: proceed with Shapley — signal exists.")
    else:
        print(f"\n  V(empty)={v_empty:.4f}  V(full)={v_full:.4f}  gap={gap:.4f}")
        print("\n  ✓  SIGNAL EXISTS: proceed with Shapley experiment.")
        print(f"  Expected φ range: roughly ±{gap/len(experiences):.4f} per experience")

    # Save result
    result = {
        "v_empty": v_empty,
        "v_one_experience": {first_id: v_one},
        "v_full": v_full,
        "gap": v_full - v_empty,
        "batch_size": args.batch_size,
        "n_experiences": len(experiences),
        "signal_exists": gap >= 0.01,
    }
    with open("logs/shapley_signal_check.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved to logs/shapley_signal_check.json")


if __name__ == "__main__":
    asyncio.run(main())
