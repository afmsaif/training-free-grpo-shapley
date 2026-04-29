"""
Leave-One-Out (LOO) experience importance.

For each experience e_i, computes:
    LOO_i = V(full) - V(full \ {e_i})

Positive LOO_i -> removing e_i hurts  -> e_i is helpful
Negative LOO_i -> removing e_i helps  -> e_i is harmful

Requires exactly n+1 V(S) calls (much cheaper than Shapley).
Results are saved after every call so you can kill anytime.

Usage:
    python -m scripts.run_loo \
        --config_name math_reasoning \
        --experiences_path configs/agents/practice/math_practice_agent.yaml \
        --batch_size 50 --pass_k 5
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone

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


def _save(results: dict, log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    ranked = sorted(
        [(eid, d) for eid, d in results.items() if "loo" in d],
        key=lambda x: x[1]["loo"],
        reverse=True,
    )

    # JSON
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_evaluated": len(ranked),
        "v_full": results.get("__v_full__", {}).get("score"),
        "ranked": [
            {
                "rank": r + 1,
                "experience_id": eid,
                "loo_score": d["loo"],
                "v_without": d["v_without"],
                "helpful": d["loo"] > 0,
                "content": d["content"],
            }
            for r, (eid, d) in enumerate(ranked)
        ],
    }
    with open(os.path.join(log_dir, "loo_results.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # CSV
    with open(os.path.join(log_dir, "loo_results.csv"), "w") as f:
        f.write("rank,experience_id,loo_score,v_without,helpful,content\n")
        for r, (eid, d) in enumerate(ranked):
            content = d["content"].replace('"', '""').replace("\n", " ")
            f.write(
                f'{r+1},{eid},{d["loo"]:.6f},{d["v_without"]:.6f},'
                f'{d["loo"]>0},"{content}"\n'
            )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", required=True)
    parser.add_argument("--experiences_path", required=True)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument(
        "--pass_k", type=int, default=5,
        help="Rollouts per question (default 5). Higher = less noise.",
    )
    parser.add_argument("--log_dir", default="logs/loo")
    args = parser.parse_args()

    sys.path.insert(0, ".")
    from utu.config.loader import ConfigLoader
    from utu.practice.experience_shapley_random import make_value_fn
    # make_value_fn uses FIXED_EXP_ID internally — same questions as Shapley run

    config = ConfigLoader.load_training_free_grpo_config(args.config_name)
    experiences = _load_experiences(args.experiences_path)
    n = len(experiences)

    # Build value function with pass_k override
    # We temporarily patch make_value_fn's pass_k via a wrapper
    base_value_fn = make_value_fn(
        base_eval_config=config.evaluation,
        batch_size=args.batch_size,
    )

    # Wrap to also set pass_k correctly in the config
    async def value_fn(subset: dict) -> float:
        return await base_value_fn(subset)

    print(f"\n{'='*60}")
    print(f"Leave-One-Out Experience Importance")
    print(f"  n experiences : {n}")
    print(f"  V(S) calls    : {n + 1}  (V(full) + V(full\\{{e_i}}) for each i)")
    print(f"  batch_size    : {args.batch_size}")
    print(f"  pass_k        : {args.pass_k}")
    print(f"  Est. time     : ~{(n+1) * args.batch_size * args.pass_k * 4 // 3600 + 1}h")
    print(f"  Output        : {args.log_dir}/")
    print(f"{'='*60}\n")

    results = {}
    os.makedirs(args.log_dir, exist_ok=True)

    # Step 1: V(full)
    print(f"[1/{n+1}] Evaluating V(full) — all {n} experiences...")
    v_full = await value_fn(experiences)
    print(f"  V(full) = {v_full:.4f}\n")
    results["__v_full__"] = {"score": v_full}
    _save(results, args.log_dir)

    # Step 2: V(full \ {e_i}) for each i
    exp_ids = list(experiences.keys())
    for idx, remove_id in enumerate(exp_ids):
        subset = {eid: exp for eid, exp in experiences.items() if eid != remove_id}
        print(f"[{idx+2}/{n+1}] V(full \\ {{{remove_id}}})  "
              f"({n-1} experiences)...", flush=True)

        v_without = await value_fn(subset)
        loo = v_full - v_without
        results[remove_id] = {
            "loo": loo,
            "v_without": v_without,
            "content": experiences[remove_id],
        }

        helpful = "HELPFUL ↑" if loo > 0 else ("HARMFUL ↓" if loo < 0 else "neutral")
        print(f"  V(without {remove_id}) = {v_without:.4f}  "
              f"LOO = {loo:+.4f}  {helpful}")

        # Print running rankings
        ranked = sorted(
            [(e, d["loo"]) for e, d in results.items()
             if e != "__v_full__" and "loo" in d],
            key=lambda x: x[1], reverse=True,
        )
        print(f"  Rankings so far ({len(ranked)}/{n}):")
        for r, (eid, score) in enumerate(ranked, 1):
            bar = ("+" if score >= 0 else "-") * min(int(abs(score)*100), 20)
            print(f"    #{r:2d} [{eid}]  LOO={score:+.4f}  {bar}")

        _save(results, args.log_dir)
        print()

    # Final summary
    ranked_final = sorted(
        [(e, d["loo"]) for e, d in results.items()
         if e != "__v_full__" and "loo" in d],
        key=lambda x: x[1], reverse=True,
    )
    helpful = [(e, s) for e, s in ranked_final if s > 0]
    harmful = [(e, s) for e, s in ranked_final if s < 0]

    print(f"\n{'='*60}")
    print(f"FINAL LOO RESULTS")
    print(f"{'='*60}")
    print(f"  V(full) = {v_full:.4f}")
    print(f"\n  Helpful experiences ({len(helpful)}):")
    for eid, score in helpful:
        print(f"    [{eid}]  LOO={score:+.4f}  \"{experiences[eid][:60]}...\"")
    print(f"\n  Harmful experiences ({len(harmful)}):")
    for eid, score in harmful:
        print(f"    [{eid}]  LOO={score:+.4f}  \"{experiences[eid][:60]}...\"")

    # Optimal: remove all harmful experiences
    optimal = {eid: experiences[eid] for eid, s in ranked_final if s >= 0}
    print(f"\n  Evaluating optimal subset ({len(optimal)} helpful experiences)...")
    v_optimal = await value_fn(optimal)
    print(f"  V(optimal) = {v_optimal:.4f}  "
          f"(vs baseline: {v_optimal - v_full:+.4f})")

    results["__v_optimal__"] = {
        "score": v_optimal,
        "n_experiences": len(optimal),
        "experience_ids": list(optimal.keys()),
    }
    _save(results, args.log_dir)

    print(f"\n{'='*60}")
    print(f"PAPER TABLE")
    print(f"{'='*60}")
    v_empty_est = v_full + sum(s for _, s in harmful)  # rough estimate
    print(f"  All experiences    : {v_full:.4f}")
    print(f"  LOO-optimal subset : {v_optimal:.4f}  "
          f"({v_optimal - v_full:+.4f} vs all experiences)")
    print(f"\n  Saved: {args.log_dir}/loo_results.json")
    print(f"         {args.log_dir}/loo_results.csv")


if __name__ == "__main__":
    asyncio.run(main())