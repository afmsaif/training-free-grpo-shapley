"""
Run Experience Shapley valuation + gradual addition experiment.

Uses stratified subset sampling (Castro et al. 2009) to estimate Shapley
values efficiently without enumerating full permutations.

Usage examples
--------------
# Full run with defaults (n=11 experiences, typically ~600-1200 V(S) calls)
python -m scripts.run_experience_shapley \
    --config_name math_reasoning \
    --experiences_path configs/agents/practice/math_practice_agent.yaml

# Faster run: fewer samples per stratum, tighter convergence window
python -m scripts.run_experience_shapley \
    --config_name math_reasoning \
    --experiences_path configs/agents/practice/math_practice_agent.yaml \
    --samples_per_stratum 2 \
    --max_rounds 6 \
    --convergence_eps 0.015

# Skip Shapley recomputation, reuse saved values for gradual addition only
python -m scripts.run_experience_shapley \
    --config_name math_reasoning \
    --experiences_path configs/agents/practice/math_practice_agent.yaml \
    --skip_shapley \
    --shapley_values_path logs/shapley/shapley_values.json
"""

import argparse
import asyncio
import json
import logging
import re
import sys

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s[%(name)s] - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _load_experiences_from_agent_yaml(path: str) -> dict[str, str]:
    """
    Extract experiences injected into the agent instructions by
    TrainingFreeGRPO._create_agent_config_with_experiences().

    Parses lines like:
        [G0]. Always verify the answer by ...
        [G1]. When the problem involves ...
    """
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    instructions = (
        config.get("agent", {}).get("agent", {}).get("instructions", "")
        or config.get("agent", {}).get("instructions", "")
    )

    pattern = re.compile(r"^\[([^\]]+)\]\.\s+(.+)$", re.MULTILINE)
    matches = pattern.findall(instructions)

    if not matches:
        raise ValueError(
            f"No experiences found in {path}. "
            "Expected lines like '[G0]. experience content...'"
        )

    experiences = {exp_id: content for exp_id, content in matches}
    logger.info("Loaded %d experiences from %s", len(experiences), path)
    return experiences


async def main():
    parser = argparse.ArgumentParser(
        description="Experience Shapley valuation + gradual addition experiment"
    )
    parser.add_argument(
        "--config_name", required=True,
        help="Hydra config name (e.g. math_reasoning)",
    )
    parser.add_argument(
        "--experiences_path", required=True,
        help="Path to agent YAML with injected experiences",
    )
    parser.add_argument(
        "--batch_size", type=int, default=50,
        help="Rollout batch size for each V(S) evaluation (default: 50)",
    )
    parser.add_argument(
        "--samples_per_stratum", type=int, default=3,
        help="Random subsets per (stratum s, experience i) per round. "
             "Default 3. Increase to 5 if estimates are noisy.",
    )
    parser.add_argument(
        "--max_rounds", type=int, default=10,
        help="Max sampling rounds (default: 10). "
             "Typically converges at round 3-5 for n=11 experiences.",
    )
    parser.add_argument(
        "--convergence_eps", type=float, default=0.01,
        help="Convergence: stop when mean|Δφ| < eps (default: 0.01). "
             "0.01 is appropriate given Pass@1 noise of ~±0.02.",
    )
    parser.add_argument(
        "--patience", type=int, default=2,
        help="Consecutive stable rounds before convergence (default: 2)",
    )
    parser.add_argument(
        "--log_dir", default="logs/shapley",
        help="Output directory (default: logs/shapley)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--skip_shapley", action="store_true",
        help="Skip Shapley estimation; use --shapley_values_path instead",
    )
    parser.add_argument(
        "--shapley_values_path", default=None,
        help="Path to precomputed shapley_values.json (with --skip_shapley)",
    )
    args = parser.parse_args()

    sys.path.insert(0, ".")
    from utu.config.loader import ConfigLoader
    from utu.practice.experience_shapley import run_experience_shapley_experiment

    config = ConfigLoader.load_training_free_grpo_config(args.config_name)
    base_eval_config = config.evaluation

    experiences = _load_experiences_from_agent_yaml(args.experiences_path)

    precomputed_shapley = None
    if args.skip_shapley:
        if args.shapley_values_path is None:
            parser.error("--skip_shapley requires --shapley_values_path")
        with open(args.shapley_values_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        precomputed_shapley = {
            entry["experience_id"]: entry["shapley_value"]
            for entry in saved["ranked_experiences"]
        }
        logger.info(
            "Loaded %d precomputed Shapley values from %s",
            len(precomputed_shapley), args.shapley_values_path,
        )

    # Estimate budget before starting
    n = len(experiences)
    worst_case = 2 * n * n * args.samples_per_stratum * args.max_rounds
    logger.info(
        "V(S) call budget: worst case=%d, expected (cache+early stop)=%d-%d",
        worst_case, worst_case // 6, worst_case // 3,
    )

    results = await run_experience_shapley_experiment(
        experiences=experiences,
        base_eval_config=base_eval_config,
        batch_size=args.batch_size,
        samples_per_stratum=args.samples_per_stratum,
        max_rounds=args.max_rounds,
        convergence_eps=args.convergence_eps,
        patience=args.patience,
        seed=args.seed,
        log_dir=args.log_dir,
        skip_shapley=args.skip_shapley,
        precomputed_shapley=precomputed_shapley,
    )

    # --- Terminal summary ---
    print("\n" + "=" * 65)
    print("EXPERIENCE SHAPLEY RESULTS  (stratified subset sampling)")
    print("=" * 65)

    sv = results["shapley_values"]
    ranked = sorted(sv.items(), key=lambda x: x[1], reverse=True)
    print(f"\nRanked experiences ({len(ranked)} total):")
    for rank, (exp_id, val) in enumerate(ranked, 1):
        preview = experiences[exp_id][:75].replace("\n", " ")
        sign = "+" if val >= 0 else ""
        print(f"  #{rank:2d}  [{exp_id}]  φ={sign}{val:.5f}  \"{preview}...\"")

    ga = results["gradual_addition_results"]
    v_empty = ga["baselines"]["v_empty"]
    v_full  = ga["baselines"]["v_full"]
    optimal = ga["shapley_optimal_subset"]

    print(f"\nBaselines:")
    print(f"  V(∅)    = {v_empty:.4f}  (no experiences)")
    print(f"  V(full) = {v_full:.4f}  (all {len(experiences)} experiences)")
    print(f"\nShapley-optimal subset:")
    print(f"  k = {optimal['num_experiences']} experiences  →  Pass@1 = {optimal['pass_at_1']:.4f}")
    print(f"  Experiences: {optimal['experiences']}")

    print(f"\nGradual addition curve:")
    print(f"  {'k':>3}  {'Shapley':>10}  {'Random(avg)':>12}  {'Δ vs Full':>10}")
    for s, r in zip(ga["shapley_order_curve"], ga["random_order_avg_curve"]):
        k   = s["num_experiences"]
        dv  = s["pass_at_1"] - v_full
        print(f"  {k:3d}  {s['pass_at_1']:10.4f}  {r['pass_at_1']:12.4f}  {dv:+10.4f}")

    print(f"\nAll results saved to: {args.log_dir}/")
    print("  shapley_values.json    — ranked Shapley values + per-stratum diagnostics")
    print("  gradual_addition.json  — full gradual addition curves")
    print("  rounds.jsonl           — per-round convergence log")


if __name__ == "__main__":
    asyncio.run(main())