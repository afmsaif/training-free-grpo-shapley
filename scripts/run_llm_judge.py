"""
Run LLM-as-Judge experience ranking.

Two modes:

Mode 1 — Standalone ranking (comparison with Shapley scores)
    Given an agent YAML with experiences, ask the LLM to rank them.
    Compare the LLM ranking with the Shapley ranking side by side.

Mode 2 — Iterative GRPO with LLM judge (instead of Shapley)
    Same as run_iterative_shapley_grpo.py but uses LLM judge scores
    instead of Shapley for within-step experience selection.

Usage:
    # Mode 1: rank existing experiences
    python -m scripts.run_llm_judge \\
        --config_name math_reasoning \\
        --experiences_path configs/agents/practice/math_practice_agent.yaml \\
        --shapley_path logs/shapley/shapley_progress.json  # optional comparison

    # Mode 2: iterative GRPO with LLM judge
    python -m scripts.run_llm_judge \\
        --config_name math_reasoning \\
        --iterative \\
        --n_candidates 20 --n_keep 10 --n_refinement_rounds 2
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


def _load_shapley(path: str) -> dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {e["experience_id"]: e["psi_value"] for e in data["ranked_experiences"]}



def _make_extractor_config(config, extractor_port=None, extractor_model=None):
    """
    Build an AgentConfig that points the LLM client to the extractor server.

    The client (SimplifiedAsyncOpenAI) reads the base URL from:
      1. model_provider.base_url field  (if set in config)
      2. UTU_LLM_BASE_URL env variable  (fallback)

    Since model_dump() shows base_url is not stored in the config schema,
    we set UTU_LLM_EXTRACTOR_BASE_URL and UTU_LLM_EXTRACTOR_MODEL as env vars
    and patch model_provider directly so the ExperienceUpdater LLM client
    picks up the extractor server URL.

    If extractor_port is None, returns the original config unchanged.
    """
    import os
    import copy

    if extractor_port is None:
        return config.evaluation.agent

    extractor_base_url = f"http://localhost:{extractor_port}/v1"
    extractor_agent_config = copy.deepcopy(config.evaluation.agent)

    # model_provider may be a Pydantic model — set via attribute or __dict__
    provider = extractor_agent_config.model.model_provider
    try:
        # Try direct attribute set (works if field exists but is None)
        provider.base_url = extractor_base_url
    except (AttributeError, ValueError):
        pass

    try:
        # Fallback: set via __dict__ (bypasses Pydantic validation)
        provider.__dict__["base_url"] = extractor_base_url
    except Exception:
        pass

    # Also set env var as a belt-and-suspenders fallback —
    # SimplifiedAsyncOpenAI reads UTU_LLM_BASE_URL if base_url arg is None
    # We use a process-level env var that will be read by the extractor LLM
    # Note: this affects ALL LLM calls in this process, so we restore it after
    # building the extractor. The rollout agent uses the agents SDK which reads
    # a different env path, so it is not affected.
    os.environ["_EXTRACTOR_BASE_URL"] = extractor_base_url
    os.environ["_EXTRACTOR_MODEL"] = extractor_model or provider.model

    if extractor_model:
        try:
            provider.model = extractor_model
        except Exception:
            provider.__dict__["model"] = extractor_model

    return extractor_agent_config


async def run_standalone_judge(args, config, experiences):
    """Mode 1: rank existing experiences and compare with Shapley."""
    from utu.practice.experience_llm_judge import LLMExperienceJudge

    extractor_agent_config = _make_extractor_config(
        config, args.extractor_port, args.extractor_model
    )
    if args.extractor_port:
        print(f"Using extractor model at port {args.extractor_port} "
              f"({args.extractor_model or 'model from config'})")
    judge = LLMExperienceJudge(
        llm_config=extractor_agent_config,
        agent_objective=config.practice.agent_objective,
        learning_objective=config.practice.learning_objective,
        log_dir=args.log_dir,
    )

    print(f"\nAsking LLM to rank {len(experiences)} experiences...")
    print(f"This makes 1 LLM call (fast — no rollouts needed)\n")

    result = await judge.judge(
        experiences=experiences,
        rollouts=None,
        label="standalone",
    )

    # Print LLM judge results
    print(f"\n{'='*65}")
    print(f"LLM JUDGE RANKINGS")
    print(f"{'='*65}")
    print(f"\n  {result['reasoning']}\n")
    print(f"  {'Rank':>4}  {'Exp':>6}  {'Score':>7}  "
          f"{'psi_proxy':>10}  {'Remove':>7}  Content")
    print(f"  {'-'*80}")
    remove_set = set(result["remove_ids"])
    for rank, eid in enumerate(result["ranked_ids"], 1):
        score = result["scores"].get(eid, 5.0)
        psi   = result["psi_proxy"].get(eid, 0.0)
        flag  = "REMOVE" if eid in remove_set else ""
        preview = experiences.get(eid, "")[:55].replace("\n", " ")
        print(f"  {rank:>4}  [{eid:>4}]  {score:>7.1f}  "
              f"{psi:>10.4f}  {flag:>7}  \"{preview}\"")

    # Compare with Shapley if provided
    if args.shapley_path:
        shapley = _load_shapley(args.shapley_path)
        shapley = {k: v for k, v in shapley.items() if k in experiences}
        shapley_ranked = sorted(
            shapley.keys(), key=lambda x: shapley[x], reverse=True
        )

        print(f"\n{'='*65}")
        print(f"SHAPLEY vs LLM JUDGE COMPARISON")
        print(f"{'='*65}")
        print(f"\n  {'Exp':>6}  {'Shapley rank':>13}  {'LLM rank':>9}  "
              f"{'Shapley psi':>12}  {'LLM score':>10}  {'Agreement':>10}")
        print(f"  {'-'*75}")

        judge_rank_map = {
            eid: rank for rank, eid in enumerate(result["ranked_ids"], 1)
        }
        shapley_rank_map = {
            eid: rank for rank, eid in enumerate(shapley_ranked, 1)
        }

        agreements = []
        for eid in sorted(experiences.keys()):
            s_rank = shapley_rank_map.get(eid, "-")
            j_rank = judge_rank_map.get(eid, "-")
            s_psi  = shapley.get(eid, 0.0)
            j_score = result["scores"].get(eid, 5.0)

            # Agreement: both positive OR both negative
            s_positive = s_psi > 0
            j_positive = j_score > 5
            agree = "✓ agree" if s_positive == j_positive else "✗ disagree"
            agreements.append(s_positive == j_positive)

            print(f"  [{eid:>4}]  {str(s_rank):>13}  {str(j_rank):>9}  "
                  f"{s_psi:>12.5f}  {j_score:>10.1f}  {agree:>10}")

        agreement_pct = sum(agreements) / len(agreements) * 100
        print(f"\n  Agreement rate: {agreement_pct:.1f}% "
              f"({sum(agreements)}/{len(agreements)} experiences "
              f"classified the same way)")

        # Rank correlation (Spearman)
        n = len(experiences)
        common = [e for e in experiences if e in shapley_rank_map and e in judge_rank_map]
        if len(common) >= 3:
            d_sq = sum(
                (shapley_rank_map[e] - judge_rank_map[e]) ** 2
                for e in common
            )
            rho = 1 - 6 * d_sq / (n * (n**2 - 1))
            print(f"  Spearman rank correlation: ρ = {rho:.3f}")
            if rho > 0.7:
                print(f"  → Strong agreement: LLM judge and Shapley produce similar rankings")
            elif rho > 0.4:
                print(f"  → Moderate agreement: some alignment between methods")
            else:
                print(f"  → Weak agreement: methods disagree on rankings")

    print(f"\nOutputs saved to: {args.log_dir}/")
    print(f"  judge_standalone.json  — full judge output")
    print(f"  judge_standalone.csv   — ranked table")


async def run_iterative_with_judge(args, config):
    """Mode 2: iterative GRPO using LLM judge for within-step scoring."""
    from utu.practice.iterative_shapley_grpo import IterativeShapleyGRPO
    from utu.practice.experience_llm_judge import LLMJudgeScorer

    # Monkey-patch IterativeShapleyGRPO to use LLM judge instead of Shapley
    original_refinement = IterativeShapleyGRPO._iterative_refinement_step

    extractor_agent_config = _make_extractor_config(
        config, args.extractor_port, args.extractor_model
    )
    if args.extractor_port:
        print(f"Rollout agent    : port 8000 (Qwen2.5-7B)")
        print(f"Experience judge : port {args.extractor_port} "
              f"({args.extractor_model or 'extractor model'})")
    else:
        print(f"Single-model mode: both rollout and judge use port 8000")

    # LLMJudgeScorer builds a SimplifiedAsyncOpenAI client.
    # Temporarily set env var so it picks up port 8001.
    if args.extractor_port:
        import os
        _prev = os.environ.get("UTU_LLM_BASE_URL")
        os.environ["UTU_LLM_BASE_URL"] = f"http://localhost:{args.extractor_port}/v1"
        os.environ["UTU_LLM_API_KEY"] = "xxx"

    judge_scorer = LLMJudgeScorer(
        llm_config=extractor_agent_config,
        agent_objective=config.practice.agent_objective,
        learning_objective=config.practice.learning_objective,
        log_dir=args.log_dir,
    )

    if args.extractor_port:
        if _prev is not None:
            os.environ["UTU_LLM_BASE_URL"] = _prev
        else:
            os.environ.pop("UTU_LLM_BASE_URL", None)

    async def refinement_with_judge(self, epoch, batch_idx, step):
        """Override inner loop to use LLM judge for scoring."""
        import copy
        from agents import custom_span
        from utu.practice.experience_shapley_random import make_value_fn
        from utu.practice.rollout_manager import RolloutManager
        from utu.practice.utils import TaskRecorder
        from utu.utils import get_logger as _gl

        _logger = _gl(__name__)
        current_best = copy.deepcopy(self.recorder.experiences or {})
        all_round_scores = []

        for round_idx in range(self.n_refinement_rounds + 1):
            _logger.info(
                "Step %d Round %d/%d (LLM judge mode)",
                step, round_idx, self.n_refinement_rounds,
            )

            # Rollout
            with custom_span(f"Round {round_idx} rollout"):
                rollout_config = self._make_rollout_config_with_experiences(
                    current_best, step=step, round_idx=round_idx
                )
                rollout_mgr = RolloutManager(
                    config=rollout_config,
                    batch_size=self.config.practice.batch_size,
                )
                all_data = rollout_mgr.load_epoch_data(
                    epoch=epoch,
                    shuffle=self.config.practice.shuffle_data,
                    truncate=self.config.practice.rollout_data_truncate,
                )
                # Compute number of available batches and wrap batch_idx
                # to avoid IndexError when batch_idx exceeds dataset size.
                # With truncate=100, grpo_n=5, batch_size=50:
                #   500 samples / (50*5) = 2 batches → batch_idx 0 and 1 only
                samples_per_batch = (
                    self.config.practice.batch_size * self.config.practice.grpo_n
                )
                num_available = len(all_data) // samples_per_batch
                safe_batch_idx = batch_idx % max(num_available, 1)
                if safe_batch_idx != batch_idx:
                    _logger.info(
                        "Step %d Round %d: batch_idx=%d wrapped to %d "
                        "(only %d batches available with truncate=%d)",
                        step, round_idx, batch_idx, safe_batch_idx,
                        num_available,
                        self.config.practice.rollout_data_truncate,
                    )
                rollouts, stat = await rollout_mgr.main(
                    batch_idx=safe_batch_idx,
                    recorder=self.recorder,
                    use_cache=False,
                )

            # Generate candidates
            with custom_span(f"Round {round_idx} generation"):
                temp_recorder = TaskRecorder(
                    experiment_name=f"{self.recorder.experiment_name}_r{round_idx}"
                )
                temp_recorder.experiences = copy.deepcopy(current_best)
                candidates = await self.experience_updater.run(
                    rollouts=rollouts,
                    recorder=temp_recorder,
                    concurrency=self.config.practice.rollout_concurrency,
                    given_ground_truth=self.config.practice.given_ground_truth,
                    num_experiences=self.n_candidates,
                )

            if not candidates:
                break

            if round_idx < self.n_refinement_rounds:
                # Score with LLM judge instead of Shapley
                with custom_span(f"Round {round_idx} LLM judge scoring"):
                    scores = await judge_scorer.score(
                        candidate_experiences=candidates,
                        round_idx=round_idx,
                        rollouts=rollouts,
                        step=step,
                    )

                survivors = self._select_top_k(
                    candidates, scores,
                    n_keep=self.n_keep,
                    prune_threshold=self.prune_threshold,
                )
                all_round_scores.append({
                    "round": round_idx,
                    "scorer": "llm_judge",
                    "n_candidates": len(candidates),
                    "n_survivors": len(survivors),
                    "scores": {k: round(v, 5) for k, v in scores.items()},
                })
                current_best = survivors
            else:
                all_round_scores.append({
                    "round": round_idx,
                    "scorer": "none (final round)",
                    "n_candidates": len(candidates),
                    "n_survivors": len(candidates),
                    "scores": {},
                })
                current_best = candidates

        self.refinement_log.append({
            "step": step,
            "scorer": "llm_judge",
            "rounds": all_round_scores,
            "final_n_experiences": len(current_best),
        })
        return current_best

    # Patch the method
    IterativeShapleyGRPO._iterative_refinement_step = refinement_with_judge

    # Override experience_updater to use the stronger extractor model.
    # SimplifiedAsyncOpenAI reads base_url from its constructor arg.
    # We build the updater with a patched config AND pass base_url explicitly
    # by temporarily setting UTU_LLM_BASE_URL so the client picks it up.
    if args.extractor_port:
        import os
        from utu.practice.experience_updater_shapley import ExperienceUpdater
        from utu.utils import SimplifiedAsyncOpenAI

        extractor_base_url = f"http://localhost:{args.extractor_port}/v1"
        extractor_model_name = args.extractor_model or config.evaluation.agent.model.model_provider.model

        # Temporarily override env var so SimplifiedAsyncOpenAI picks up port 8001
        _prev_base_url = os.environ.get("UTU_LLM_BASE_URL")
        _prev_api_key  = os.environ.get("UTU_LLM_API_KEY")
        os.environ["UTU_LLM_BASE_URL"] = extractor_base_url
        os.environ["UTU_LLM_API_KEY"]  = "xxx"  # vLLM doesn't need a real key

        extractor_updater = ExperienceUpdater(
            config=extractor_agent_config,
            agent_objective=config.practice.agent_objective,
            learning_objective=config.practice.learning_objective,
        )

        # Restore env vars — rollout agent must keep using port 8000
        if _prev_base_url is not None:
            os.environ["UTU_LLM_BASE_URL"] = _prev_base_url
        else:
            del os.environ["UTU_LLM_BASE_URL"]
        if _prev_api_key is not None:
            os.environ["UTU_LLM_API_KEY"] = _prev_api_key
        else:
            os.environ.pop("UTU_LLM_API_KEY", None)

        print(f"  ExperienceUpdater → port {args.extractor_port} ({extractor_model_name})")
    else:
        extractor_updater = None  # will use default from build()

    config.practice.restart_step = args.restart_step
    grpo = IterativeShapleyGRPO(
        config=config,
        n_candidates=args.n_candidates,
        n_keep=args.n_keep,
        n_refinement_rounds=args.n_refinement_rounds,
        quick_m=0,   # not used — LLM judge handles scoring
        quick_eval_batch_size=0,
        prune_threshold=args.prune_threshold,
        external_shapley_path=args.external_shapley_path,
    )

    print(f"\n{'='*65}")
    print(f"Iterative GRPO with LLM Judge")
    print(f"{'='*65}")
    print(f"  n_candidates        : {args.n_candidates}")
    print(f"  n_keep              : {args.n_keep}")
    print(f"  n_refinement_rounds : {args.n_refinement_rounds}")
    print(f"  Scoring method      : LLM-as-judge (1 API call per round)")
    print(f"  vs Shapley method   : ~{args.quick_m * args.n_candidates} V(S) calls per round")
    print(f"{'='*65}\n")

    # If using extractor model, override the experience_updater after build()
    if extractor_updater is not None:
        await grpo.build()
        grpo.experience_updater = extractor_updater
        agent_config_path = await grpo.run()
    else:
        agent_config_path = await grpo.run()
    print(f"\nDone. Agent config: {agent_config_path}")
    print(f"Judge logs: {args.log_dir}/")


async def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge experience ranking"
    )
    parser.add_argument("--config_name", required=True)
    parser.add_argument("--log_dir", default="logs/llm_judge")

    # Mode 1 args
    parser.add_argument("--experiences_path", default=None,
                        help="Agent YAML path (Mode 1: standalone ranking)")
    parser.add_argument("--shapley_path", default=None,
                        help="shapley_progress.json for comparison (optional)")

    # Mode 2 args
    parser.add_argument("--iterative", action="store_true",
                        help="Run iterative GRPO with LLM judge (Mode 2)")
    parser.add_argument("--restart_step", type=int, default=0)
    parser.add_argument("--n_candidates", type=int, default=20)
    parser.add_argument("--n_keep", type=int, default=10)
    parser.add_argument("--n_refinement_rounds", type=int, default=2)
    parser.add_argument("--quick_m", type=int, default=5,
                        help="Used only for budget display comparison")
    parser.add_argument("--prune_threshold", type=float, default=-0.2,
                        help="LLM psi_proxy threshold for pruning (default -0.2, "
                             "= LLM score < 4/10)")
    parser.add_argument("--external_shapley_path", default=None)
    parser.add_argument(
        "--extractor_port", type=int, default=None,
        help="Port of the stronger model for experience extraction and LLM judge. "
             "e.g. 8001 for DeepSeek-R1-Distill-Qwen-32B on GPU 2,3. "
             "If not set, uses the same model as the rollout agent (port 8000).",
    )
    parser.add_argument(
        "--extractor_model", type=str, default=None,
        help="Model name for the extractor (e.g. deepseek-ai/DeepSeek-R1-Distill-Qwen-32B). "
             "Only needed if different from the rollout model.",
    )

    args = parser.parse_args()

    sys.path.insert(0, ".")
    from utu.config.loader import ConfigLoader
    config = ConfigLoader.load_training_free_grpo_config(args.config_name)

    import os
    os.makedirs(args.log_dir, exist_ok=True)

    if args.iterative:
        await run_iterative_with_judge(args, config)
    else:
        if not args.experiences_path:
            parser.error(
                "Mode 1 requires --experiences_path. "
                "Use --iterative for Mode 2."
            )
        experiences = _load_experiences(args.experiences_path)
        await run_standalone_judge(args, config, experiences)


if __name__ == "__main__":
    asyncio.run(main())