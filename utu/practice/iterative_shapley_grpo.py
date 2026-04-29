"""
Iterative Shapley-Guided Experience Refinement for Training-Free GRPO.

Core idea
---------
Standard Training-Free GRPO generates experiences once per step, with no
feedback about which experiences help or hurt. This module adds an inner
refinement loop per step:

    Round 0: Run rollouts with current memory → generate M candidate experiences
    Round 1: Score candidates with quick Shapley → keep top-k positive ones
             Run rollouts with top-k injected → generate M new candidates
    Round 2: Score again → keep better top-k
             ...
    Final:   Save best experiences to memory, proceed to next step

Why this is better than a single generate-then-prune approach:
- After round 1, the agent prompt contains only helpful experiences
- Better prompt → better reasoning → better rollout trajectories
- Better trajectories → more informative experience updates
- This compounds: each round the signal quality improves

Design choices
--------------
quick_shapley_m : int (default 5)
    Number of random subsets for within-step Shapley scoring.
    Small enough to be fast (5 * n_candidates calls), large enough to
    give a directional signal. We use individual marginals V({e_i}) - V({})
    as a fast proxy when m is very small.

n_candidates : int (default 20)
    How many experiences to generate per round. More candidates → better
    selection pool but slower.

n_keep : int (default 10)
    How many experiences to keep after each scoring round.

n_refinement_rounds : int (default 2)
    How many generate→score→filter iterations per step.
    2 rounds is a good tradeoff: one pruning pass + one quality improvement.

quick_eval_batch_size : int (default 20)
    Questions per V(S) call during within-step scoring.
    Small for speed (not for final evaluation accuracy).
"""

import asyncio
import copy
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import yaml
from agents import custom_span, function_span, gen_trace_id, trace

from ..config import TrainingFreeGRPOConfig
from ..config.eval_config import DataConfig
from ..utils import DIR_ROOT, get_logger
from ..utils.experience_cache import ExperienceCache
from .data_manager import TrainingFreeGRPODataManager
from .experience_updater_shapley import ExperienceUpdater
from .experience_shapley_random import ExperienceShapley, make_value_fn
from .rollout_manager import RolloutManager
from .training_free_grpo import TrainingFreeGRPO
from .utils import TaskRecorder

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Quick within-step Shapley scorer
# ---------------------------------------------------------------------------

class QuickExperienceScorer:
    """
    Lightweight Shapley scorer for within-step candidate evaluation.

    Uses a small number of random subsets (quick_m) to estimate psi_i
    for each candidate experience. This is not as accurate as the full
    Shapley estimation but is fast enough to run within a training step.

    For very small quick_m (1-3), falls back to individual marginals:
        score_i = V({e_i}) - V({})
    which ignores interactions but is computed with just n+1 V(S) calls.
    """

    def __init__(
        self,
        base_eval_config,
        question_start: int,
        quick_eval_batch_size: int = 20,
        quick_m: int = 5,
        k: int = 10,
        step: int = 0,
        seed: int = 42,
    ):
        self.base_eval_config = base_eval_config
        self.question_start = question_start
        self.quick_eval_batch_size = quick_eval_batch_size
        self.quick_m = quick_m
        self.k = k
        self.step = step
        self.seed = seed

    async def score(
        self,
        candidate_experiences: dict[str, str],
        round_idx: int,
    ) -> dict[str, float]:
        """
        Score all candidate experiences using quick Shapley estimation.

        Returns {exp_id: psi_score} sorted descending.
        """
        n = len(candidate_experiences)
        if n == 0:
            return {}

        split_name = f"qscore_step{self.step}_r{round_idx}"
        value_fn = make_value_fn(
            base_eval_config=self.base_eval_config,
            batch_size=self.quick_eval_batch_size,
            question_start=self.question_start,
            split_name=split_name,
        )

        if self.quick_m <= 2 or n <= 3:
            # Ultra-fast fallback: individual marginals only
            # score_i = V({e_i}) - V({})
            logger.info(
                "QuickScorer: using individual marginals "
                "(quick_m=%d too small for subset sampling with n=%d)",
                self.quick_m, n,
            )
            scores = await self._individual_marginals(
                candidate_experiences, value_fn
            )
        else:
            # Cardinality-restricted Shapley with quick_m samples
            estimator = ExperienceShapley(
                experiences=candidate_experiences,
                value_fn=value_fn,
                m=self.quick_m,
                k=min(self.k, n),
                seed=self.seed + round_idx,
                log_dir=f"logs/quick_shapley/step{self.step}_round{round_idx}",
            )
            scores = await estimator.run()

        logger.info(
            "QuickScorer round %d: %d experiences scored  "
            "helpful=%d harmful=%d",
            round_idx, len(scores),
            sum(1 for v in scores.values() if v > 0),
            sum(1 for v in scores.values() if v < 0),
        )
        return scores

    async def _individual_marginals(
        self,
        experiences: dict[str, str],
        value_fn,
    ) -> dict[str, float]:
        """V({e_i}) - V({}) for each experience."""
        v_empty = await value_fn({})
        scores = {}
        for exp_id, content in experiences.items():
            v_single = await value_fn({exp_id: content})
            scores[exp_id] = v_single - v_empty
        return scores


# ---------------------------------------------------------------------------
# Iterative GRPO with Shapley refinement
# ---------------------------------------------------------------------------

class IterativeShapleyGRPO(TrainingFreeGRPO):
    """
    Training-Free GRPO with iterative within-step Shapley refinement.

    Parameters
    ----------
    config : TrainingFreeGRPOConfig
        Standard GRPO config.
    n_candidates : int
        Candidate experiences to generate per round (default 20).
    n_keep : int
        Experiences to keep after Shapley scoring (default 10).
    n_refinement_rounds : int
        Generate→score→filter rounds per step (default 2).
    quick_m : int
        Random subsets for quick within-step Shapley (default 5).
    quick_eval_batch_size : int
        Questions per V(S) call during scoring (default 20).
    shapley_question_start : int
        First question index for scoring eval set (default 100,
        i.e. after training questions).
    prune_threshold : float
        Experiences with psi below this are removed (default -0.01).
    external_shapley_path : str | None
        Path to pre-computed shapley_progress.json from a previous run.
        If provided, Option 1+2+3 are also applied using these scores.
    """

    def __init__(
        self,
        config: TrainingFreeGRPOConfig,
        n_candidates: int = 20,
        n_keep: int = 10,
        n_refinement_rounds: int = 2,
        quick_m: int = 5,
        quick_eval_batch_size: int = 20,
        shapley_question_start: int = 100,
        prune_threshold: float = -0.01,
        external_shapley_path: str = None,
    ):
        super().__init__(config)
        self.n_candidates = n_candidates
        self.n_keep = n_keep
        self.n_refinement_rounds = n_refinement_rounds
        self.quick_m = quick_m
        self.quick_eval_batch_size = quick_eval_batch_size
        self.shapley_question_start = shapley_question_start
        self.prune_threshold = prune_threshold

        # Load external Shapley scores if provided (for option 1+2+3)
        self.external_shapley: dict[str, float] = {}
        if external_shapley_path:
            try:
                with open(external_shapley_path, "r") as f:
                    data = json.load(f)
                self.external_shapley = {
                    e["experience_id"]: e["psi_value"]
                    for e in data["ranked_experiences"]
                }
                logger.info(
                    "Loaded %d external Shapley scores from %s",
                    len(self.external_shapley), external_shapley_path,
                )
            except FileNotFoundError:
                logger.warning(
                    "External Shapley file not found: %s", external_shapley_path
                )

        # Per-step refinement history for analysis
        self.refinement_log: list[dict] = []

    async def practice(self):
        """
        Override practice() with iterative refinement loop.

        Per step:
            Round 0: rollout + generate n_candidates experiences
            Round 1..n_refinement_rounds:
                - Score candidates with quick Shapley
                - Keep top n_keep positive-scoring ones
                - Re-run rollouts with survivors as agent context
                - Generate n_candidates new experiences
            Final: merge survivors into memory, prune harmful ones
        """
        for epoch in range(self.config.practice.epochs):
            logger.info("Epoch %d start", epoch)

            epoch_data = self.practice_rollout_manager.load_epoch_data(
                epoch,
                shuffle=self.config.practice.shuffle_data,
                truncate=self.config.practice.rollout_data_truncate,
            )

            num_batches = len(epoch_data) // (
                self.config.practice.batch_size * self.config.practice.grpo_n
            )

            for batch_idx in range(num_batches):
                step = epoch * num_batches + batch_idx
                logger.info("Step %d (epoch %d, batch %d)", step, epoch, batch_idx)

                step_trace_id = gen_trace_id()
                with trace(
                    f"[{self.recorder.experiment_name}] Step {step}",
                    trace_id=step_trace_id,
                ):
                    if self._should_use_cache(step):
                        cached = ExperienceCache.load_experiences(
                            experiment_name=self.recorder.experiment_name,
                            step=step,
                        )
                        if cached is not None:
                            logger.info("Step %d: using cached experiences", step)
                            self.recorder.experiences_update(cached)
                            continue

                    # ── Iterative refinement loop ──
                    best_experiences = await self._iterative_refinement_step(
                        epoch=epoch,
                        batch_idx=batch_idx,
                        step=step,
                    )

                    # ── Apply external Shapley pruning (Option 3) ──
                    if self.external_shapley:
                        best_experiences = self._prune_by_shapley(
                            best_experiences, self.external_shapley, step
                        )

                    # ── Assign final IDs and save ──
                    best_experiences = {
                        f"G{i}": exp
                        for i, exp in enumerate(best_experiences.values())
                    }
                    self.recorder.experiences_update(best_experiences)

                    ExperienceCache.save_experiences(
                        experiment_name=self.recorder.experiment_name,
                        step=step,
                        experiences=best_experiences,
                        epoch=epoch,
                        batch=batch_idx,
                    )
                    logger.info(
                        "Step %d complete. Final experiences: %d",
                        step, len(best_experiences),
                    )

                    # Save refinement log
                    self._save_refinement_log(step)

    async def _iterative_refinement_step(
        self,
        epoch: int,
        batch_idx: int,
        step: int,
    ) -> dict[str, str]:
        """
        Core iterative refinement for one step.

        Returns the best set of experiences found across all rounds.
        """
        # Track the best experiences found so far (by cumulative Shapley score)
        current_best: dict[str, str] = copy.deepcopy(
            self.recorder.experiences or {}
        )
        all_round_scores: list[dict] = []

        # Scorer using a held-out eval split
        scorer = QuickExperienceScorer(
            base_eval_config=self.config.evaluation,
            question_start=self.shapley_question_start,
            quick_eval_batch_size=self.quick_eval_batch_size,
            quick_m=self.quick_m,
            k=self.n_keep,
            step=step,
            seed=42 + step,
        )

        for round_idx in range(self.n_refinement_rounds + 1):
            logger.info(
                "Step %d Round %d/%d  "
                "current_memory=%d",
                step, round_idx, self.n_refinement_rounds,
                len(current_best),
            )

            # ── Rollout with current best experiences ──
            # Temporarily inject current_best into agent config
            with custom_span(f"Round {round_idx} rollout"):
                rollout_config = self._make_rollout_config_with_experiences(
                    current_best,
                    step=step,
                    round_idx=round_idx,
                )
                rollout_mgr = RolloutManager(
                    config=rollout_config,
                    batch_size=self.config.practice.batch_size,
                )
                # Load practice data, then compute safe batch_idx.
                # truncate=100, grpo_n=5, batch_size=50 → 2 batches max.
                # Wrap batch_idx to avoid IndexError when outer loop has
                # more steps than the truncated dataset has batches.
                all_data = rollout_mgr.load_epoch_data(
                    epoch=epoch,
                    shuffle=self.config.practice.shuffle_data,
                    truncate=self.config.practice.rollout_data_truncate,
                )
                samples_per_batch = (
                    self.config.practice.batch_size * self.config.practice.grpo_n
                )
                num_available = len(all_data) // samples_per_batch
                safe_batch_idx = batch_idx % max(num_available, 1)
                if safe_batch_idx != batch_idx:
                    logger.info(
                        "Step %d Round %d: batch_idx=%d → %d "
                        "(%d batches available)",
                        step, round_idx, batch_idx, safe_batch_idx, num_available,
                    )
                rollouts, stat = await rollout_mgr.main(
                    batch_idx=safe_batch_idx,
                    recorder=self.recorder,
                    use_cache=False,
                )
            logger.info("Round %d rollout stat: %s", round_idx, stat)

            # ── Generate n_candidates experiences from rollouts ──
            with custom_span(f"Round {round_idx} experience generation"):
                # Use a temporary recorder so we don't pollute main memory
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
                    shapley_scores=(
                        self.external_shapley if self.external_shapley else None
                    ),
                )
            logger.info(
                "Round %d generated %d candidate experiences",
                round_idx, len(candidates),
            )

            if not candidates:
                logger.warning("Round %d: no candidates generated, stopping.", round_idx)
                break

            # ── If not the last round: score candidates and filter ──
            if round_idx < self.n_refinement_rounds:
                with custom_span(f"Round {round_idx} Shapley scoring"):
                    scores = await scorer.score(candidates, round_idx)

                # Keep top n_keep positive-scoring experiences
                survivors = self._select_top_k(
                    candidates, scores, n_keep=self.n_keep,
                    prune_threshold=self.prune_threshold,
                )

                all_round_scores.append({
                    "round": round_idx,
                    "n_candidates": len(candidates),
                    "n_survivors": len(survivors),
                    "scores": {k: round(v, 5) for k, v in scores.items()},
                    "survivors": list(survivors.keys()),
                })
                logger.info(
                    "Round %d: %d candidates → %d survivors (psi≥%.3f)",
                    round_idx, len(candidates), len(survivors),
                    self.prune_threshold,
                )
                # Survivors become context for next round
                current_best = survivors
            else:
                # Last round: use all candidates (no more scoring)
                all_round_scores.append({
                    "round": round_idx,
                    "n_candidates": len(candidates),
                    "n_survivors": len(candidates),
                    "scores": {},
                    "survivors": list(candidates.keys()),
                })
                current_best = candidates

        self.refinement_log.append({
            "step": step,
            "n_refinement_rounds": self.n_refinement_rounds,
            "rounds": all_round_scores,
            "final_n_experiences": len(current_best),
        })

        return current_best

    def _select_top_k(
        self,
        candidates: dict[str, str],
        scores: dict[str, float],
        n_keep: int,
        prune_threshold: float,
    ) -> dict[str, str]:
        """
        Keep top n_keep experiences by Shapley score,
        filtering out those below prune_threshold.
        """
        # Filter to positive-ish scores first
        eligible = {
            eid: score for eid, score in scores.items()
            if score >= prune_threshold and eid in candidates
        }
        # Sort by score descending, keep top n_keep
        ranked = sorted(eligible.items(), key=lambda x: x[1], reverse=True)
        kept_ids = [eid for eid, _ in ranked[:n_keep]]
        return {eid: candidates[eid] for eid in kept_ids if eid in candidates}

    def _prune_by_shapley(
        self,
        experiences: dict[str, str],
        shapley_scores: dict[str, float],
        step: int,
    ) -> dict[str, str]:
        """Remove experiences with external psi < prune_threshold."""
        pruned = {
            k: v for k, v in experiences.items()
            if shapley_scores.get(k, 0.0) >= self.prune_threshold
        }
        n_removed = len(experiences) - len(pruned)
        if n_removed > 0:
            logger.info(
                "Step %d: external Shapley pruned %d harmful experiences. "
                "%d remaining.",
                step, n_removed, len(pruned),
            )
        return pruned

    def _make_rollout_config_with_experiences(
        self,
        experiences: dict[str, str],
        step: int = 0,
        round_idx: int = 0,
    ):
        """Build a rollout config with given experiences injected into agent prompt.

        Uses a unique exp_id per step+round to avoid the data manager returning
        cached AIME24 samples from a previous exp_id that shares the same name.
        """
        practice_config = self.config.evaluation.model_copy(deep=True)
        practice_config.pass_k = self.config.practice.grpo_n

        # Critical: unique exp_id prevents the DB from reusing AIME24 cached data
        practice_config.exp_id = (
            f"{self.recorder.experiment_name}_s{step}_r{round_idx}"
        )

        # Use practice dataset (DAPO-Math-17k), not AIME24
        practice_config.data = DataConfig(
            dataset=self.config.data.practice_dataset_name
        )
        practice_config.agent.model.model_settings.temperature = (
            self.config.practice.rollout_temperature
        )

        if experiences:
            experience_text = (
                "\n\nWhen solving problems, you MUST first carefully read "
                "and understand the helpful instructions and experiences:\n"
            )
            experience_text += "\n".join(
                [f"[{i}]. {e}" for i, e in experiences.items()]
            )
            current_instructions = (
                practice_config.agent.agent.instructions
                or "You are a helpful assistant."
            )
            practice_config.agent.agent.instructions = (
                current_instructions + experience_text
            )

        return practice_config

    def _save_refinement_log(self, step: int) -> None:
        """Save the refinement history for analysis."""
        os.makedirs("logs/refinement", exist_ok=True)
        path = f"logs/refinement/step_{step}.json"
        entry = next(
            (r for r in self.refinement_log if r["step"] == step), None
        )
        if entry:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)

    def _create_agent_config_with_experiences(
        self, experiences: dict[str, str]
    ) -> str:
        """
        Option 1: filter harmful experiences before injecting into agent YAML.
        Uses external Shapley scores if available.
        """
        if self.external_shapley:
            n_before = len(experiences)
            experiences = {
                k: v for k, v in experiences.items()
                if self.external_shapley.get(k, 0.0) >= self.prune_threshold
            }
            n_removed = n_before - len(experiences)
            if n_removed > 0:
                logger.info(
                    "Option 1 filter: removed %d harmful experiences "
                    "from final agent config. %d remaining.",
                    n_removed, len(experiences),
                )
        return super()._create_agent_config_with_experiences(experiences)