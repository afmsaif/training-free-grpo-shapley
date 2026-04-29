"""
Experience Shapley: Equitable valuation of experiences for Training-free GRPO.

Adapts Data Shapley (Ghorbani & Zou, ICML 2019) to rank the experiences
produced by Training-free GRPO.

Mapping from the paper to this setting:
  - Players         = individual experiences (strings in memory dict)
  - V(S)            = Pass@1 (≈ Mean@k) of the agent when only experiences
                      in subset S are injected into its system prompt
  - Shapley value i = average marginal contribution of experience i to Pass@1

Estimator
---------
We use **Stratified Subset Sampling** (Castro et al., 2009; Maleki et al., 2013)
rather than full TMC-Shapley permutation sampling.

The Shapley value decomposes as a weighted average over subset sizes:

    φ_i = (1/n) * Σ_{s=0}^{n-1}  E_{S~Uniform(size-s subsets of N\\{i})}
                                    [ V(S∪{i}) - V(S) ]

We estimate each stratum s independently by drawing `samples_per_stratum`
random subsets, and repeat in rounds until φ estimates stabilise.

Why stratified sampling instead of TMC permutations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Castro et al. (2009) proved that stratified sampling achieves strictly
  lower variance than naive Monte Carlo permutation sampling for the same
  number of V(S) evaluations.
- Middle strata (s ≈ n/2) have the highest variance in marginal contributions;
  the stratified approach lets us allocate equal samples per stratum rather
  than letting permutation sampling under-sample them.
- For small n (you have n=11 experiences), the total budget is very manageable:
  worst case 2 * n * n * samples_per_stratum * max_rounds V(S) calls, but a
  V(S) cache eliminates repeated evaluations of identical subsets across
  rounds and strata, reducing actual calls by ~40-60%.
- Convergence is checked per-round: we stop as soon as mean|Δφ| < eps for
  `patience` consecutive rounds, typically round 3-5 for n=11.

Literature
----------
Castro, J., Gomez, D., & Tejada, J. (2009). Polynomial calculation of the
  Shapley value based on sampling. Computers & Operations Research, 36(5).
Maleki, S. et al. (2013). Bounding the estimation error of sampling-based
  Shapley value approximation. arXiv:1306.4265.
Ghorbani, A. & Zou, J. (2019). Data Shapley: Equitable Valuation of Data
  for Machine Learning. ICML 2019.
Lundberg, S. & Lee, S-I. (2017). A unified approach to interpreting model
  predictions. NeurIPS. [KernelSHAP uses a similar weighted subset scheme.]
"""

import asyncio
import copy
import json
import logging
import os
import random
from datetime import datetime, timezone
from typing import Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stratified Subset Sampling Shapley estimator
# ---------------------------------------------------------------------------

class ExperienceShapley:
    """
    Estimate Shapley values via stratified subset sampling.

    For each stratum s in {0, ..., n-1} and each experience i, we draw
    `samples_per_stratum` random subsets S of size s not containing i,
    evaluate V(S∪{i}) - V(S), and accumulate a running mean.

    Rounds continue until mean absolute change in φ < convergence_eps
    for `patience` consecutive rounds, or max_rounds is reached.

    Parameters
    ----------
    experiences : dict[str, str]
        Experience memory from Training-free GRPO.
    value_fn : Callable[[dict[str, str]], Awaitable[float]]
        Async function V(subset) -> Pass@1 score.
    samples_per_stratum : int
        Random subsets per (stratum s, experience i) per round.
        Start with 3; increase to 5 if estimates remain noisy after
        convergence (check rounds.jsonl to diagnose).
    max_rounds : int
        Hard cap on rounds. For n=11, samples_per_stratum=3:
        worst-case V(S) calls = 2*11*11*3*max_rounds (before cache).
    convergence_eps : float
        Convergence threshold on mean|Δφ|.  0.01 is appropriate for
        Pass@1 scores that are themselves noisy at the ±0.02 level.
    patience : int
        Consecutive stable rounds required before stopping.
    v_cache_size : int
        Max number of V(S) results to cache. Identical subsets arising
        across strata/rounds are served from cache without re-evaluation.
    seed : int
        Random seed for reproducibility.
    log_dir : str
        Directory for rounds.jsonl and shapley_values.json.
    """

    def __init__(
        self,
        experiences: dict[str, str],
        value_fn: Callable,
        samples_per_stratum: int = 3,
        max_rounds: int = 10,
        convergence_eps: float = 0.01,
        patience: int = 2,
        v_cache_size: int = 512,
        seed: int = 42,
        log_dir: str = "logs/shapley",
    ):
        self.experiences = experiences
        self.exp_ids = list(experiences.keys())
        self.n = len(self.exp_ids)
        self.value_fn = value_fn
        self.samples_per_stratum = samples_per_stratum
        self.max_rounds = max_rounds
        self.convergence_eps = convergence_eps
        self.patience = patience
        self.seed = seed
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # V(S) cache keyed by frozenset of experience indices
        self._v_cache: dict[frozenset, float] = {}
        self._v_cache_size = v_cache_size
        self._v_calls_total = 0
        self._v_calls_cached = 0

        # Accumulated marginals: _marginals[i_idx][s] = [list of observed V(S∪i)-V(S)]
        self._marginals: list[list[list[float]]] = [
            [[] for _ in range(self.n)] for _ in range(self.n)
        ]

        self.phi: dict[str, float] = {eid: 0.0 for eid in self.exp_ids}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> dict[str, float]:
        """
        Run stratified subset sampling until convergence or max_rounds.

        Returns
        -------
        dict[str, float]
            {experience_id: shapley_value} sorted descending by value.
        """
        rng = random.Random(self.seed)
        phi_prev = {eid: 0.0 for eid in self.exp_ids}
        consecutive_stable = 0

        logger.info(
            "Stratified Shapley: n=%d, samples_per_stratum=%d, max_rounds=%d, "
            "convergence_eps=%.4f, patience=%d",
            self.n, self.samples_per_stratum, self.max_rounds,
            self.convergence_eps, self.patience,
        )

        for round_idx in range(1, self.max_rounds + 1):
            logger.info("=== Round %d / %d ===", round_idx, self.max_rounds)

            await self._sample_round(rng)
            self.phi = self._compute_phi()

            mean_abs_change = self._mean_abs_change(self.phi, phi_prev)
            self._log_round(round_idx, phi_prev, mean_abs_change)

            logger.info(
                "Round %d | mean|Δφ|=%.5f | V(S) calls=%d (cached=%d)",
                round_idx, mean_abs_change,
                self._v_calls_total, self._v_calls_cached,
            )

            # Log current rankings so you can watch convergence in terminal
            ranked_preview = sorted(
                self.phi.items(), key=lambda x: x[1], reverse=True
            )
            for rank, (eid, val) in enumerate(ranked_preview, 1):
                logger.info("  #%2d  [%s]  φ=%+.5f", rank, eid, val)

            if mean_abs_change < self.convergence_eps:
                consecutive_stable += 1
                if consecutive_stable >= self.patience:
                    logger.info(
                        "Converged at round %d (%d V(S) calls, %d from cache).",
                        round_idx, self._v_calls_total, self._v_calls_cached,
                    )
                    break
            else:
                consecutive_stable = 0

            phi_prev = dict(self.phi)

        ranked = dict(
            sorted(self.phi.items(), key=lambda x: x[1], reverse=True)
        )
        self._save_final(ranked)
        return ranked

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    async def _sample_round(self, rng: random.Random) -> None:
        """
        Draw new subsets for every (stratum s, experience i) pair,
        collect all unique V(S) evaluations needed, run them concurrently
        (using cache), and accumulate marginals.
        """
        # Build evaluation plan: list of (frozen_S, frozen_S_with_i, i_idx, s)
        plan: list[tuple[frozenset, frozenset, int, int]] = []
        for s in range(self.n):
            for i_idx in range(self.n):
                pool = [j for j in range(self.n) if j != i_idx]
                if s > len(pool):
                    continue   # stratum impossible for this experience
                for _ in range(self.samples_per_stratum):
                    chosen = frozenset(rng.sample(pool, s))
                    plan.append((chosen, chosen | {i_idx}, i_idx, s))

        # Collect all unique subsets not yet cached
        needed: set[frozenset] = set()
        for fs_without, fs_with, _, _ in plan:
            if fs_without not in self._v_cache:
                needed.add(fs_without)
            if fs_with not in self._v_cache:
                needed.add(fs_with)

        # Evaluate sequentially — each V(S) call is a heavy rollout batch.
        # Firing them concurrently would overwhelm vLLM and cause DB
        # exp_id collisions.  The cache absorbs duplicates across rounds.
        for fs in needed:
            score = await self._eval_subset(fs)
            self._store_cache(fs, score)

        # Accumulate marginals from plan
        for fs_without, fs_with, i_idx, s in plan:
            marginal = self._v_cache[fs_with] - self._v_cache[fs_without]
            self._marginals[i_idx][s].append(marginal)

    async def _eval_subset(self, fs: frozenset) -> float:
        self._v_calls_total += 1
        subset_dict = {
            self.exp_ids[i]: self.experiences[self.exp_ids[i]] for i in fs
        }
        return await self.value_fn(subset_dict)

    def _store_cache(self, fs: frozenset, score: float) -> None:
        if len(self._v_cache) >= self._v_cache_size:
            evict = next(iter(self._v_cache))
            del self._v_cache[evict]
        self._v_cache[fs] = score

    # ------------------------------------------------------------------
    # φ computation  (Castro et al. 2009 Eq. 3)
    # ------------------------------------------------------------------

    def _compute_phi(self) -> dict[str, float]:
        """
        φ_i = (1/n) * Σ_{s=0}^{n-1}  mean( marginals[i][s] )

        Strata with no observations yet are excluded from the average
        (they contribute 0 weight in the running estimate).
        """
        phi = {}
        for i_idx, exp_id in enumerate(self.exp_ids):
            stratum_means = []
            for s in range(self.n):
                obs = self._marginals[i_idx][s]
                if obs:
                    stratum_means.append(sum(obs) / len(obs))
            phi[exp_id] = sum(stratum_means) / self.n if stratum_means else 0.0
        return phi

    def _mean_abs_change(self, phi_new: dict, phi_old: dict) -> float:
        changes = [abs(phi_new[eid] - phi_old.get(eid, 0.0)) for eid in self.exp_ids]
        return sum(changes) / len(changes) if changes else 0.0

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_round(self, round_idx: int, phi_prev: dict, mean_abs_change: float) -> None:
        stratum_counts = {
            f"s={s}": sum(len(self._marginals[i][s]) for i in range(self.n))
            for s in range(self.n)
        }
        line = {
            "round": round_idx,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mean_abs_phi_change": mean_abs_change,
            "phi": dict(self.phi),
            "phi_change_per_experience": {
                eid: round(abs(self.phi[eid] - phi_prev.get(eid, 0.0)), 6)
                for eid in self.exp_ids
            },
            "v_calls_total": self._v_calls_total,
            "v_calls_cached": self._v_calls_cached,
            "stratum_sample_counts": stratum_counts,
        }
        path = os.path.join(self.log_dir, "rounds.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    def _save_final(self, ranked: dict[str, float]) -> None:
        path = os.path.join(self.log_dir, "shapley_values.json")
        payload = {
            "estimator": "stratified_subset_sampling",
            "references": [
                "Castro et al. (2009) Computers & Operations Research 36(5)",
                "Maleki et al. (2013) arXiv:1306.4265",
                "Ghorbani & Zou (2019) ICML",
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_experiences": self.n,
            "samples_per_stratum": self.samples_per_stratum,
            "v_calls_total": self._v_calls_total,
            "v_calls_cached": self._v_calls_cached,
            "total_value_explained": sum(ranked.values()),
            "ranked_experiences": [
                {
                    "rank": rank + 1,
                    "experience_id": exp_id,
                    "shapley_value": sv,
                    "content": self.experiences[exp_id],
                    # Per-stratum means for diagnostic inspection
                    "stratum_means": [
                        round(
                            sum(self._marginals[self.exp_ids.index(exp_id)][s]) /
                            len(self._marginals[self.exp_ids.index(exp_id)][s]), 6
                        ) if self._marginals[self.exp_ids.index(exp_id)][s] else None
                        for s in range(self.n)
                    ],
                }
                for rank, (exp_id, sv) in enumerate(ranked.items())
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("Shapley values saved to %s", path)


# ---------------------------------------------------------------------------
# Gradual addition experiment
# ---------------------------------------------------------------------------

class GradualAdditionExperiment:
    """
    Evaluate Pass@1 as experiences are added one-by-one in order of
    decreasing Shapley value vs. random order.
    """

    def __init__(
        self,
        experiences: dict[str, str],
        shapley_values: dict[str, float],
        value_fn: Callable,
        seed: int = 42,
        log_dir: str = "logs/shapley",
    ):
        self.experiences = experiences
        self.shapley_values = shapley_values
        self.value_fn = value_fn
        self.seed = seed
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    async def run(self) -> dict:
        ranked_ids = sorted(
            self.shapley_values.keys(),
            key=lambda x: self.shapley_values[x],
            reverse=True,
        )

        shapley_curve = await self._eval_cumulative(ranked_ids, label="shapley")

        rng = random.Random(self.seed)
        all_ids = list(self.experiences.keys())
        random_curves = []
        for trial in range(5):
            shuffled = all_ids[:]
            rng.shuffle(shuffled)
            curve = await self._eval_cumulative(shuffled, label=f"random_{trial}")
            random_curves.append(curve)

        avg_random = [
            {
                "num_experiences": step + 1,
                "pass_at_1": sum(c[step]["pass_at_1"] for c in random_curves) / 5,
            }
            for step in range(len(all_ids))
        ]

        v_empty = await self.value_fn({})
        v_full = await self.value_fn(self.experiences)

        best_step = max(shapley_curve, key=lambda x: x["pass_at_1"])

        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_experiences": len(self.experiences),
            "baselines": {"v_empty": v_empty, "v_full": v_full},
            "shapley_optimal_subset": {
                "num_experiences": best_step["num_experiences"],
                "pass_at_1": best_step["pass_at_1"],
                "experiences": [ranked_ids[i] for i in range(best_step["num_experiences"])],
            },
            "shapley_order_curve": shapley_curve,
            "random_order_avg_curve": avg_random,
            "shapley_values": self.shapley_values,
            "ranked_experience_contents": [
                {
                    "rank": i + 1,
                    "id": exp_id,
                    "shapley_value": self.shapley_values[exp_id],
                    "content": self.experiences[exp_id],
                }
                for i, exp_id in enumerate(ranked_ids)
            ],
        }

        path = os.path.join(self.log_dir, "gradual_addition.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Gradual addition results saved to %s", path)
        return results

    async def _eval_cumulative(self, ordered_ids: list[str], label: str) -> list[dict]:
        curve = []
        current_subset: dict[str, str] = {}
        for i, exp_id in enumerate(ordered_ids):
            current_subset[exp_id] = self.experiences[exp_id]
            score = await self.value_fn(copy.copy(current_subset))
            curve.append({
                "num_experiences": i + 1,
                "exp_id": exp_id,
                "shapley_value": self.shapley_values.get(exp_id),
                "pass_at_1": score,
                "label": label,
            })
            logger.info(
                "[%s] k=%d  exp=%s  φ=%.4f  Pass@1=%.4f",
                label, i + 1, exp_id,
                self.shapley_values.get(exp_id, 0.0), score,
            )
        return curve


# ---------------------------------------------------------------------------
# Value function factory
# ---------------------------------------------------------------------------

def make_value_fn(base_eval_config, batch_size: int = 50) -> Callable:
    """
    Build async V(subset) -> Pass@1.
    Injects only the experiences in `subset` into the agent system prompt,
    runs one rollout batch, returns Mean@k as Pass@1 estimate.
    """
    from .rollout_manager import RolloutManager
    from .utils import TaskRecorder

    async def value_fn(subset: dict[str, str]) -> float:
        eval_config = base_eval_config.model_copy(deep=True)

        if subset:
            experience_text = (
                "\n\nWhen solving problems, you MUST first carefully read "
                "and understand the helpful instructions and experiences:\n"
            )
            experience_text += "\n".join(
                [f"[{i}]. {e}" for i, e in subset.items()]
            )
            current_instructions = (
                eval_config.agent.agent.instructions
                or "You are a helpful assistant."
            )
            eval_config.agent.agent.instructions = (
                current_instructions + experience_text
            )

        # Global counter for unique exp_ids — avoids DB collisions when
        # value_fn is called rapidly in succession.
        if not hasattr(value_fn, "_call_counter"):
            value_fn._call_counter = 0
        value_fn._call_counter += 1
        call_id = value_fn._call_counter

        # Build a unique, deterministic exp_id from the subset contents
        subset_key = "_".join(sorted(subset.keys())) if subset else "empty"
        exp_name = f"shapley_v{call_id}_{subset_key}"[:64]  # DB name length limit

        call_recorder = TaskRecorder(experiment_name=exp_name)

        rollout_mgr = RolloutManager(config=eval_config, batch_size=batch_size)
        # truncate=batch_size ensures we only evaluate `batch_size` samples
        # per V(S) call instead of the full dataset (which was 960 samples).
        rollout_mgr.load_epoch_data(epoch=0, shuffle=False, truncate=batch_size)
        _, stats = await rollout_mgr.main(
            batch_idx=None,
            recorder=call_recorder,
            use_cache=False,
        )
        return stats.get("Mean@5", stats.get("Mean@1", 0.0))

    return value_fn


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

async def run_experience_shapley_experiment(
    experiences: dict[str, str],
    base_eval_config,
    batch_size: int = 50,
    samples_per_stratum: int = 3,
    max_rounds: int = 10,
    convergence_eps: float = 0.01,
    patience: int = 2,
    seed: int = 42,
    log_dir: str = "logs/shapley",
    skip_shapley: bool = False,
    precomputed_shapley: dict[str, float] | None = None,
) -> dict:
    """
    Full pipeline: estimate Shapley values, then run gradual addition experiment.

    Estimated V(S) call budget (n=11 experiences, default params):
        Worst case (no cache, max_rounds=10):
            2 * 11 * 11 * 3 * 10 = 7,260 calls
        Typical (cache + early convergence at round ~4):
            ~600 - 1,200 calls
        Each call = one full rollout batch evaluation.
    """
    value_fn = make_value_fn(base_eval_config=base_eval_config, batch_size=batch_size)

    if skip_shapley and precomputed_shapley is not None:
        logger.info("Using precomputed Shapley values, skipping estimation.")
        shapley_values = precomputed_shapley
    else:
        logger.info(
            "Estimating Shapley values: n=%d experiences, "
            "samples_per_stratum=%d, max_rounds=%d, eps=%.4f",
            len(experiences), samples_per_stratum, max_rounds, convergence_eps,
        )
        estimator = ExperienceShapley(
            experiences=experiences,
            value_fn=value_fn,
            samples_per_stratum=samples_per_stratum,
            max_rounds=max_rounds,
            convergence_eps=convergence_eps,
            patience=patience,
            seed=seed,
            log_dir=log_dir,
        )
        shapley_values = await estimator.run()

    logger.info("Running gradual addition experiment...")
    experiment = GradualAdditionExperiment(
        experiences=experiences,
        shapley_values=shapley_values,
        value_fn=value_fn,
        seed=seed,
        log_dir=log_dir,
    )
    gradual_results = await experiment.run()

    return {
        "shapley_values": shapley_values,
        "gradual_addition_results": gradual_results,
    }