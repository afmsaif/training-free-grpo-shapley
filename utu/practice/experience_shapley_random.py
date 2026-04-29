# """
# Experience Shapley: Cardinality-Restricted Shapley Values.

# Dataset layout (DAPO-Math-17k, shuffle=False):
#     questions 0   .. T-1        : TRAINING  (used by GRPO to generate experiences)
#     questions T   .. T+S-1      : SHAPLEY   (used to compute psi_i values)
#     questions T+S .. T+S+E-1    : EVAL      (used only for final comparison)

# where:
#     T = train_questions  (default 100, = rollout_data_truncate in config)
#     S = shapley_size     (default 100, questions for computing Shapley scores)
#     E = eval_size        (default 100, questions for final 4-way comparison)

# This guarantees:
#   - Shapley scores are not inflated by evaluating on training questions
#   - Final comparison is not inflated by evaluating on Shapley-estimation questions
#   - All four configurations in the comparison (no-exp, all-exp, positive-exp,
#     top-k-exp) are evaluated on the identical held-out eval split
# """

# import asyncio
# import json
# import logging
# import os
# import random
# from datetime import datetime, timezone
# from typing import Callable

# logger = logging.getLogger(__name__)


# # ---------------------------------------------------------------------------
# # Value function factory
# # ---------------------------------------------------------------------------

# def make_value_fn(
#     base_eval_config,
#     batch_size: int = 50,
#     question_start: int = 0,
#     split_name: str = "eval",
# ) -> Callable:
#     """
#     Build an async V(subset) -> Mean@1 function that evaluates on a fixed
#     slice [question_start : question_start+batch_size] of DAPO-Math-17k.

#     Parameters
#     ----------
#     base_eval_config : EvalConfig
#         Base evaluation config from TrainingFreeGRPOConfig.
#     batch_size : int
#         Number of questions per evaluation call.
#     question_start : int
#         First question index in the slice (0-based).
#         Set to train_questions for Shapley split.
#         Set to train_questions+shapley_size for eval split.
#     split_name : str
#         Label used in exp_id and log messages ("shapley" or "eval").
#     """
#     from .rollout_manager import RolloutManager
#     from .utils import TaskRecorder

#     question_end   = question_start + batch_size
#     load_truncate  = question_end           # load exactly enough to cover the slice
#     test_batch_idx = question_start // batch_size

#     # Unique exp_id encodes the exact slice — no DB conflicts across splits or runs
#     FIXED_EXP_ID = f"shapley_{split_name}_q{question_start}_q{question_end}"

#     call_counter = {"n": 0}

#     logger.info(
#         "[make_value_fn:%s] questions %d..%d  exp_id=%s",
#         split_name, question_start, question_end - 1, FIXED_EXP_ID,
#     )

#     async def value_fn(subset: dict[str, str]) -> float:
#         eval_config = base_eval_config.model_copy(deep=True)

#         # Use DAPO-Math-17k (practice dataset), not AIME24
#         eval_config.data.dataset = "DAPO-Math-17k"
#         eval_config.pass_k = 1           # one rollout per question -> Mean@1
#         eval_config.exp_id = FIXED_EXP_ID

#         if subset:
#             experience_text = (
#                 "\n\nWhen solving problems, you MUST first carefully read "
#                 "and understand the helpful instructions and experiences:\n"
#             )
#             experience_text += "\n".join(
#                 [f"[{i}]. {e}" for i, e in subset.items()]
#             )
#             instructions = (
#                 eval_config.agent.agent.instructions
#                 or "You are a helpful assistant."
#             )
#             eval_config.agent.agent.instructions = instructions + experience_text

#         call_counter["n"] += 1
#         subset_key = "_".join(sorted(subset.keys())) if subset else "empty"
#         logger.info(
#             "[%s] V(S) call #%d  |S|=%d  q%d..%d  subset=%s",
#             split_name, call_counter["n"], len(subset),
#             question_start, question_end - 1, subset_key,
#         )

#         recorder = TaskRecorder(experiment_name=FIXED_EXP_ID)
#         mgr = RolloutManager(config=eval_config, batch_size=batch_size)

#         # Load exactly enough data and select the right batch
#         mgr.load_epoch_data(epoch=0, shuffle=False, truncate=load_truncate)
#         _, stats = await mgr.main(
#             batch_idx=test_batch_idx, recorder=recorder, use_cache=False
#         )

#         mean_key = next((k for k in stats if k.startswith("Mean@")), None)
#         if mean_key is None:
#             logger.warning("[%s] No Mean@k key in stats: %s", split_name, stats)
#             return 0.0
#         score = stats[mean_key]
#         print(
#             f"    [{split_name}] V(S) #{call_counter['n']}  "
#             f"{mean_key}={score:.4f}  "
#             f"|subset|={len(subset)}  "
#             f"q{question_start}..{question_end-1}"
#         )
#         return score

#     return value_fn


# # ---------------------------------------------------------------------------
# # Cardinality-restricted Shapley estimator
# # ---------------------------------------------------------------------------

# class ExperienceShapley:
#     """
#     Estimate psi_i (cardinality-restricted Shapley at budget k) using
#     m random subsets of size k-1, evaluated on the SHAPLEY split only.

#     psi_i = mean over random S of size k-1 (not containing i) of:
#             V(S u {i}) - V(S)
#     """

#     def __init__(
#         self,
#         experiences: dict[str, str],
#         value_fn: Callable,
#         m: int = 10,
#         k: int = 10,
#         seed: int = 42,
#         log_dir: str = "logs/shapley",
#     ):
#         self.experiences = experiences
#         self.exp_ids = list(experiences.keys())
#         self.n = len(self.exp_ids)
#         self.m = m
#         self.k = min(k, self.n)
#         self.value_fn = value_fn
#         self.seed = seed
#         self.log_dir = log_dir
#         os.makedirs(log_dir, exist_ok=True)

#         self._cache: dict[frozenset, float] = {}
#         self._v_calls = 0
#         self._v_cache_hits = 0
#         self._marginals: list[list[float]] = [[] for _ in range(self.n)]

#     async def run(self) -> dict[str, float]:
#         rng = random.Random(self.seed)

#         print(f"\n{'='*65}")
#         print(f"Cardinality-Restricted Shapley  (n={self.n}, k={self.k}, m={self.m})")
#         print(f"  Each S: |S|=k-1={self.k-1}  |outside S|=n-k={self.n-self.k}")
#         print(f"  Est. V(S) calls: ~{self.m * (1 + self.n - self.k)}")
#         print(f"  Saving to: {self.log_dir}/")
#         print(f"  Kill anytime — files saved after every V(S) call")
#         print(f"{'='*65}\n")

#         for sample_idx in range(self.m):
#             pool = list(range(self.n))
#             S = frozenset(rng.sample(pool, self.k - 1))

#             print(f"--- Sample {sample_idx+1}/{self.m}  "
#                   f"|S|={len(S)}  outside={self.n-len(S)}  "
#                   f"V-calls so far={self._v_calls} ---")

#             await self._process_subset(S, sample_idx + 1)
#             print()

#         phi = self._compute_phi()
#         ranked = dict(sorted(phi.items(), key=lambda x: x[1], reverse=True))
#         self._save(phi, self.m, final=True)
#         return ranked

#     async def _process_subset(self, S: frozenset, sample_num: int) -> None:
#         v_S = await self._eval(S)
#         self._save(self._compute_phi(), sample_num, label="after V(S)")

#         outside = [i for i in range(self.n) if i not in S]
#         for step, i in enumerate(outside):
#             S_plus_i = S | frozenset([i])
#             v_plus = await self._eval(S_plus_i)
#             self._marginals[i].append(v_plus - v_S)

#             phi = self._compute_phi()
#             ranked = sorted(phi.items(), key=lambda x: x[1], reverse=True)
#             print(f"  [{step+1}/{len(outside)}] marginal [{self.exp_ids[i]}]"
#                   f" = {v_plus - v_S:+.4f}  "
#                   f"(psi={phi[self.exp_ids[i]]:+.4f}, n={len(self._marginals[i])})")
#             print("  Current rankings:")
#             for rank, (eid, val) in enumerate(ranked, 1):
#                 n_obs = len(self._marginals[self.exp_ids.index(eid)])
#                 bar = ("+" if val >= 0 else "-") * min(int(abs(val)*200), 20)
#                 print(f"    #{rank:2d} [{eid}]  psi={val:+.4f}  {bar:<20}  n={n_obs}")
#             self._save(phi, sample_num,
#                        label=f"sample {sample_num} step {step+1}")

#     async def _eval(self, fs: frozenset) -> float:
#         if fs in self._cache:
#             self._v_cache_hits += 1
#             score = self._cache[fs]
#             ids = "{" + ",".join(self.exp_ids[i] for i in fs) + "}" if fs else "∅"
#             print(f"    [cache] V({ids}) = {score:.4f}")
#             return score

#         self._v_calls += 1
#         subset_dict = {
#             self.exp_ids[i]: self.experiences[self.exp_ids[i]] for i in fs
#         }
#         ids = "{" + ",".join(self.exp_ids[i] for i in fs) + "}" if fs else "∅"
#         print(f"    [eval #{self._v_calls}] V({ids}) ...", flush=True)
#         score = await self.value_fn(subset_dict)
#         self._cache[fs] = score
#         return score

#     def _compute_phi(self) -> dict[str, float]:
#         return {
#             self.exp_ids[i]: (
#                 sum(self._marginals[i]) / len(self._marginals[i])
#                 if self._marginals[i] else 0.0
#             )
#             for i in range(self.n)
#         }

#     def _save(self, phi: dict, sample_count: int,
#               final: bool = False, label: str = "") -> None:
#         ranked = sorted(phi.items(), key=lambda x: x[1], reverse=True)
#         ts = datetime.now(timezone.utc).isoformat()

#         # 1. JSON
#         payload = {
#             "timestamp": ts,
#             "estimator": "cardinality_restricted_shapley",
#             "k": self.k,
#             "samples_evaluated": sample_count,
#             "m_total": self.m,
#             "v_calls": self._v_calls,
#             "v_cache_hits": self._v_cache_hits,
#             "final": final,
#             "ranked_experiences": [
#                 {
#                     "rank": r + 1,
#                     "experience_id": eid,
#                     "psi_value": round(val, 6),
#                     "n_marginals": len(self._marginals[self.exp_ids.index(eid)]),
#                     "all_marginals": [
#                         round(d, 6)
#                         for d in self._marginals[self.exp_ids.index(eid)]
#                     ],
#                     "content": self.experiences[eid],
#                 }
#                 for r, (eid, val) in enumerate(ranked)
#             ],
#         }
#         json_path = os.path.join(self.log_dir, "shapley_progress.json")
#         with open(json_path, "w", encoding="utf-8") as f:
#             json.dump(payload, f, indent=2, ensure_ascii=False)

#         # 2. CSV
#         csv_path = os.path.join(self.log_dir, "shapley_progress.csv")
#         with open(csv_path, "w", encoding="utf-8") as f:
#             f.write("rank,experience_id,psi_value,n_marginals,content\n")
#             for r, (eid, val) in enumerate(ranked):
#                 n_obs = len(self._marginals[self.exp_ids.index(eid)])
#                 content = self.experiences[eid].replace('"', '""').replace("\n", " ")
#                 f.write(f'{r+1},{eid},{val:.6f},{n_obs},"{content}"\n')

#         # 3. JSONL history
#         jsonl_path = os.path.join(self.log_dir, "shapley_history.jsonl")
#         with open(jsonl_path, "a", encoding="utf-8") as f:
#             f.write(json.dumps({
#                 "timestamp": ts,
#                 "sample": sample_count,
#                 "v_calls": self._v_calls,
#                 "psi": {eid: round(val, 6) for eid, val in ranked},
#             }, ensure_ascii=False) + "\n")

#         status = "FINAL" if final else label
#         print(f"    [saved: {status}] -> {self.log_dir}/")

# """
# Experience Shapley: Cardinality-Restricted Shapley Values.

# Dataset layout (DAPO-Math-17k, shuffle=False):
#     questions 0   .. T-1        : TRAINING  (used by GRPO to generate experiences)
#     questions T   .. T+S-1      : SHAPLEY   (used to compute psi_i values)
#     questions T+S .. T+S+E-1    : EVAL      (used only for final comparison)

# where:
#     T = train_questions  (default 100, = rollout_data_truncate in config)
#     S = shapley_size     (default 100, questions for computing Shapley scores)
#     E = eval_size        (default 100, questions for final 4-way comparison)

# This guarantees:
#   - Shapley scores are not inflated by evaluating on training questions
#   - Final comparison is not inflated by evaluating on Shapley-estimation questions
#   - All four configurations in the comparison (no-exp, all-exp, positive-exp,
#     top-k-exp) are evaluated on the identical held-out eval split
# """

# import asyncio
# import json
# import logging
# import os
# import random
# from datetime import datetime, timezone
# from typing import Callable

# logger = logging.getLogger(__name__)


# # ---------------------------------------------------------------------------
# # Value function
# # ---------------------------------------------------------------------------

# def make_value_fn(
#     base_eval_config,
#     batch_size: int = 50,
#     question_start: int = 0,
#     split_name: str = "eval",
#     temperature_override: float = None,
# ) -> Callable:
#     """
#     V(subset) -> Mean@1 on a fixed question slice of DAPO-Math-17k.

#     Parameters
#     ----------
#     batch_size         : number of questions per evaluation
#     question_start     : first question index (0-based)
#     split_name         : label for exp_id and logs (e.g. 'shapley', 'eval_s42')
#     temperature_override: set to 0.0 for deterministic decoding,
#                          None to use config value
#     """
#     from .rollout_manager import RolloutManager
#     from .utils import TaskRecorder

#     question_end   = question_start + batch_size
#     load_truncate  = question_end
#     test_batch_idx = question_start // batch_size

#     # Encode split params in exp_id to avoid DB conflicts across runs/seeds
#     FIXED_EXP_ID = f"shap_{split_name}_q{question_start}_{question_end}"

#     call_counter = {"n": 0}

#     logger.info(
#         "[make_value_fn:%s] q%d..%d  batch_idx=%d  temp=%s  exp_id=%s",
#         split_name, question_start, question_end - 1,
#         test_batch_idx, temperature_override, FIXED_EXP_ID,
#     )

#     async def value_fn(subset: dict[str, str]) -> float:
#         eval_config = base_eval_config.model_copy(deep=True)

#         eval_config.data.dataset = "DAPO-Math-17k"
#         eval_config.pass_k = 1
#         eval_config.exp_id = FIXED_EXP_ID

#         # Temperature override for deterministic decoding
#         if temperature_override is not None:
#             try:
#                 eval_config.agent.model.model_settings.temperature = temperature_override
#             except Exception as e:
#                 logger.warning("Could not set temperature: %s", e)

#         if subset:
#             experience_text = (
#                 "\n\nWhen solving problems, you MUST first carefully read "
#                 "and understand the helpful instructions and experiences:\n"
#             )
#             experience_text += "\n".join(
#                 [f"[{i}]. {e}" for i, e in subset.items()]
#             )
#             instructions = (
#                 eval_config.agent.agent.instructions
#                 or "You are a helpful assistant."
#             )
#             eval_config.agent.agent.instructions = instructions + experience_text

#         call_counter["n"] += 1
#         subset_key = "_".join(sorted(subset.keys())) if subset else "empty"
#         logger.info(
#             "[%s] V(S) #%d  |S|=%d  q%d..%d  subset=%s",
#             split_name, call_counter["n"], len(subset),
#             question_start, question_end - 1, subset_key,
#         )

#         recorder = TaskRecorder(experiment_name=FIXED_EXP_ID)
#         mgr = RolloutManager(config=eval_config, batch_size=batch_size)
#         mgr.load_epoch_data(epoch=0, shuffle=False, truncate=load_truncate)
#         _, stats = await mgr.main(
#             batch_idx=test_batch_idx, recorder=recorder, use_cache=False
#         )

#         mean_key = next((k for k in stats if k.startswith("Mean@")), None)
#         if mean_key is None:
#             logger.warning("[%s] No Mean@k in stats: %s", split_name, stats)
#             return 0.0
#         score = stats[mean_key]
#         print(
#             f"    [{split_name}] V(S) #{call_counter['n']}  "
#             f"{mean_key}={score:.4f}  "
#             f"|subset|={len(subset)}  "
#             f"q{question_start}..{question_end-1}  "
#             f"temp={temperature_override}"
#         )
#         return score

#     return value_fn


# # ---------------------------------------------------------------------------
# # Cardinality-restricted Shapley estimator
# # ---------------------------------------------------------------------------

# class ExperienceShapley:
#     """
#     Estimate psi_i (cardinality-restricted Shapley at budget k) using
#     m random subsets of size k-1, evaluated on the SHAPLEY split only.

#     psi_i = mean over random S of size k-1 (not containing i) of:
#             V(S u {i}) - V(S)
#     """

#     def __init__(
#         self,
#         experiences: dict[str, str],
#         value_fn: Callable,
#         m: int = 10,
#         k: int = 10,
#         seed: int = 42,
#         log_dir: str = "logs/shapley",
#     ):
#         self.experiences = experiences
#         self.exp_ids = list(experiences.keys())
#         self.n = len(self.exp_ids)
#         self.m = m
#         self.k = min(k, self.n)
#         self.value_fn = value_fn
#         self.seed = seed
#         self.log_dir = log_dir
#         os.makedirs(log_dir, exist_ok=True)

#         self._cache: dict[frozenset, float] = {}
#         self._v_calls = 0
#         self._v_cache_hits = 0
#         self._marginals: list[list[float]] = [[] for _ in range(self.n)]

#     async def run(self) -> dict[str, float]:
#         rng = random.Random(self.seed)

#         print(f"\n{'='*65}")
#         print(f"Cardinality-Restricted Shapley  (n={self.n}, k={self.k}, m={self.m})")
#         print(f"  Each S: |S|=k-1={self.k-1}  |outside S|=n-k={self.n-self.k}")
#         print(f"  Est. V(S) calls: ~{self.m * (1 + self.n - self.k)}")
#         print(f"  Saving to: {self.log_dir}/")
#         print(f"  Kill anytime — files saved after every V(S) call")
#         print(f"{'='*65}\n")

#         for sample_idx in range(self.m):
#             pool = list(range(self.n))
#             S = frozenset(rng.sample(pool, self.k - 1))

#             print(f"--- Sample {sample_idx+1}/{self.m}  "
#                   f"|S|={len(S)}  outside={self.n-len(S)}  "
#                   f"V-calls so far={self._v_calls} ---")

#             await self._process_subset(S, sample_idx + 1)
#             print()

#         phi = self._compute_phi()
#         ranked = dict(sorted(phi.items(), key=lambda x: x[1], reverse=True))
#         self._save(phi, self.m, final=True)
#         return ranked

#     async def _process_subset(self, S: frozenset, sample_num: int) -> None:
#         v_S = await self._eval(S)
#         self._save(self._compute_phi(), sample_num, label="after V(S)")

#         outside = [i for i in range(self.n) if i not in S]
#         for step, i in enumerate(outside):
#             S_plus_i = S | frozenset([i])
#             v_plus = await self._eval(S_plus_i)
#             self._marginals[i].append(v_plus - v_S)

#             phi = self._compute_phi()
#             ranked = sorted(phi.items(), key=lambda x: x[1], reverse=True)
#             print(f"  [{step+1}/{len(outside)}] marginal [{self.exp_ids[i]}]"
#                   f" = {v_plus - v_S:+.4f}  "
#                   f"(psi={phi[self.exp_ids[i]]:+.4f}, n={len(self._marginals[i])})")
#             print("  Current rankings:")
#             for rank, (eid, val) in enumerate(ranked, 1):
#                 n_obs = len(self._marginals[self.exp_ids.index(eid)])
#                 bar = ("+" if val >= 0 else "-") * min(int(abs(val)*200), 20)
#                 print(f"    #{rank:2d} [{eid}]  psi={val:+.4f}  {bar:<20}  n={n_obs}")
#             self._save(phi, sample_num,
#                        label=f"sample {sample_num} step {step+1}")

#     async def _eval(self, fs: frozenset) -> float:
#         if fs in self._cache:
#             self._v_cache_hits += 1
#             score = self._cache[fs]
#             ids = "{" + ",".join(self.exp_ids[i] for i in fs) + "}" if fs else "∅"
#             print(f"    [cache] V({ids}) = {score:.4f}")
#             return score

#         self._v_calls += 1
#         subset_dict = {
#             self.exp_ids[i]: self.experiences[self.exp_ids[i]] for i in fs
#         }
#         ids = "{" + ",".join(self.exp_ids[i] for i in fs) + "}" if fs else "∅"
#         print(f"    [eval #{self._v_calls}] V({ids}) ...", flush=True)
#         score = await self.value_fn(subset_dict)
#         self._cache[fs] = score
#         return score

#     def _compute_phi(self) -> dict[str, float]:
#         return {
#             self.exp_ids[i]: (
#                 sum(self._marginals[i]) / len(self._marginals[i])
#                 if self._marginals[i] else 0.0
#             )
#             for i in range(self.n)
#         }

#     def _save(self, phi: dict, sample_count: int,
#               final: bool = False, label: str = "") -> None:
#         ranked = sorted(phi.items(), key=lambda x: x[1], reverse=True)
#         ts = datetime.now(timezone.utc).isoformat()

#         # 1. JSON
#         payload = {
#             "timestamp": ts,
#             "estimator": "cardinality_restricted_shapley",
#             "k": self.k,
#             "samples_evaluated": sample_count,
#             "m_total": self.m,
#             "v_calls": self._v_calls,
#             "v_cache_hits": self._v_cache_hits,
#             "final": final,
#             "ranked_experiences": [
#                 {
#                     "rank": r + 1,
#                     "experience_id": eid,
#                     "psi_value": round(val, 6),
#                     "n_marginals": len(self._marginals[self.exp_ids.index(eid)]),
#                     "all_marginals": [
#                         round(d, 6)
#                         for d in self._marginals[self.exp_ids.index(eid)]
#                     ],
#                     "content": self.experiences[eid],
#                 }
#                 for r, (eid, val) in enumerate(ranked)
#             ],
#         }
#         json_path = os.path.join(self.log_dir, "shapley_progress.json")
#         with open(json_path, "w", encoding="utf-8") as f:
#             json.dump(payload, f, indent=2, ensure_ascii=False)

#         # 2. CSV
#         csv_path = os.path.join(self.log_dir, "shapley_progress.csv")
#         with open(csv_path, "w", encoding="utf-8") as f:
#             f.write("rank,experience_id,psi_value,n_marginals,content\n")
#             for r, (eid, val) in enumerate(ranked):
#                 n_obs = len(self._marginals[self.exp_ids.index(eid)])
#                 content = self.experiences[eid].replace('"', '""').replace("\n", " ")
#                 f.write(f'{r+1},{eid},{val:.6f},{n_obs},"{content}"\n')

#         # 3. JSONL history
#         jsonl_path = os.path.join(self.log_dir, "shapley_history.jsonl")
#         with open(jsonl_path, "a", encoding="utf-8") as f:
#             f.write(json.dumps({
#                 "timestamp": ts,
#                 "sample": sample_count,
#                 "v_calls": self._v_calls,
#                 "psi": {eid: round(val, 6) for eid, val in ranked},
#             }, ensure_ascii=False) + "\n")

#         status = "FINAL" if final else label
#         print(f"    [saved: {status}] -> {self.log_dir}/")

"""
Experience Shapley: Cardinality-Restricted Shapley Values.

Dataset layout (DAPO-Math-17k, shuffle=False):
    questions 0   .. T-1        : TRAINING  (used by GRPO to generate experiences)
    questions T   .. T+S-1      : SHAPLEY   (used to compute psi_i values)
    questions T+S .. T+S+E-1    : EVAL      (used only for final comparison)

where:
    T = train_questions  (default 100, = rollout_data_truncate in config)
    S = shapley_size     (default 100, questions for computing Shapley scores)
    E = eval_size        (default 100, questions for final 4-way comparison)

This guarantees:
  - Shapley scores are not inflated by evaluating on training questions
  - Final comparison is not inflated by evaluating on Shapley-estimation questions
  - All four configurations in the comparison (no-exp, all-exp, positive-exp,
    top-k-exp) are evaluated on the identical held-out eval split
"""

import asyncio
import json
import logging
import os
import random
from datetime import datetime, timezone
from typing import Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Value function
# ---------------------------------------------------------------------------

def make_value_fn(
    base_eval_config,
    batch_size: int = 50,
    question_start: int = 0,
    split_name: str = "eval",
    temperature_override: float = None,
) -> Callable:
    """
    V(subset) -> Mean@1 on a fixed question slice of the practice dataset.

    Parameters
    ----------
    batch_size         : number of questions per evaluation
    question_start     : first question index (0-based)
    split_name         : label for exp_id and logs (e.g. 'shapley', 'eval_s42')
    temperature_override: set to 0.0 for deterministic decoding,
                         None to use config value
    """
    from .rollout_manager import RolloutManager
    from .utils import TaskRecorder

    question_end   = question_start + batch_size
    load_truncate  = question_end
    test_batch_idx = question_start // batch_size

    # Encode split params in exp_id to avoid DB conflicts across runs/seeds
    FIXED_EXP_ID = f"shap_{split_name}_q{question_start}_{question_end}"

    call_counter = {"n": 0}

    logger.info(
        "[make_value_fn:%s] q%d..%d  batch_idx=%d  temp=%s  exp_id=%s",
        split_name, question_start, question_end - 1,
        test_batch_idx, temperature_override, FIXED_EXP_ID,
    )

    async def value_fn(subset: dict[str, str]) -> float:
        eval_config = base_eval_config.model_copy(deep=True)

        eval_config.data.dataset = base_eval_config.data.dataset
        eval_config.pass_k = 1
        eval_config.exp_id = FIXED_EXP_ID

        # Temperature override for deterministic decoding
        if temperature_override is not None:
            try:
                eval_config.agent.model.model_settings.temperature = temperature_override
            except Exception as e:
                logger.warning("Could not set temperature: %s", e)

        if subset:
            experience_text = (
                "\n\nWhen solving problems, you MUST first carefully read "
                "and understand the helpful instructions and experiences:\n"
            )
            experience_text += "\n".join(
                [f"[{i}]. {e}" for i, e in subset.items()]
            )
            instructions = (
                eval_config.agent.agent.instructions
                or "You are a helpful assistant."
            )
            eval_config.agent.agent.instructions = instructions + experience_text

        call_counter["n"] += 1
        subset_key = "_".join(sorted(subset.keys())) if subset else "empty"
        logger.info(
            "[%s] V(S) #%d  |S|=%d  q%d..%d  subset=%s",
            split_name, call_counter["n"], len(subset),
            question_start, question_end - 1, subset_key,
        )

        recorder = TaskRecorder(experiment_name=FIXED_EXP_ID)
        mgr = RolloutManager(config=eval_config, batch_size=batch_size)
        mgr.load_epoch_data(epoch=0, shuffle=False, truncate=load_truncate)
        _, stats = await mgr.main(
            batch_idx=test_batch_idx, recorder=recorder, use_cache=False
        )

        # Guard against empty stats (all rollouts failed — e.g. missing API key,
        # wrong dataset, or model errors). Return 0.0 so Shapley estimation
        # continues rather than crashing.
        if not stats:
            logger.warning(
                "[%s] Empty stats — all rollouts failed for subset=%s. "
                "Check SERPER_API_KEY, dataset name, and model availability. "
                "Returning 0.0.",
                split_name, subset_key,
            )
            return 0.0

        mean_key = next((k for k in stats if k.startswith("Mean@")), None)
        if mean_key is None:
            logger.warning("[%s] No Mean@k in stats: %s", split_name, stats)
            return 0.0
        score = stats[mean_key]
        print(
            f"    [{split_name}] V(S) #{call_counter['n']}  "
            f"{mean_key}={score:.4f}  "
            f"|subset|={len(subset)}  "
            f"q{question_start}..{question_end-1}  "
            f"temp={temperature_override}"
        )
        return score

    return value_fn


# ---------------------------------------------------------------------------
# Cardinality-restricted Shapley estimator
# ---------------------------------------------------------------------------

class ExperienceShapley:
    """
    Estimate psi_i (cardinality-restricted Shapley at budget k) using
    m random subsets of size k-1, evaluated on the SHAPLEY split only.

    psi_i = mean over random S of size k-1 (not containing i) of:
            V(S u {i}) - V(S)
    """

    def __init__(
        self,
        experiences: dict[str, str],
        value_fn: Callable,
        m: int = 10,
        k: int = 10,
        seed: int = 42,
        log_dir: str = "logs/shapley",
    ):
        self.experiences = experiences
        self.exp_ids = list(experiences.keys())
        self.n = len(self.exp_ids)
        self.m = m
        self.k = min(k, self.n)
        self.value_fn = value_fn
        self.seed = seed
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self._cache: dict[frozenset, float] = {}
        self._v_calls = 0
        self._v_cache_hits = 0
        self._marginals: list[list[float]] = [[] for _ in range(self.n)]

    async def run(self) -> dict[str, float]:
        rng = random.Random(self.seed)

        print(f"\n{'='*65}")
        print(f"Cardinality-Restricted Shapley  (n={self.n}, k={self.k}, m={self.m})")
        print(f"  Each S: |S|=k-1={self.k-1}  |outside S|=n-k={self.n-self.k}")
        print(f"  Est. V(S) calls: ~{self.m * (1 + self.n - self.k)}")
        print(f"  Saving to: {self.log_dir}/")
        print(f"  Kill anytime — files saved after every V(S) call")
        print(f"{'='*65}\n")

        for sample_idx in range(self.m):
            pool = list(range(self.n))
            S = frozenset(rng.sample(pool, self.k - 1))

            print(f"--- Sample {sample_idx+1}/{self.m}  "
                  f"|S|={len(S)}  outside={self.n-len(S)}  "
                  f"V-calls so far={self._v_calls} ---")

            await self._process_subset(S, sample_idx + 1)
            print()

        phi = self._compute_phi()
        ranked = dict(sorted(phi.items(), key=lambda x: x[1], reverse=True))
        self._save(phi, self.m, final=True)
        return ranked

    async def _process_subset(self, S: frozenset, sample_num: int) -> None:
        v_S = await self._eval(S)
        self._save(self._compute_phi(), sample_num, label="after V(S)")

        outside = [i for i in range(self.n) if i not in S]
        for step, i in enumerate(outside):
            S_plus_i = S | frozenset([i])
            v_plus = await self._eval(S_plus_i)
            self._marginals[i].append(v_plus - v_S)

            phi = self._compute_phi()
            ranked = sorted(phi.items(), key=lambda x: x[1], reverse=True)
            print(f"  [{step+1}/{len(outside)}] marginal [{self.exp_ids[i]}]"
                  f" = {v_plus - v_S:+.4f}  "
                  f"(psi={phi[self.exp_ids[i]]:+.4f}, n={len(self._marginals[i])})")
            print("  Current rankings:")
            for rank, (eid, val) in enumerate(ranked, 1):
                n_obs = len(self._marginals[self.exp_ids.index(eid)])
                bar = ("+" if val >= 0 else "-") * min(int(abs(val)*200), 20)
                print(f"    #{rank:2d} [{eid}]  psi={val:+.4f}  {bar:<20}  n={n_obs}")
            self._save(phi, sample_num,
                       label=f"sample {sample_num} step {step+1}")

    async def _eval(self, fs: frozenset) -> float:
        if fs in self._cache:
            self._v_cache_hits += 1
            score = self._cache[fs]
            ids = "{" + ",".join(self.exp_ids[i] for i in fs) + "}" if fs else "∅"
            print(f"    [cache] V({ids}) = {score:.4f}")
            return score

        self._v_calls += 1
        subset_dict = {
            self.exp_ids[i]: self.experiences[self.exp_ids[i]] for i in fs
        }
        ids = "{" + ",".join(self.exp_ids[i] for i in fs) + "}" if fs else "∅"
        print(f"    [eval #{self._v_calls}] V({ids}) ...", flush=True)
        score = await self.value_fn(subset_dict)
        self._cache[fs] = score
        return score

    def _compute_phi(self) -> dict[str, float]:
        return {
            self.exp_ids[i]: (
                sum(self._marginals[i]) / len(self._marginals[i])
                if self._marginals[i] else 0.0
            )
            for i in range(self.n)
        }

    def _save(self, phi: dict, sample_count: int,
              final: bool = False, label: str = "") -> None:
        ranked = sorted(phi.items(), key=lambda x: x[1], reverse=True)
        ts = datetime.now(timezone.utc).isoformat()

        # 1. JSON
        payload = {
            "timestamp": ts,
            "estimator": "cardinality_restricted_shapley",
            "k": self.k,
            "samples_evaluated": sample_count,
            "m_total": self.m,
            "v_calls": self._v_calls,
            "v_cache_hits": self._v_cache_hits,
            "final": final,
            "ranked_experiences": [
                {
                    "rank": r + 1,
                    "experience_id": eid,
                    "psi_value": round(val, 6),
                    "n_marginals": len(self._marginals[self.exp_ids.index(eid)]),
                    "all_marginals": [
                        round(d, 6)
                        for d in self._marginals[self.exp_ids.index(eid)]
                    ],
                    "content": self.experiences[eid],
                }
                for r, (eid, val) in enumerate(ranked)
            ],
        }
        json_path = os.path.join(self.log_dir, "shapley_progress.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        # 2. CSV
        csv_path = os.path.join(self.log_dir, "shapley_progress.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("rank,experience_id,psi_value,n_marginals,content\n")
            for r, (eid, val) in enumerate(ranked):
                n_obs = len(self._marginals[self.exp_ids.index(eid)])
                content = self.experiences[eid].replace('"', '""').replace("\n", " ")
                f.write(f'{r+1},{eid},{val:.6f},{n_obs},"{content}"\n')

        # 3. JSONL history
        jsonl_path = os.path.join(self.log_dir, "shapley_history.jsonl")
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": ts,
                "sample": sample_count,
                "v_calls": self._v_calls,
                "psi": {eid: round(val, 6) for eid, val in ranked},
            }, ensure_ascii=False) + "\n")

        status = "FINAL" if final else label
        print(f"    [saved: {status}] -> {self.log_dir}/")