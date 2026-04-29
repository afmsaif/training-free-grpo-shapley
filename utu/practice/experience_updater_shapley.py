"""
Experience updater for training-free GRPO.
"""

import asyncio
import copy
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone

from agents import custom_span
from tqdm import tqdm

from ..config import AgentConfig
from ..db import EvaluationSample
from ..utils import FileUtils, SimplifiedAsyncOpenAI, get_logger
from .utils import TaskRecorder

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _safe_serialize(obj):
    """Best-effort JSON serialization for arbitrary objects."""
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # Pydantic models
    if hasattr(obj, "model_dump"):
        return _safe_serialize(obj.model_dump())
    return str(obj)


def compute_memory_diff(old: dict, new: dict) -> dict:
    """Compute added / deleted / updated experiences between two memory snapshots."""
    old_keys = set(old.keys())
    new_keys = set(new.keys())

    added = {k: new[k] for k in new_keys - old_keys}
    deleted = {k: old[k] for k in old_keys - new_keys}
    updated = {k: new[k] for k in old_keys & new_keys if old[k] != new[k]}

    return {
        "added_experiences": added,
        "deleted_experiences": deleted,
        "updated_experiences": updated,
        "total_memory_size_before": len(old),
        "total_memory_size_after": len(new),
    }


def _write_step_log(log_data: dict, epoch, step) -> None:
    """Write per-step JSON log to logs/epoch_{epoch}/step_{step}.json."""
    if epoch is None or step is None:
        return
    log_dir = os.path.join("logs", f"epoch_{epoch}")
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"step_{step}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_safe_serialize(log_data), f, indent=2, ensure_ascii=False)


def _append_summary_log(line: dict) -> None:
    """Append one summary line to logs/summary.jsonl."""
    os.makedirs("logs", exist_ok=True)
    with open(os.path.join("logs", "summary.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(_safe_serialize(line), ensure_ascii=False) + "\n")


# Maximum characters of a single trajectory sent to the summarizer LLM.
# 32 768-token models allow ~24 000 chars of trajectory before the system/user
# overhead pushes the request over the limit (1 token ≈ 4 chars).
_MAX_TRAJECTORY_CHARS: int = 20_000

# Seconds to wait for a single LLM API call before declaring it hung.
_LLM_CALL_TIMEOUT: int = 120


def _repair_json(raw: str) -> str:
    """Fix lone backslashes that the LLM emits in math/LaTeX strings.

    ``json.loads`` rejects sequences like ``\\frac`` or ``\\sqrt`` because
    ``\\f`` and ``\\s`` are not valid JSON escape sequences.  We replace every
    backslash that is *not* followed by a recognised JSON escape character with
    a double-backslash so the string round-trips correctly.

    Valid single-char JSON escapes: " \\ / b f n r t u
    """
    return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)


def _save_prompt_once(stage: str, system_prompt: str, user_prompt: str) -> None:
    """Save the rendered prompts for a stage to logs/prompts/{stage}.json.

    Only writes on the first call (initial phase) — subsequent calls do not
    overwrite, so the file always reflects what the LLM saw at step zero.
    """
    prompt_dir = os.path.join("logs", "prompts")
    os.makedirs(prompt_dir, exist_ok=True)
    path = os.path.join(prompt_dir, f"{stage}.json")
    if os.path.exists(path):          # already captured — keep initial version
        return
    payload = {
        "stage": stage,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "note": "First rendered prompt for this stage (initial phase).",
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ExperienceUpdater:
    def __init__(self, config: AgentConfig, agent_objective: str, learning_objective: str):
        self.config = config
        self.agent_objective = agent_objective
        self.learning_objective = learning_objective
        self.prompts = FileUtils.load_prompts("practice/experience.yaml")
        self.llm = SimplifiedAsyncOpenAI(**config.model.model_provider.model_dump())

    async def run(
        self,
        rollouts: list[EvaluationSample],
        recorder: TaskRecorder,
        concurrency: int = 16,
        given_ground_truth: bool = True,
        num_experiences: int = 2,
        epoch: int = None,
        step: int = None,
        shapley_scores: dict[str, float] = None,
    ) -> None:
        """Update experiences based on rollouts.

        Parameters
        ----------
        shapley_scores : dict[str, float], optional
            Cardinality-restricted Shapley values (psi_i) for each experience.
            When provided, three feedback mechanisms are activated:
              - Option 2: _format_exp_and_ops annotates each experience with its
                psi value and a HELPFUL/HARMFUL/NEUTRAL label, guiding the batch
                update LLM to prefer deleting or rewriting harmful experiences.
              - Option 2: _group_update appends psi labels to existing experiences
                shown in the group-level update prompt.
            Use run_shapley.py to compute these scores before each GRPO epoch.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        memory_before = copy.deepcopy(recorder.experiences or {})

        # 1. Summarize trajectory for each rollout
        with custom_span("Trajectory Summarization"):
            problem_to_summarized_rollouts = await self._single_rollout_summary(
                rollouts=rollouts, concurrency=concurrency, given_ground_truth=given_ground_truth
            )

        # 2. Generate semantic group advantages based on summarized rollouts
        with custom_span("Semantic Group Advantage"):
            new_experiences = await self._group_advantage(
                problem_to_summarized_rollouts=problem_to_summarized_rollouts,
                concurrency=concurrency,
                given_ground_truth=given_ground_truth,
                num_experiences=num_experiences,
            )

        # 3. group update experiences
        with custom_span("Group update"):
            critiques = await self._group_update(
                recorder=recorder,
                new_experiences=new_experiences,
                concurrency=concurrency,
                shapley_scores=shapley_scores,
            )

        # 4. batch update experiences
        with custom_span("Batch update"):
            new_experiences = await self._batch_update(
                recorder=recorder,
                critiques=critiques,
                shapley_scores=shapley_scores,
            )

        # 5. assign new experience IDs
        new_experiences = {f"G{i}": exp for i, exp in enumerate(new_experiences.values())}
        recorder.experiences_update(new_experiences)

        # ------------------------------------------------------------------
        # Research-grade logging
        # ------------------------------------------------------------------
        memory_after = copy.deepcopy(recorder.experiences or {})
        memory_diff = compute_memory_diff(memory_before, memory_after)

        # Collect all operations across critiques
        all_operations = []
        for c in (critiques or []):
            ops = c.get("operations") if isinstance(c, dict) else []
            if ops:
                all_operations.extend(ops)

        # Per-group reward stats
        group_stats = []
        for group in (new_experiences if isinstance(new_experiences, list) else []):
            if not isinstance(group, dict):
                continue
            rollout_list = group.get("rollouts") or []
            rewards = [r.get("reward") for r in rollout_list if isinstance(r, dict) and "reward" in r]
            avg_r = sum(rewards) / len(rewards) if rewards else None
            variance = (
                sum((r - avg_r) ** 2 for r in rewards) / len(rewards) if rewards and avg_r is not None else None
            )
            group_stats.append({"avg_reward": avg_r, "reward_variance": variance})

        # Flat reward list from raw rollouts for summary
        all_rewards = [r.reward for r in rollouts if hasattr(r, "reward") and r.reward is not None]
        avg_reward_global = sum(all_rewards) / len(all_rewards) if all_rewards else None

        # Build rollout group log
        group_logs = []
        for problem, summarized in (problem_to_summarized_rollouts or {}).items():
            rewards = [r.get("reward") for r in summarized if isinstance(r, dict) and "reward" in r]
            avg_r = sum(rewards) / len(rewards) if rewards else None
            group_logs.append({
                "group_id": problem,
                "group_size": len(summarized),
                "rewards": rewards,
                "avg_reward": avg_r,
            })

        # Build raw rollout log
        raw_rollout_logs = []
        for r in (rollouts or []):
            try:
                traj = json.loads(r.trajectories)[0]["trajectory"] if r.trajectories else None
            except Exception:
                traj = None
            raw_rollout_logs.append({
                "question": getattr(r, "raw_question", None),
                "reward": getattr(r, "reward", None),
                "trajectory": traj,
                "reasoning": getattr(r, "reasoning", None),
            })

        # Build per-rollout summary log
        rollout_summary_logs = []
        for problem, summarized in (problem_to_summarized_rollouts or {}).items():
            for item in summarized:
                rollout_summary_logs.append({
                    "question": item.get("raw_question"),
                    "reward": item.get("reward"),
                    "trajectory_summary": item.get("trajectory_summary"),
                })

        # Build group advantage log
        group_advantage_logs = []
        for ga in (critiques or []):
            if not isinstance(ga, dict):
                continue
            rollout_list = ga.get("rollouts") or []
            rewards = [r.get("reward") for r in rollout_list if isinstance(r, dict)]
            avg_r = sum(rewards) / len(rewards) if rewards else None
            variance = (
                sum((r - avg_r) ** 2 for r in rewards) / len(rewards) if rewards and avg_r is not None else None
            )
            group_advantage_logs.append({
                "critique": ga.get("critique"),
                "extracted_experiences": ga.get("experiences"),
                "num_rollouts": len(rollout_list),
                "group_avg_reward": avg_r,
                "group_reward_variance": variance,
            })

        # Build group update log
        group_update_logs = []
        for c in (critiques or []):
            if not isinstance(c, dict):
                continue
            ops = c.get("operations") or []
            group_update_logs.append({
                "extracted_experiences": c.get("experiences"),
                "parsed_operations": ops,
                "operation_types": [op.get("operation") for op in ops if isinstance(op, dict)],
            })

        step_log = {
            # --- 1. Metadata ---
            "metadata": {
                "epoch": epoch,
                "step": step,
                "timestamp": timestamp,
                "num_rollouts": len(rollouts),
            },
            # --- 2. Raw rollouts ---
            "raw_rollouts": raw_rollout_logs,
            # --- 3. Rollout grouping ---
            "rollout_grouping": {
                "num_groups": len(group_logs),
                "groups": group_logs,
            },
            # --- 4. Rollout summaries ---
            "rollout_summaries": rollout_summary_logs,
            # --- 5. Group advantage ---
            "group_advantage": group_advantage_logs,
            # --- 6. Group update ---
            "group_update": group_update_logs,
            # --- 7. Batch update / memory transition ---
            "batch_update": {
                "memory_before": memory_before,
                "memory_after": memory_after,
            },
            # --- 8. Memory diff ---
            "memory_diff": memory_diff,
            # --- 9. Learning signals ---
            "learning_signals": {
                "number_of_new_experiences": len(memory_diff.get("added_experiences", {})),
                "number_of_operations": len(all_operations),
                "avg_reward_per_group": [g["avg_reward"] for g in group_logs],
                "reward_variance_per_group": [
                    ga.get("group_reward_variance") for ga in group_advantage_logs
                ],
            },
        }

        _write_step_log(step_log, epoch, step)

        # Global summary line
        _append_summary_log({
            "epoch": epoch,
            "step": step,
            "avg_reward": avg_reward_global,
            "num_experiences": len(memory_after),
            "num_operations": len(all_operations),
            "memory_size": len(memory_after),
        })

        return new_experiences

    async def _single_rollout_summary(
        self,
        rollouts: list[EvaluationSample],
        concurrency: int,
        given_ground_truth: bool,
    ) -> dict[str, list[str]]:
        """Summarize each rollout's trajectory."""
        # group by problems
        problems_to_rollouts = defaultdict(list)
        for rollout in rollouts:
            if len(rollout.trajectories) > 0:
                problems_to_rollouts[rollout.raw_question].append(rollout)

        # only summarize the group whose rollouts are partially correct
        all_rollouts_to_process = []
        for rollouts in problems_to_rollouts.values():
            if given_ground_truth:
                # only for those partially correct
                scores = [each.reward for each in rollouts]
                avg_score = sum(scores) / len(scores)
                if avg_score > 0 and avg_score < 1:
                    all_rollouts_to_process.extend(rollouts)
            else:
                all_rollouts_to_process.extend(rollouts)

        semaphore = asyncio.Semaphore(concurrency)

        async def summarize_with_semaphore(item: EvaluationSample):
            async with semaphore:
                try:
                    with custom_span("summary single rollout"):
                        sp = FileUtils.get_jinja_template_str(
                            self.prompts["SINGLE_ROLLOUT_SUMMARY_TEMPLATE_SP"]
                        ).render(
                            agent_objective=self.agent_objective,
                            learning_objective=self.learning_objective,
                        )
                        up = FileUtils.get_jinja_template_str(
                            self.prompts["SINGLE_ROLLOUT_SUMMARY_TEMPLATE_UP"]
                        ).render(
                            question=item.raw_question,
                            trajectory=json.loads(item.trajectories)[0]["trajectory"][:_MAX_TRAJECTORY_CHARS],
                            answer=item.correct_answer if given_ground_truth else "[REDACTED]",
                            critique=item.reasoning or "[No critique provided]",
                        )
                        _save_prompt_once("single_rollout_summary", sp, up)
                        response = await asyncio.wait_for(
                            self.llm.query_one(
                                messages=[
                                    {"role": "system", "content": sp},
                                    {"role": "user", "content": up},
                                ],
                                **self.config.model.model_params.model_dump(),
                            ),
                            timeout=_LLM_CALL_TIMEOUT,
                        )
                    return {"trajectory_summary": response, **item.model_dump()}
                except Exception as e:
                    logger.warning(f"Warning: failed in single rollout summary, {e}")
                    return None

        # parallel running
        tasks = [summarize_with_semaphore(item) for item in all_rollouts_to_process]
        results = defaultdict(list)
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Single rollout summary"):
            result = await task
            if result is not None:
                problem = result["raw_question"]
                results[problem].append(result)
        return results

    async def _group_advantage(
        self,
        problem_to_summarized_rollouts: dict[str, list[dict]],
        concurrency: int,
        given_ground_truth: bool,
        num_experiences: int,
    ) -> dict[str, dict]:
        """Generate critique for each query based on summarized rollouts."""
        all_rollouts = []
        for rollouts in problem_to_summarized_rollouts.values():
            if given_ground_truth:
                # only for those partially correct
                scores = [each["reward"] for each in rollouts]
                avg_score = sum(scores) / len(scores)
                if avg_score > 0 and avg_score < 1:
                    all_rollouts.append(rollouts)
            else:
                all_rollouts.append(rollouts)

        semaphore = asyncio.Semaphore(concurrency)

        async def critique_with_semaphore(rollouts_per_problem: list[dict]):
            async with semaphore:
                try:
                    with custom_span("single query group advantage"):
                        formatted_trajectories = "\n\n".join(
                            [
                                f"Attempt {i + 1} (Reward {each['reward'] if given_ground_truth else '[REDACTED]'}):\n"
                                f"{each['trajectory_summary']}"
                                for i, each in enumerate(rollouts_per_problem)
                            ]
                        )
                        sp = FileUtils.get_jinja_template_str(self.prompts["SINGLE_QUERY_GROUP_ADVANTAGE_SP"]).render(
                            agent_objective=self.agent_objective,
                            learning_objective=self.learning_objective,
                            num_experiences=num_experiences,
                        )
                        up = FileUtils.get_jinja_template_str(self.prompts["SINGLE_QUERY_GROUP_ADVANTAGE_UP"]).render(
                            question=rollouts_per_problem[0]["raw_question"],
                            answer=rollouts_per_problem[0]["correct_answer"] if given_ground_truth else "[REDACTED]",
                            trajectories=formatted_trajectories,
                        )
                        _save_prompt_once("group_advantage", sp, up)
                        response = await asyncio.wait_for(
                            self.llm.query_one(
                                messages=[
                                    {"role": "system", "content": sp},
                                    {"role": "user", "content": up},
                                ],
                                **self.config.model.model_params.model_dump(),
                            ),
                            timeout=_LLM_CALL_TIMEOUT,
                        )

                        # extract experiences from the response
                        pattern = re.compile(r"<Experiences>\s*(.*?)\s*</Experiences>", re.DOTALL | re.IGNORECASE)
                        match = pattern.search(response)
                        experiences = match.group(1).strip() if match else ""
                    return {"rollouts": rollouts_per_problem, "critique": response, "experiences": experiences}
                except Exception as e:
                    logger.warning(f"Warning: failed in single group advantage, {e}")
                    return None

        # parallel running
        results = []
        tasks = [critique_with_semaphore(rollouts_per_problem) for rollouts_per_problem in all_rollouts]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Single query group advantage"):
            result = await task
            if result is not None:
                results.append(result)

        return results

    async def _group_update(
        self,
        recorder: TaskRecorder,
        new_experiences: list[dict],
        concurrency: int,
        shapley_scores: dict[str, float] = None,
    ) -> dict[str, str]:
        """Group update experiences based on critiques."""
        semaphore = asyncio.Semaphore(concurrency)

        async def group_update_with_semaphore(new_experience: dict):
            async with semaphore:
                try:
                    with custom_span("single group update"):
                        # get current experiences from recorder
                        curr_experiences = recorder.experiences or {}
                        if shapley_scores and curr_experiences:
                            def _annotate(exp_id, text):
                                psi = shapley_scores.get(exp_id)
                                if psi is None:
                                    return f"[{exp_id}]. {text}"
                                label = "HELPFUL" if psi > 0.01 else ("HARMFUL" if psi < -0.01 else "NEUTRAL")
                                return f"[{exp_id}]. {text} [psi={psi:+.4f} {label}]"
                            formatted_experiences = "\n".join(
                                [_annotate(i, e) for i, e in curr_experiences.items()]
                            )
                        else:
                            formatted_experiences = (
                                "\n".join([f"[{i}]. {e}" for i, e in curr_experiences.items()])
                                if curr_experiences
                                else "None"
                            )
                        sp = FileUtils.get_jinja_template_str(
                            self.prompts["GROUP_EXPERIENCE_UPDATE_TEMPLATE_SP"]
                        ).render(
                            agent_objective=self.agent_objective,
                            learning_objective=self.learning_objective,
                        )
                        up = FileUtils.get_jinja_template_str(
                            self.prompts["GROUP_EXPERIENCE_UPDATE_TEMPLATE_UP"]
                        ).render(
                            existing_experiences=formatted_experiences,
                            new_experiences=new_experience["experiences"],
                        )
                        _save_prompt_once("group_update", sp, up)
                        response = await asyncio.wait_for(
                            self.llm.query_one(
                                messages=[
                                    {"role": "system", "content": sp},
                                    {"role": "user", "content": up},
                                ],
                                **self.config.model.model_params.model_dump(),
                            ),
                            timeout=_LLM_CALL_TIMEOUT,
                        )
                        # parse response — repair lone backslashes from math/LaTeX
                        # before handing to json.loads (fixes "Invalid \escape" errors)
                        raw_json = response.split("```json")[-1].split("```")[0]
                        operations = json.loads(_repair_json(raw_json))
                    return {"operations": operations, **new_experience}
                except Exception as e:
                    logger.warning(f"Warning: failed in group update experience, {e}")
                    return None

        # parallel running
        results = []
        tasks = [group_update_with_semaphore(new_experience) for new_experience in new_experiences]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Group update"):
            result = await task
            if result is not None:
                results.append(result)
        return results

    async def _batch_update(
        self, recorder: TaskRecorder, critiques: list[dict], max_retries: int = 3,
        shapley_scores: dict[str, float] = None,
    ) -> dict[str, dict]:
        """Batch update experiences based on critiques."""
        # get current experiences from recorder
        logger.info("Batch update")
        # collect operations
        all_operations = []
        for each in critiques:
            all_operations.extend(each["operations"])
        print("- Num of operations to process:", len(all_operations))

        # use LLM to get the revision plan
        experiences = recorder.experiences or {}
        revision_plan = []
        for _ in range(max_retries):
            try:
                sp = FileUtils.get_jinja_template_str(self.prompts["BATCH_EXPERIENCE_UPDATE_TEMPLATE_SP"]).render(
                    agent_objective=self.agent_objective,
                    learning_objective=self.learning_objective,
                )
                up = FileUtils.get_jinja_template_str(self.prompts["BATCH_EXPERIENCE_UPDATE_TEMPLATE_UP"]).render(
                    experiences_and_operations=self._format_exp_and_ops(experiences, all_operations, shapley_scores=shapley_scores)
                )
                _save_prompt_once("batch_update", sp, up)
                response = await asyncio.wait_for(
                    self.llm.query_one(
                        messages=[
                            {"role": "system", "content": sp},
                            {"role": "user", "content": up},
                        ],
                        **self.config.model.model_params.model_dump(),
                    ),
                    timeout=_LLM_CALL_TIMEOUT,
                )
                # parse response — repair lone backslashes before json.loads
                raw_json = response.split("```json")[-1].split("```")[0]
                revision_plan = json.loads(_repair_json(raw_json))
                break
            except Exception:
                print("Warning: failed to decode in updating general experiences")

        # apply revision plan to get new experiences
        max_ID = len(experiences)
        new_experiences = copy.deepcopy(experiences)
        for plan in revision_plan:
            operation = plan.get("operation", "ADD")
            content = plan.get("content", "")
            target_id = plan.get("id", None)
            if not content:
                continue

            if operation == "ADD":
                new_experiences[f"{max_ID}"] = content
                max_ID += 1
            elif operation == "UPDATE":
                if target_id in new_experiences:
                    new_experiences[target_id] = content
                else:
                    # directly add new experience
                    new_experiences[f"{max_ID}"] = content
                    max_ID += 1
            elif operation == "DELETE":
                if target_id in new_experiences:
                    del new_experiences[target_id]
        print("- Num of candidate experiences:", len(new_experiences))
        return new_experiences

    def _format_exp_and_ops(
        self,
        experiences: dict[str, str],
        operations: list[dict],
        shapley_scores: dict[str, float] = None,
    ) -> str:
        """Format experiences and operations, with optional Shapley feedback.

        When shapley_scores is provided, each experience is annotated with
        its psi value and a plain-language label (HELPFUL / HARMFUL / NEUTRAL).
        This guides the LLM to preferentially replace or delete harmful
        experiences and preserve helpful ones in the next revision plan.
        """
        if not operations:
            return "No batch operations."

        def _shapley_annotation(exp_id: str) -> str:
            """Return a Shapley feedback line for this experience, if available."""
            if not shapley_scores or exp_id not in shapley_scores:
                return ""
            psi = shapley_scores[exp_id]
            if psi > 0.01:
                return (
                    f"[SHAPLEY FEEDBACK: psi={psi:+.4f} — HELPFUL. "
                    f"This experience improves agent performance. "
                    f"Preserve its core pattern when updating.]\n"
                )
            elif psi < -0.01:
                return (
                    f"[SHAPLEY FEEDBACK: psi={psi:+.4f} — HARMFUL. "
                    f"This experience HURTS agent performance. "
                    f"Strongly prefer DELETE or rewrite to fix the harmful pattern.]\n"
                )
            else:
                return (
                    f"[SHAPLEY FEEDBACK: psi={psi:+.4f} — NEUTRAL. "
                    f"Minimal measured impact. Update only if clearly beneficial.]\n"
                )

        # Format existing experiences and their related operations
        formatted_res = []
        for id, exp in experiences.items():
            curr_str = f"Experience {id}:\nContent: {exp}\n"
            curr_str += _shapley_annotation(id)
            related_ops = [op for op in operations if op.get("id") == id]
            if related_ops:
                curr_str += "Related Operations:\n"
                op_str = []
                for op in related_ops:
                    op_str.append(f"{json.dumps(op, ensure_ascii=False, indent=2)}")
                op_str = "\n".join(op_str)
                curr_str += op_str
            else:
                curr_str += "No related operations."
            formatted_res.append(curr_str)

        # Format operations without specific IDs
        no_id_ops = [op for op in operations if not op.get("id", None)]
        if no_id_ops:
            curr_str = "Operations without specific Experience ID:\n"
            op_str = []
            for op in no_id_ops:
                op_str.append(f"{json.dumps(op, ensure_ascii=False, indent=2)}")
            op_str = "\n".join(op_str)
            curr_str += op_str
            formatted_res.append(curr_str)

        return "\n\n".join(formatted_res)
