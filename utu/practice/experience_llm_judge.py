"""
LLM-as-Judge Experience Ranker.

Uses an LLM to evaluate and rank experiences based on their likely
utility for the agent's objective. This serves as a comparison baseline
against cardinality-restricted Shapley values.

The LLM judge is given:
  - The agent objective
  - The learning objective
  - All candidate experiences
  - (optionally) a sample of rollout trajectories showing what the agent
    struggles with

And asked to:
  1. Score each experience (0-10) with reasoning
  2. Rank them from most to least beneficial
  3. Identify harmful/redundant experiences to remove

Comparison with Shapley:
  - Shapley: model-free, measures actual impact on Pass@1, expensive
  - LLM judge: model-based, measures perceived utility, cheap and fast
  - Both can be used for the iterative refinement loop in
    iterative_shapley_grpo.py by swapping the scorer

Files:
  experience_llm_judge.py     ← this file (put in utu/practice/)
  run_llm_judge.py            ← CLI script (put in scripts/)
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Callable

from ..utils import FileUtils, SimplifiedAsyncOpenAI, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """\
You are an expert AI researcher evaluating the quality of learned experiences \
for a math-solving agent. Your task is to assess how beneficial each experience \
is for improving the agent's performance on mathematical reasoning tasks.

An experience is a short instruction or heuristic that the agent reads before \
solving problems. Good experiences help the agent avoid common mistakes, use \
tools effectively, and reason more carefully. Bad experiences mislead the agent, \
contradict good practices, or are too vague to be useful.

You must be critical and honest. An experience that sounds plausible but does \
not actually help performance should receive a low score.\
"""

_JUDGE_USER_PROMPT = """\
Agent objective:
{agent_objective}

Learning objective:
{learning_objective}

{rollout_context}

Below are {n} candidate experiences. For each experience:
1. Give a score from 0 to 10 (0 = actively harmful, 5 = neutral/unclear, \
10 = highly beneficial)
2. Give a one-sentence justification
3. Classify as: HELPFUL / NEUTRAL / HARMFUL

After scoring all experiences individually, provide:
- A final ranked list from most to least beneficial
- A list of experiences you recommend REMOVING (score < 4 or classified HARMFUL)

Respond in this exact JSON format:
```json
{{
  "individual_scores": [
    {{
      "experience_id": "G0",
      "score": 8,
      "classification": "HELPFUL",
      "justification": "Encourages systematic verification which reduces errors."
    }},
    ...
  ],
  "ranked_ids": ["G3", "G0", "G7", ...],
  "remove_ids": ["G22", "G23"],
  "judge_reasoning": "Overall assessment of the experience set..."
}}
```

Experiences to evaluate:
{experiences_text}
"""

_ROLLOUT_CONTEXT_PROMPT = """\
Here are examples of recent agent trajectories showing what the agent \
currently struggles with (use this to assess which experiences would help most):

{rollout_examples}

"""


# ---------------------------------------------------------------------------
# Main judge class
# ---------------------------------------------------------------------------

class LLMExperienceJudge:
    """
    Uses an LLM to rank and score experiences.

    Parameters
    ----------
    llm_config : AgentConfig
        Config for the LLM to use as judge. Can be the same LLM used
        for experience generation or a stronger model (e.g. GPT-4).
    agent_objective : str
        Description of what the agent is trying to do.
    learning_objective : str
        Description of what good experiences should teach.
    log_dir : str
        Directory for saving judge outputs.
    """

    def __init__(
        self,
        llm_config,
        agent_objective: str,
        learning_objective: str,
        log_dir: str = "logs/llm_judge",
    ):
        self.agent_objective = agent_objective
        self.learning_objective = learning_objective
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.llm = SimplifiedAsyncOpenAI(
            **llm_config.model.model_provider.model_dump()
        )
        self.llm_params = llm_config.model.model_params.model_dump()

    async def judge(
        self,
        experiences: dict[str, str],
        rollouts: list = None,
        max_retries: int = 3,
        label: str = "",
    ) -> dict:
        """
        Judge all experiences and return ranked scores.

        Parameters
        ----------
        experiences : dict[str, str]
            {experience_id: content} to evaluate.
        rollouts : list[EvaluationSample], optional
            Recent rollout samples. Used to give the judge context about
            what the agent currently struggles with.
        max_retries : int
            Retry attempts for JSON parsing failures.
        label : str
            Label for logging (e.g. "step_3_round_1").

        Returns
        -------
        dict with keys:
            scores       : {exp_id: float}  0-10 scores
            psi_proxy    : {exp_id: float}  normalised to [-1, +1] for
                           drop-in compatibility with Shapley scores
            ranked_ids   : [exp_id, ...]    most to least beneficial
            remove_ids   : [exp_id, ...]    recommended for removal
            raw_response : str              full LLM response
            reasoning    : str              judge's overall assessment
        """
        if not experiences:
            return {
                "scores": {}, "psi_proxy": {}, "ranked_ids": [],
                "remove_ids": [], "raw_response": "", "reasoning": "",
            }

        experiences_text = self._format_experiences(experiences)
        rollout_context = self._format_rollout_context(rollouts)

        user_prompt = _JUDGE_USER_PROMPT.format(
            agent_objective=self.agent_objective,
            learning_objective=self.learning_objective,
            rollout_context=rollout_context,
            n=len(experiences),
            experiences_text=experiences_text,
        )

        result = None
        raw_response = ""
        for attempt in range(max_retries):
            try:
                raw_response = await self.llm.query_one(
                    messages=[
                        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    **self.llm_params,
                )
                # Parse JSON from response
                json_match = re.search(
                    r"```json\s*(.*?)\s*```", raw_response, re.DOTALL
                )
                json_str = json_match.group(1) if json_match else raw_response
                # Fix common LLM JSON issues
                json_str = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)
                result = json.loads(json_str)
                break
            except Exception as e:
                logger.warning(
                    "LLM judge attempt %d/%d failed: %s",
                    attempt + 1, max_retries, e,
                )

        if result is None:
            logger.error("LLM judge failed after %d attempts.", max_retries)
            # Return neutral scores as fallback
            return {
                "scores": {eid: 5.0 for eid in experiences},
                "psi_proxy": {eid: 0.0 for eid in experiences},
                "ranked_ids": list(experiences.keys()),
                "remove_ids": [],
                "raw_response": raw_response,
                "reasoning": "Judge failed — neutral scores assigned.",
            }

        # Extract individual scores
        scores: dict[str, float] = {}
        for item in result.get("individual_scores", []):
            eid = item.get("experience_id", "")
            if eid in experiences:
                scores[eid] = float(item.get("score", 5))

        # Fill in any missing experiences with neutral score
        for eid in experiences:
            if eid not in scores:
                scores[eid] = 5.0

        # Normalise scores to [-1, +1] as psi_proxy for Shapley compatibility
        # Score 5 = neutral (0), 10 = +1, 0 = -1
        psi_proxy = {
            eid: (score - 5.0) / 5.0
            for eid, score in scores.items()
        }

        ranked_ids = result.get("ranked_ids", [])
        # Fill in any missing from ranked list
        scored_not_ranked = [e for e in experiences if e not in ranked_ids]
        ranked_ids = ranked_ids + scored_not_ranked

        remove_ids = result.get("remove_ids", [])
        reasoning = result.get("judge_reasoning", "")

        # Save output
        output = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "n_experiences": len(experiences),
            "scores": {eid: round(s, 4) for eid, s in scores.items()},
            "psi_proxy": {eid: round(p, 4) for eid, p in psi_proxy.items()},
            "ranked_ids": ranked_ids,
            "remove_ids": remove_ids,
            "reasoning": reasoning,
            "individual_scores": result.get("individual_scores", []),
            "raw_response": raw_response,
            "experiences": experiences,
        }
        self._save(output, label)

        logger.info(
            "LLM judge [%s]: %d experiences scored. "
            "Top: %s. Remove: %s",
            label, len(scores),
            ranked_ids[:3] if ranked_ids else [],
            remove_ids,
        )

        return output

    def _format_experiences(self, experiences: dict[str, str]) -> str:
        lines = []
        for exp_id, content in experiences.items():
            lines.append(f"[{exp_id}]\n{content}\n")
        return "\n".join(lines)

    def _format_rollout_context(self, rollouts: list = None) -> str:
        if not rollouts:
            return ""
        # Pick up to 3 failed rollouts (reward=0) as examples of struggles
        failed = [
            r for r in rollouts
            if hasattr(r, "reward") and r.reward == 0
        ][:3]
        if not failed:
            return ""
        examples = []
        for i, r in enumerate(failed, 1):
            question = getattr(r, "raw_question", "")[:200]
            reasoning = getattr(r, "reasoning", "") or ""
            examples.append(
                f"Example {i}:\n"
                f"Question: {question}\n"
                f"Agent reasoning: {reasoning[:300]}\n"
                f"(reward=0, answered incorrectly)"
            )
        return _ROLLOUT_CONTEXT_PROMPT.format(
            rollout_examples="\n\n".join(examples)
        )

    def _save(self, output: dict, label: str) -> None:
        fname = f"judge_{label}.json" if label else "judge_latest.json"
        path = os.path.join(self.log_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        # Also save a clean CSV for easy inspection
        csv_path = os.path.join(self.log_dir, f"judge_{label}.csv" if label else "judge_latest.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("rank,experience_id,score,psi_proxy,remove,content\n")
            ranked = output["ranked_ids"]
            remove_set = set(output["remove_ids"])
            for rank, eid in enumerate(ranked, 1):
                score = output["scores"].get(eid, 5.0)
                psi = output["psi_proxy"].get(eid, 0.0)
                remove = eid in remove_set
                content = output["experiences"].get(eid, "").replace('"', '""').replace("\n", " ")
                f.write(f'{rank},{eid},{score:.1f},{psi:.4f},{remove},"{content}"\n')

        logger.info("LLM judge output saved to %s", path)


# ---------------------------------------------------------------------------
# Drop-in replacement scorer for iterative_shapley_grpo.py
# ---------------------------------------------------------------------------

class LLMJudgeScorer:
    """
    Drop-in replacement for QuickExperienceScorer in iterative_shapley_grpo.py.

    Returns psi_proxy values in [-1, +1] compatible with the Shapley
    selection logic (_select_top_k, _prune_by_shapley).

    Usage:
        scorer = LLMJudgeScorer(
            llm_config=config.evaluation.agent,
            agent_objective=config.practice.agent_objective,
            learning_objective=config.practice.learning_objective,
        )
        scores = await scorer.score(candidate_experiences, round_idx)
    """

    def __init__(
        self,
        llm_config,
        agent_objective: str,
        learning_objective: str,
        log_dir: str = "logs/llm_judge",
    ):
        self.judge = LLMExperienceJudge(
            llm_config=llm_config,
            agent_objective=agent_objective,
            learning_objective=learning_objective,
            log_dir=log_dir,
        )

    async def score(
        self,
        candidate_experiences: dict[str, str],
        round_idx: int,
        rollouts: list = None,
        step: int = 0,
    ) -> dict[str, float]:
        """
        Score experiences using LLM judge.
        Returns {exp_id: psi_proxy} in [-1, +1].
        Compatible with ExperienceShapley score output format.
        """
        result = await self.judge.judge(
            experiences=candidate_experiences,
            rollouts=rollouts,
            label=f"step{step}_round{round_idx}",
        )
        return result["psi_proxy"]
