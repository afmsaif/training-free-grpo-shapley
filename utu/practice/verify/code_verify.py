"""
Verification functions for SWE-bench and AppWorld datasets.
Place this file at: utu/practice/verify/code_verify.py

Uses DeepSeek-R1-32B as the judge (port 8001) to evaluate whether
the agent's response correctly addresses the task.
"""

import os
import re
import json
import asyncio
from openai import OpenAI


# ---------------------------------------------------------------------------
# LLM judge client (DeepSeek on port 8001)
# ---------------------------------------------------------------------------

def _get_judge_client():
    """Get OpenAI-compatible client pointing to the judge model."""
    base_url = os.environ.get("JUDGE_LLM_BASE_URL", "http://localhost:8001/v1")
    api_key  = os.environ.get("JUDGE_LLM_API_KEY", "xxx")
    model    = os.environ.get("JUDGE_LLM_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    return OpenAI(base_url=base_url, api_key=api_key), model


def _judge(prompt: str, max_tokens: int = 256) -> str:
    """Call the LLM judge synchronously."""
    try:
        client, model = _get_judge_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# SWE-bench verification
# ---------------------------------------------------------------------------

_SWEBENCH_JUDGE_PROMPT = """You are evaluating whether an AI agent correctly resolved a GitHub issue.

ISSUE DESCRIPTION:
{issue}

GOLD PATCH (reference solution):
{gold_patch}

AGENT'S RESPONSE:
{response}

Evaluate whether the agent's response:
1. Contains a git patch in the correct diff format
2. Addresses the same files and functions as the gold patch
3. Makes logically correct changes to fix the described issue

Respond with ONLY a JSON object:
{{"score": 0.0 or 0.5 or 1.0, "reasoning": "one sentence explanation"}}

Scoring:
- 1.0: patch is correct and complete
- 0.5: patch is partially correct (right direction but missing something)
- 0.0: patch is wrong, missing, or doesn't address the issue
"""

def swebench_verify_func(sample, timeout_score=0, **kwargs):
    """
    Verify a SWE-bench response using LLM-as-judge.

    Args:
        sample: EvaluationSample with .response and .correct_answer (gold patch)
    Returns:
        {"reward": float, "reasoning": str}
    """
    response = getattr(sample, "response", "") or ""
    gold_patch = getattr(sample, "correct_answer", "") or ""
    question = getattr(sample, "raw_question", "") or ""

    if not response.strip():
        return {"reward": 0.0, "reasoning": "Empty response"}

    # Fast check: does it contain a patch?
    has_patch = "<patch>" in response or "diff --git" in response or "@@" in response
    if not has_patch:
        return {"reward": 0.0, "reasoning": "No patch found in response"}

    # Extract issue description from question (first 500 chars after "GitHub Issue:")
    issue_match = re.search(r"GitHub Issue:\n(.+?)(?:\n\nRelevant|$)", question, re.DOTALL)
    issue_text = issue_match.group(1)[:500] if issue_match else question[:500]

    prompt = _SWEBENCH_JUDGE_PROMPT.format(
        issue=issue_text,
        gold_patch=gold_patch[:1000],
        response=response[:1500],
    )

    judge_response = _judge(prompt)

    try:
        # Parse JSON from judge response
        json_match = re.search(r"\{.*?\}", judge_response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            score = float(result.get("score", 0.0))
            reasoning = result.get("reasoning", "")
            return {"reward": min(max(score, 0.0), 1.0), "reasoning": reasoning}
    except Exception:
        pass

    # Fallback: keyword match on gold patch files
    gold_files = re.findall(r"diff --git a/(\S+)", gold_patch)
    response_files = re.findall(r"diff --git a/(\S+)", response)
    overlap = set(gold_files) & set(response_files)
    if overlap:
        return {"reward": 0.5, "reasoning": f"Modified correct files: {list(overlap)[:2]}"}

    return {"reward": 0.0, "reasoning": f"Judge response: {judge_response[:100]}"}


# ---------------------------------------------------------------------------
# AppWorld verification
# ---------------------------------------------------------------------------

_APPWORLD_JUDGE_PROMPT = """You are evaluating whether an AI agent correctly completed a digital task.

TASK:
{task}

EXPECTED SOLUTION APPROACH:
{expected}

AGENT'S RESPONSE:
{response}

Evaluate whether the agent's response:
1. Attempts to use the correct APIs/tools for the task
2. Implements the right logic (correct amounts, recipients, filters, etc.)
3. Would actually complete the task if executed

Respond with ONLY a JSON object:
{{"score": 0.0 or 0.5 or 1.0, "reasoning": "one sentence explanation"}}

Scoring:
- 1.0: correctly addresses all parts of the task
- 0.5: partially correct (gets main goal but misses details)
- 0.0: wrong approach or doesn't address the task
"""

def appworld_verify_func(sample, timeout_score=0, **kwargs):
    """
    Verify an AppWorld response using LLM-as-judge.

    Args:
        sample: EvaluationSample with .response and .correct_answer
    Returns:
        {"reward": float, "reasoning": str}
    """
    response = getattr(sample, "response", "") or ""
    expected = getattr(sample, "correct_answer", "") or ""
    question = getattr(sample, "raw_question", "") or ""

    if not response.strip():
        return {"reward": 0.0, "reasoning": "Empty response"}

    # Fast check: does response contain any code?
    has_code = any(kw in response for kw in [
        "def ", "import ", "api.", ".get(", ".post(", "execute_python",
        "python", "```", "client.", "requests."
    ])
    if not has_code:
        return {"reward": 0.0, "reasoning": "No code found in response"}

    # Extract task from question
    task_match = re.search(r"Task: (.+?)(?:\n\nWrite|$)", question, re.DOTALL)
    task_text = task_match.group(1)[:400] if task_match else question[:400]

    prompt = _APPWORLD_JUDGE_PROMPT.format(
        task=task_text,
        expected=expected[:300],
        response=response[:1500],
    )

    judge_response = _judge(prompt)

    try:
        json_match = re.search(r"\{.*?\}", judge_response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            score = float(result.get("score", 0.0))
            reasoning = result.get("reasoning", "")
            return {"reward": min(max(score, 0.0), 1.0), "reasoning": reasoning}
    except Exception:
        pass

    return {"reward": 0.0, "reasoning": f"Judge response: {judge_response[:100]}"}
