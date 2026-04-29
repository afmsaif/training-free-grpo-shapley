#!/usr/bin/env python3
"""
Step 1: Download and prepare SWE-bench Lite and AppWorld datasets
for Training-Free GRPO.

SWE-bench Lite: 300 GitHub issue resolution tasks (Python repos).
  Verification: LLM judge checks if generated patch addresses the issue.
  No Docker/unit-test execution needed.

AppWorld: 750 interactive coding tasks across 9 simulated apps.
  Verification: LLM judge checks if the API call sequence is correct.
  Simplified version without the full AppWorld engine.

Usage:
    python scripts/prepare_datasets.py --dataset swebench
    python scripts/prepare_datasets.py --dataset appworld
    python scripts/prepare_datasets.py --dataset both
"""

import argparse
import json
import os
import sys

sys.path.insert(0, ".")


# ---------------------------------------------------------------------------
# SWE-bench Lite
# ---------------------------------------------------------------------------

def prepare_swebench(output_path: str = "data/swebench_lite.jsonl", max_samples: int = 500):
    """
    Download SWE-bench Lite from HuggingFace and convert to GRPO format.

    Each sample becomes:
        question = "Repository: {repo}\n\nIssue:\n{problem_statement}\n\n
                    Relevant code:\n{text}\n\nGenerate a git patch to fix this issue."
        answer   = gold patch (used by LLM judge for grading)
    """
    print("Downloading SWE-bench Lite...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        os.system("pip install datasets --break-system-packages -q")
        from datasets import load_dataset

    # Use oracle version which includes retrieved code context
    ds = load_dataset("princeton-nlp/SWE-bench_Lite_oracle", split="test")
    print(f"  Loaded {len(ds)} SWE-bench Lite samples")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for item in ds:
            if count >= max_samples:
                break

            # Build the question — include repo, issue, and oracle code context
            question = (
                f"Repository: {item['repo']}\n\n"
                f"GitHub Issue:\n{item['problem_statement']}\n\n"
                f"Relevant code context:\n{item['text'][:3000]}\n\n"
                f"Generate a git patch in the following format to fix this issue:\n"
                f"<patch>\n"
                f"diff --git a/path/to/file.py b/path/to/file.py\n"
                f"--- a/path/to/file.py\n"
                f"+++ b/path/to/file.py\n"
                f"@@ -line,count +line,count @@\n"
                f" context line\n"
                f"-removed line\n"
                f"+added line\n"
                f"</patch>"
            )

            sample = {
                "dataset": "SWEBench",
                "source": "training_free_grpo",
                "question": question,
                "answer": item["patch"],          # gold patch for LLM judge
                "metadata": {
                    "instance_id": item["instance_id"],
                    "repo": item["repo"],
                    "base_commit": item["base_commit"],
                },
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Saved {count} samples to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# AppWorld (simplified — no execution engine required)
# ---------------------------------------------------------------------------

def prepare_appworld(output_path: str = "data/appworld.jsonl", max_samples: int = 500):
    """
    Download AppWorld tasks and convert to GRPO format.

    AppWorld tasks involve writing Python code to call APIs across 9 apps:
    Spotify, Amazon, Venmo, Gmail, Google Calendar, Splitwise, etc.

    Simplified version: the agent generates Python API call code.
    Verification: LLM judge checks if the code correctly addresses the task.

    Uses the HuggingFace version of AppWorld tasks.
    """
    print("Downloading AppWorld dataset...")
    try:
        from datasets import load_dataset
    except ImportError:
        os.system("pip install datasets --break-system-packages -q")
        from datasets import load_dataset

    # AppWorld tasks on HuggingFace
    try:
        ds = load_dataset("hamishivi/appworld_env_train", split="train")
        print(f"  Loaded {len(ds)} AppWorld samples from HuggingFace")
        use_hf = True
    except Exception as e:
        print(f"  HuggingFace version not available ({e}), using synthetic tasks...")
        use_hf = False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = 0

    if use_hf:
        with open(output_path, "w", encoding="utf-8") as f:
            for item in ds:
                if count >= max_samples:
                    break
                # Adapt HF AppWorld format
                question = item.get("task", item.get("question", str(item)))
                answer = item.get("solution", item.get("answer", ""))
                sample = {
                    "dataset": "AppWorld",
                    "source": "training_free_grpo",
                    "question": (
                        f"You are an AI assistant with access to Python APIs for "
                        f"various apps (Spotify, Amazon, Gmail, Venmo, etc.).\n\n"
                        f"Task: {question}\n\n"
                        f"Write Python code using the available APIs to complete this task. "
                        f"Use the execute_python_code tool to run your code."
                    ),
                    "answer": answer,
                }
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                count += 1
    else:
        # Fallback: create representative AppWorld-style tasks from the paper
        _create_synthetic_appworld_tasks(output_path, max_samples)
        count = max_samples

    print(f"  Saved {count} samples to {output_path}")
    return output_path


def _create_synthetic_appworld_tasks(output_path: str, n: int):
    """
    Create representative AppWorld-style tasks based on the paper's task categories.
    These cover the same app domains and difficulty levels as the real benchmark.
    """
    templates = [
        # Spotify tasks
        {
            "question": "Using the Spotify API, find the top 5 most played songs by Taylor Swift "
                       "and create a new playlist called 'Taylor's Hits' with those songs for user 'alice'.",
            "answer": "Use spotify.search(artist='Taylor Swift'), get top tracks, "
                     "create playlist, add tracks.",
        },
        {
            "question": "Find all songs in the user's 'Workout' playlist on Spotify that have "
                       "a tempo above 120 BPM and move them to a new playlist called 'High Energy'.",
            "answer": "Get playlist tracks, check audio features for tempo, create new playlist, add matching tracks.",
        },
        # Amazon tasks
        {
            "question": "The user wants to return an Amazon order placed in the last 30 days "
                       "for 'Wireless Headphones'. Find the order, initiate a return, and send "
                       "a confirmation email via Gmail.",
            "answer": "Search orders by date range and item name, initiate return request, send email.",
        },
        # Venmo/financial tasks
        {
            "question": "Split a $120 dinner bill equally among 4 friends (alice, bob, carol, dave) "
                       "using Venmo. Send each person a request for their share with the note 'Dinner 4/15'.",
            "answer": "Calculate $30 per person, send Venmo requests to each friend.",
        },
        # Gmail tasks
        {
            "question": "Search Gmail for all emails from 'notifications@github.com' received this week "
                       "and summarize the key action items in a Google Doc.",
            "answer": "Search Gmail with sender filter and date range, extract content, create doc.",
        },
        # Multi-app tasks
        {
            "question": "A user has a meeting on Google Calendar tomorrow at 2pm. Find the meeting, "
                       "get all attendees, and send them each a Venmo request for $15 for 'Team lunch'.",
            "answer": "Get calendar event, extract attendees, send Venmo requests.",
        },
        # Splitwise tasks
        {
            "question": "Get all outstanding balances in the user's Splitwise groups and send "
                       "reminder messages via Gmail to people who owe more than $50.",
            "answer": "Get Splitwise debts, filter by amount, send Gmail reminders.",
        },
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(n):
            template = templates[i % len(templates)]
            sample = {
                "dataset": "AppWorld",
                "source": "training_free_grpo",
                "question": (
                    f"You are an AI assistant with access to Python APIs for "
                    f"various apps (Spotify, Amazon, Gmail, Venmo, Google Calendar, Splitwise).\n\n"
                    f"Task: {template['question']}\n\n"
                    f"Write Python code using the available APIs to complete this task. "
                    f"Use the execute_python_code tool to run your code."
                ),
                "answer": template["answer"],
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Upload to GRPO database
# ---------------------------------------------------------------------------

def upload_dataset(jsonl_path: str, dataset_name: str):
    """Upload prepared JSONL to the GRPO database."""
    print(f"\nUploading {dataset_name} to GRPO database...")
    cmd = (
        f"python scripts/data/upload_dataset.py "
        f"--file_path {jsonl_path} "
        f"--dataset_name {dataset_name}"
    )
    ret = os.system(cmd)
    if ret == 0:
        print(f"  ✓ {dataset_name} uploaded successfully")
    else:
        print(f"  ✗ Upload failed. Run manually:\n    {cmd}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["swebench", "appworld", "both"],
        default="both", help="Which dataset to prepare",
    )
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--upload", action="store_true", default=True,
                        help="Upload to GRPO database after preparing")
    parser.add_argument("--no_upload", dest="upload", action="store_false")
    args = parser.parse_args()

    if args.dataset in ("swebench", "both"):
        path = prepare_swebench(max_samples=args.max_samples)
        if args.upload:
            upload_dataset(path, "SWEBench")

    if args.dataset in ("appworld", "both"):
        path = prepare_appworld(max_samples=args.max_samples)
        if args.upload:
            upload_dataset(path, "AppWorld")

    print("\nDone. Next steps:")
    print("  1. Create verification functions:")
    print("     utu/practice/verify/swebench_verify.py")
    print("     utu/practice/verify/appworld_verify.py")
    print("  2. Create eval configs:")
    print("     configs/eval/swebench/swebench_eval.yaml")
    print("     configs/eval/appworld/appworld_eval.yaml")
    print("  3. Create practice configs:")
    print("     configs/practice/swebench_practice.yaml")
    print("     configs/practice/appworld_practice.yaml")
    print("  4. Run: python scripts/run_full_pipeline.py --dataset swebench")
