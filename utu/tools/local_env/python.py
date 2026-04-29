"""
- [ ] polish _execute_python_code_sync
"""
from IPython.core.interactiveshell import InteractiveShell
from traitlets.config import Config
import asyncio
import base64
import contextlib
import glob
import io
import os
import re
import traceback
from typing import TYPE_CHECKING

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from IPython.core.interactiveshell import InteractiveShell
    from traitlets.config.loader import Config

    matplotlib.use("Agg")
except ImportError:
    pass

if TYPE_CHECKING:
    from IPython.core.history import HistoryManager
    from traitlets.config.loader import Config as BaseConfig

    class Config(BaseConfig):
        HistoryManager: HistoryManager


# Used to clean ANSI escape sequences
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
MAX_MEMORY_GB = 16
CODE_HEADER = f"""
import resource
try:
    memory_limit_bytes = {MAX_MEMORY_GB * 1024 * 1024 * 1024}
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
except (ValueError, resource.error):
    pass
"""


def create_ipython_shell():
    """
    Create a persistent IPython shell instance for reuse across multiple executions.

    Returns:
        InteractiveShell: A configured IPython shell instance
    """
    InteractiveShell.clear_instance()

    config = Config()
    config.HistoryManager.enabled = False
    config.HistoryManager.hist_file = ":memory:"

    shell = InteractiveShell.instance(config=config)

    if hasattr(shell, "history_manager"):
        shell.history_manager.enabled = False

    return shell


def cleanup_ipython_shell(shell):
    """
    Clean up an IPython shell instance.

    Args:
        shell: The IPython shell instance to clean up
    """
    if shell is None:
        return

    try:
        shell.atexit_operations = lambda: None
        if hasattr(shell, "history_manager") and shell.history_manager:
            shell.history_manager.enabled = False
            shell.history_manager.end_session = lambda: None
        InteractiveShell.clear_instance()
    except Exception:  # pylint: disable=broad-except
        pass


def execute_python_code_sync(code: str, workdir: str, shell=None):
    """
    Synchronous execution of Python code.
    This function is intended to be run in a separate thread.

    Args:
        code: Python code to execute
        workdir: Working directory for execution
        shell: Optional existing IPython shell instance to reuse
    """
    original_dir = os.getcwd()
    shell_was_passed = shell is not None  # Track if shell was passed in
    try:
        # Clean up code format
        code_clean = code.strip()
        if code_clean.startswith("```python"):
            code_clean = code_clean.split("```python")[1].split("```")[0].strip()
        code_clean = CODE_HEADER + code_clean

        # Create and change to working directory
        os.makedirs(workdir, exist_ok=True)
        os.chdir(workdir)

        # Get file list before execution
        files_before = set(glob.glob("*"))

        # Create a new IPython shell instance or reuse existing one
        if shell is None:
            InteractiveShell.clear_instance()

            config = Config()
            config.HistoryManager.enabled = False
            config.HistoryManager.hist_file = ":memory:"

            shell = InteractiveShell.instance(config=config)

        if hasattr(shell, "history_manager"):
            shell.history_manager.enabled = False

        output = io.StringIO()
        error_output = io.StringIO()

        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error_output):
            shell.run_cell(code_clean)

            if plt.get_fignums():
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format="png")
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
                plt.close()

                image_name = "output_image.png"
                counter = 1
                while os.path.exists(image_name):
                    image_name = f"output_image_{counter}.png"
                    counter += 1

                with open(image_name, "wb") as f:
                    f.write(base64.b64decode(img_base64))

        stdout_result = output.getvalue()
        stderr_result = error_output.getvalue()

        stdout_result = ANSI_ESCAPE.sub("", stdout_result)
        stderr_result = ANSI_ESCAPE.sub("", stderr_result)

        files_after = set(glob.glob("*"))
        new_files = list(files_after - files_before)
        new_files = [os.path.join(workdir, f) for f in new_files]

        # Don't clear the shell instance if it was passed in (reuse mode)
        # Only clear if we created a new one
        if not shell_was_passed:
            try:
                shell.atexit_operations = lambda: None
                if hasattr(shell, "history_manager") and shell.history_manager:
                    shell.history_manager.enabled = False
                    shell.history_manager.end_session = lambda: None
                InteractiveShell.clear_instance()
            except Exception:  # pylint: disable=broad-except
                pass

        success = True
        if "Error" in stderr_result or ("Error" in stdout_result and "Traceback" in stdout_result):
            success = False
        message = "Code execution completed, no output"
        if stdout_result.strip():
            message = f"Code execution completed\nOutput:\n{stdout_result.strip()}"

        return {
            "workdir": workdir,
            "success": success,
            "message": message,
            "status": True,
            "files": new_files,
            "error": stderr_result.strip(),
        }
    except Exception as e:  # pylint: disable=broad-except
        return {
            "workdir": workdir,
            "success": False,
            "message": f"Code execution failed, error message:\n{str(e)},\nTraceback:{traceback.format_exc()}",
            "status": False,
            "files": [],
            "error": str(e),
        }
    finally:
        os.chdir(original_dir)


# ---------------------------------------------------------------------------
# Subprocess-based execution (segfault-safe)
# ---------------------------------------------------------------------------
# The original thread-based approach (run_in_executor) shares memory with
# the parent process.  When LLM-generated code triggers a native crash
# (SIGSEGV) in sympy, numpy, or a CUDA library, the segfault propagates up
# and kills the entire training process.
#
# Running code in a child subprocess fully isolates the crash: the child
# dies, the parent catches the non-zero exit code, and execution continues.
#
# Trade-off: the persistent IPython shell cannot be shared across subprocess
# boundaries, so each call starts with a fresh namespace.  This is acceptable
# because the rollout agent already treats each tool call as independent.
# ---------------------------------------------------------------------------

import json
import sys
import tempfile

_RUNNER_SCRIPT = """
import sys, os, io, glob, base64, contextlib, re, traceback

ANSI_ESCAPE = re.compile(r"\\x1b\\[[0-9;]*[a-zA-Z]")

# Memory cap — identical to the in-process version
import resource
try:
    _limit = {max_memory_bytes}
    resource.setrlimit(resource.RLIMIT_AS, (_limit, _limit))
except Exception:
    pass

code = {code_repr}
workdir = {workdir_repr}

os.makedirs(workdir, exist_ok=True)
os.chdir(workdir)
files_before = set(glob.glob("*"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    has_plt = True
except ImportError:
    has_plt = False

output      = io.StringIO()
err_output  = io.StringIO()
ns = {{"__name__": "__main__"}}

try:
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(err_output):
        exec(compile(code, "<string>", "exec"), ns)
        if has_plt and plt.get_fignums():
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            img_name = "output_image.png"
            c = 1
            while os.path.exists(img_name):
                img_name = f"output_image_{{c}}.png"
                c += 1
            with open(img_name, "wb") as f:
                f.write(buf.getvalue())
except Exception:
    err_output.write(traceback.format_exc())

stdout_result = ANSI_ESCAPE.sub("", output.getvalue())
stderr_result = ANSI_ESCAPE.sub("", err_output.getvalue())

files_after = set(glob.glob("*"))
new_files   = [os.path.join(workdir, f) for f in files_after - files_before]

success = not ("Error" in stderr_result or
               ("Error" in stdout_result and "Traceback" in stdout_result))
message = ("Code execution completed\\nOutput:\\n" + stdout_result.strip()
           if stdout_result.strip() else "Code execution completed, no output")

import json
print(json.dumps({{
    "workdir": workdir,
    "success": success,
    "message": message,
    "status": True,
    "files": new_files,
    "error": stderr_result.strip(),
}}))
"""


async def execute_python_code_async(code: str, workdir: str, timeout: int = 30, shell=None) -> dict:
    """
    Asynchronous execution of Python code in an isolated subprocess.

    The subprocess approach prevents native crashes (SIGSEGV from sympy /
    numpy / CUDA) from killing the parent training process.  The `shell`
    argument is accepted for API compatibility but ignored — each call
    runs in a fresh Python interpreter.

    Args:
        code: Python code to execute.
        workdir: Working directory for execution.
        timeout: Hard wall-clock timeout in seconds (default 30).
        shell: Ignored (kept for API compatibility with persistent-shell mode).
    """
    # Clean up markdown fences the LLM sometimes wraps code in
    code_clean = code.strip()
    if code_clean.startswith("```python"):
        code_clean = code_clean.split("```python")[1].split("```")[0].strip()
    elif code_clean.startswith("```"):
        code_clean = code_clean.split("```")[1].split("```")[0].strip()

    os.makedirs(workdir, exist_ok=True)

    # Build a self-contained runner script — no pickle, no IPC complexity
    runner_src = _RUNNER_SCRIPT.format(
        max_memory_bytes=MAX_MEMORY_GB * 1024 * 1024 * 1024,
        code_repr=repr(code_clean),
        workdir_repr=repr(str(workdir)),
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="utu_exec_"
    ) as tmp:
        tmp.write(runner_src)
        tmp_path = tmp.name

    loop = asyncio.get_running_loop()
    try:
        proc = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: __import__("subprocess").run(
                    [sys.executable, tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout + 5,   # subprocess hard kill after timeout+5s
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},  # no GPU in child
                ),
            ),
            timeout=timeout + 10,          # asyncio outer guard
        )

        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        if proc.returncode != 0:
            # Child crashed (segfault = -11, OOM = -9, etc.)
            signal_desc = {
                -11: "Segmentation fault (SIGSEGV) — likely a native library crash",
                -9:  "Killed (SIGKILL) — likely OOM",
                -6:  "Aborted (SIGABRT)",
            }.get(proc.returncode, f"exit code {proc.returncode}")

            return {
                "workdir": workdir,
                "success": False,
                "message": f"Code execution failed: {signal_desc}",
                "status": False,
                "files": [],
                "error": stderr[:1000] if stderr else signal_desc,
            }

        # Parse JSON result written by the runner
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            # Runner produced non-JSON stdout (e.g. print before json.dumps)
            return {
                "workdir": workdir,
                "success": bool(stdout),
                "message": f"Code execution completed\nOutput:\n{stdout[:2000]}",
                "status": True,
                "files": [],
                "error": stderr[:500] if stderr else "",
            }

    except asyncio.TimeoutError:
        return {
            "workdir": workdir,
            "success": False,
            "message": f"Code execution timed out after {timeout}s",
            "status": False,
            "files": [],
            "error": f"Timeout after {timeout}s",
        }
    except Exception as e:
        return {
            "workdir": workdir,
            "success": False,
            "message": f"Executor error: {e}",
            "status": False,
            "files": [],
            "error": str(e),
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass