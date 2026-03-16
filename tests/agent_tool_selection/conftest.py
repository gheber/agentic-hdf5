"""
Fixtures for agent tool selection tests.

These tests invoke the `claude` CLI in print mode and require:
  - claude CLI installed and authenticated
  - Running with: pytest -m agent
  - Must NOT be run from inside a Claude Code session (no nesting)
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.search_tools import load_catalog

PROJECT_ROOT = Path(__file__).parent.parent.parent


@pytest.fixture(scope="session", autouse=True)
def check_claude_available():
    """Skip all tests if claude CLI is not available or we're inside a session."""
    if os.environ.get("CLAUDECODE"):
        pytest.skip(
            "Cannot run agent tests inside a Claude Code session. "
            "Run from a normal terminal: pytest -m agent"
        )
    if not shutil.which("claude"):
        pytest.skip("claude CLI not found on PATH")


@pytest.fixture(scope="session")
def catalog():
    """Load the tool catalog."""
    catalog_path = PROJECT_ROOT / "tools" / "tool_catalog.json"
    return load_catalog(catalog_path)


@pytest.fixture(scope="session")
def tool_names(catalog):
    """List of all tool names in the catalog."""
    return [t["name"] for t in catalog["tools"]]


SYSTEM_PROMPT = """\
You are an HDF5 data assistant. The user will ask you to perform a task on an \
HDF5 file. You have access to the following tools (and ONLY these tools):

{tool_descriptions}

Based on the user's request, determine which single tool is most appropriate. \
Respond with ONLY this exact format, nothing else:

SELECTED_TOOL: <tool_name>"""


# --- Prompt modes ---

def _build_verbose_prompt(catalog):
    """Build verbose system prompt: description + detailed_description + keywords + use_cases."""
    tool_lines = []
    for tool in catalog["tools"]:
        keywords = ", ".join(tool.get("search_keywords", []))
        use_cases = "; ".join(tool.get("use_cases", []))
        detailed = tool.get("detailed_description", "")
        entry = f"- {tool['name']}: {tool['description']}"
        if detailed:
            entry += f" {detailed}"
        entry += f" (keywords: {keywords})"
        if use_cases:
            entry += f" (use cases: {use_cases})"
        tool_lines.append(entry)
    return "\n".join(tool_lines)


def _build_original_prompt(catalog):
    """Build original system prompt: description + keywords only."""
    tool_lines = []
    for tool in catalog["tools"]:
        keywords = ", ".join(tool.get("search_keywords", []))
        tool_lines.append(
            f"- {tool['name']}: {tool['description']} "
            f"(keywords: {keywords})"
        )
    return "\n".join(tool_lines)


PROMPT_MODES = {
    "original": _build_original_prompt,
    "verbose": _build_verbose_prompt,
}


def _build_system_prompt(catalog, mode="verbose"):
    """Build system prompt using the specified mode."""
    builder = PROMPT_MODES.get(mode, _build_verbose_prompt)
    descriptions = builder(catalog)
    return SYSTEM_PROMPT.format(tool_descriptions=descriptions)


DEFAULT_MODEL = "haiku"


def pytest_addoption(parser):
    parser.addoption(
        "--model",
        action="store",
        default=DEFAULT_MODEL,
        help=f"Claude model to use for agent tests (default: {DEFAULT_MODEL})",
    )
    parser.addoption(
        "--prompt-mode",
        action="store",
        default="verbose",
        choices=["original", "verbose"],
        help="Prompt mode: original (desc+keywords only), verbose (all catalog fields)",
    )


@pytest.fixture(scope="session")
def model(request):
    """The Claude model to use, from --model flag."""
    return request.config.getoption("--model")


@pytest.fixture(scope="session")
def prompt_mode(request):
    """The prompt mode to use, from --prompt-mode flag."""
    return request.config.getoption("--prompt-mode")


def run_agent_turn(user_prompt, catalog, model=DEFAULT_MODEL, prompt_mode="verbose", timeout=60):
    """
    Send a prompt to the claude CLI in print mode and return the response text.

    Args:
        user_prompt: The natural language prompt to send.
        catalog: The loaded tool catalog dict.
        model: Claude model alias or ID.
        prompt_mode: One of "original", "verbose", "compressed".
        timeout: Subprocess timeout in seconds.

    Returns:
        The CLI text response (str).

    Raises:
        RuntimeError: If the CLI invocation fails.
    """
    system_prompt = _build_system_prompt(catalog, mode=prompt_mode)

    cmd = [
        "claude",
        "-p",
        "--no-session-persistence",
        "--model", model,
        "--tools", "",
        "--system-prompt", system_prompt,
        user_prompt,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"claude CLI failed (exit {result.returncode}):\n"
            f"stderr: {result.stderr}\n"
            f"stdout: {result.stdout}"
        )

    return result.stdout.strip()


def extract_selected_tool(response_text):
    """
    Extract the tool name from a response containing 'SELECTED_TOOL: <name>'.

    Returns the tool name string, or None if not found.
    """
    for line in response_text.splitlines():
        line = line.strip()
        if line.upper().startswith("SELECTED_TOOL:"):
            return line.split(":", 1)[1].strip()
    return None
