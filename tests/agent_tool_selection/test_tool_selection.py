"""
Agent tool selection evaluation tests.

Tests whether Claude correctly identifies the right HDF5 tool given
natural language prompts. Invokes the claude CLI.

Run with (from a normal terminal, NOT inside Claude Code):
    pytest -m agent tests/agent_tool_selection/ -v

Requires:
    - claude CLI installed and authenticated
"""

import pytest

from .conftest import run_agent_turn, extract_selected_tool

# (user_prompt, list_of_acceptable_tool_names)
TOOL_SELECTION_CASES = [
    # Direct/obvious - chunking
    (
        "I need to change the chunk size of my dataset to improve read performance.",
        ["rechunk_dataset"],
    ),
    # Indirect/descriptive - compression
    (
        "My HDF5 file is 50GB and I need to shrink it down for sharing.",
        ["apply_filter_dataset"],
    ),
    # Visualization
    (
        "Can you make a heatmap of the temperature array in my file?",
        ["visualize"],
    ),
    # Inspection
    (
        "What's the shape, dtype, and compression settings of /sensors/pressure?",
        ["get_object_metadata"],
    ),
    # Semantic search
    (
        "I have a huge file with hundreds of datasets. Find the ones related to ocean salinity.",
        ["query_semantic_metadata"],
    ),
    # Workflow / batch documentation
    (
        "I just imported a new file and none of the datasets have descriptions. "
        "I want to auto-generate documentation for everything.",
        ["collect_objects_for_smd"],
    ),
    # Ambiguous / multi-tool acceptable
    (
        "What does the dataset at /experiment/run3/output represent?",
        ["read_semantic_metadata", "get_object_metadata"],
    ),
]

CASE_IDS = [
    "chunking",
    "compression",
    "visualization",
    "inspection",
    "semantic_search",
    "batch_documentation",
    "ambiguous_read_vs_inspect",
]


@pytest.mark.agent
class TestToolSelection:
    """Evaluate agent tool selection accuracy across prompt categories."""

    @pytest.mark.parametrize(
        "user_prompt,expected_tools",
        TOOL_SELECTION_CASES,
        ids=CASE_IDS,
    )
    def test_individual_selection(
        self, catalog, model, user_prompt, expected_tools
    ):
        """Test that Claude selects an acceptable tool for each prompt."""
        response = run_agent_turn(user_prompt, catalog, model=model)
        selected = extract_selected_tool(response)

        assert selected is not None, (
            f"Claude did not return SELECTED_TOOL in response:\n{response}"
        )
        assert selected in expected_tools, (
            f"Expected one of {expected_tools}, got '{selected}'.\n"
            f"Full response:\n{response}"
        )

