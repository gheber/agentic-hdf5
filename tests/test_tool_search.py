"""
Tests for the tool search CLI functionality.

Run with: pytest tests/test_tool_search.py
"""

import pytest
import sys
from pathlib import Path

# Add tools directory to path so we can import search_tools
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.search_tools import search_tools, load_catalog, MAX_RESULTS


@pytest.fixture
def catalog():
    """Load the tool catalog for testing."""
    catalog_path = Path(__file__).parent.parent / "tools" / "tool_catalog.json"
    return load_catalog(catalog_path)


@pytest.mark.parametrize("query,expected_tool", [
    ("chunk", "rechunk_dataset"),
    ("metadata", "get_object_metadata"),
])
def test_search_keyword(catalog, query, expected_tool):
    """Test searching for keywords returns expected tools."""
    results = search_tools(query, catalog=catalog)

    assert len(results) > 0, f"Should find at least one tool for '{query}'"

    tool_names = [tool['name'] for tool in results]
    assert expected_tool in tool_names, f"Should find {expected_tool} for '{query}' query"

    # Verify the tool has expected fields
    found_tool = next(t for t in results if t['name'] == expected_tool)
    assert 'description' in found_tool
    assert 'import' in found_tool
    assert 'search_keywords' in found_tool


def test_empty_string_raises_error(catalog):
    """Test that empty query string raises ValueError."""
    with pytest.raises(ValueError, match="must be a non-empty string"):
        search_tools("", catalog=catalog)


def test_gibberish_returns_no_tools(catalog):
    """Test that gibberish query returns no results."""
    results = search_tools("xyzzyqwertyzzzabc123notarealtool", catalog=catalog)
    assert len(results) == 0, "Gibberish query should return no results"


def test_case_insensitive_search(catalog):
    """Test that search is case-insensitive."""
    results_lower = search_tools("chunk", catalog=catalog)
    results_upper = search_tools("CHUNK", catalog=catalog)
    results_mixed = search_tools("ChUnK", catalog=catalog)

    names_lower = [t['name'] for t in results_lower]
    names_upper = [t['name'] for t in results_upper]
    names_mixed = [t['name'] for t in results_mixed]

    assert names_lower == names_upper == names_mixed, "Search should be case-insensitive"


def test_multi_word_query(catalog):
    """Test searching with multiple words."""
    results = search_tools("semantic search", catalog=catalog)

    assert len(results) > 0, "Should find tools for multi-word query"

    tool_names = [tool['name'] for tool in results]
    assert any('semantic' in name or 'query' in name or 'vectorize' in name
               for name in tool_names), "Should find semantic search tools"


def test_catalog_structure(catalog):
    """Test that catalog has expected structure."""
    assert 'tools' in catalog, "Catalog should have 'tools' key"
    assert isinstance(catalog['tools'], list), "Tools should be a list"
    assert len(catalog['tools']) > 0, "Should have at least one tool"

    first_tool = catalog['tools'][0]
    required_fields = ['name', 'import', 'description', 'search_keywords']

    for field in required_fields:
        assert field in first_tool, f"Tool should have '{field}' field"


def test_all_tools_have_required_fields(catalog):
    """Test that all tools have minimum required fields."""
    required_fields = ['name', 'import', 'description']

    for tool in catalog['tools']:
        for field in required_fields:
            assert field in tool, f"Tool '{tool.get('name', 'unknown')}' missing '{field}'"
        assert tool['name'], "Tool name should not be empty"
        assert tool['import'], "Tool import should not be empty"
