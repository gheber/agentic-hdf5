"""Tests for the @hdf5_tool decorator and registry."""

import sys
from pathlib import Path

# Ensure tools/h5py bare imports resolve
_h5py_dir = str(Path(__file__).resolve().parent.parent / "tools" / "h5py")
if _h5py_dir not in sys.path:
    sys.path.insert(0, _h5py_dir)

from tools.h5py.registry import TOOL_REGISTRY


EXPECTED_TOOL_NAMES = [
    "get_object_metadata",
    "rechunk_dataset",
    "apply_filter_dataset",
    "visualize",
    "read_semantic_metadata",
    "write_semantic_metadata",
    "collect_objects_for_smd",
    "write_smd_batch",
    "vectorize_semantic_metadata",
    "query_semantic_metadata",
    "check_cf_compliance",
]


def _ensure_registry_populated():
    """Import all tool modules to populate the registry if not already done."""
    if len(TOOL_REGISTRY) < 10:
        import tools.h5py  # noqa: F401 — triggers __init__ imports


def test_all_tools_registered():
    _ensure_registry_populated()
    registered_names = [f._hdf5_tool_meta["name"] for f in TOOL_REGISTRY]
    for name in EXPECTED_TOOL_NAMES:
        assert name in registered_names, f"{name} not found in registry"


def test_registry_count():
    _ensure_registry_populated()
    assert len(TOOL_REGISTRY) >= 11


def test_decorator_preserves_function():
    """Decorator should not alter the function itself (still callable, same name)."""
    _ensure_registry_populated()
    for func in TOOL_REGISTRY:
        meta = func._hdf5_tool_meta
        assert func.__name__ == meta["name"]
        assert callable(func)


def test_metadata_fields():
    """Each registered tool should have required metadata fields."""
    _ensure_registry_populated()
    for func in TOOL_REGISTRY:
        meta = func._hdf5_tool_meta
        assert "name" in meta
        assert "category" in meta
        assert "keywords" in meta
        assert "use_cases" in meta
        assert isinstance(meta["keywords"], list)
        assert isinstance(meta["use_cases"], list)
        assert len(meta["keywords"]) > 0
        assert len(meta["use_cases"]) > 0


def test_catalog_generation_schema():
    """generate_catalog should produce valid catalog structure."""
    from tools.generate_catalog import generate_catalog

    catalog = generate_catalog()
    assert "_generated" in catalog
    assert "version" in catalog
    assert "tools" in catalog
    assert len(catalog["tools"]) == 11

    for tool in catalog["tools"]:
        assert "name" in tool
        assert "import" in tool
        assert "description" in tool
        assert "search_keywords" in tool
        assert "category" in tool
        assert "use_cases" in tool


class TestExtractDetailedDescription:
    """Tests for _extract_detailed_description in generate_catalog."""

    def test_no_docstring(self):
        from tools.generate_catalog import _extract_detailed_description
        def func():
            pass
        assert _extract_detailed_description(func) == ""

    def test_single_line_docstring(self):
        from tools.generate_catalog import _extract_detailed_description
        def func():
            """Just a summary."""
        assert _extract_detailed_description(func) == ""

    def test_multiline_docstring(self):
        from tools.generate_catalog import _extract_detailed_description
        def func():
            """Summary line.

            This is the body of the docstring.
            It has multiple lines.

            Args:
                x: not included
            """
        result = _extract_detailed_description(func)
        assert "body of the docstring" in result
        assert "multiple lines" in result
        assert "Args" not in result

    def test_docstring_stops_at_returns(self):
        from tools.generate_catalog import _extract_detailed_description
        def func():
            """Summary.

            Body text here.

            Returns:
                Something
            """
        result = _extract_detailed_description(func)
        assert "Body text here" in result
        assert "Returns" not in result

    def test_docstring_stops_at_parameters(self):
        from tools.generate_catalog import _extract_detailed_description
        def func():
            """Summary.

            Body text.

            Parameters:
                x: something
            """
        result = _extract_detailed_description(func)
        assert "Body text" in result
        assert "Parameters" not in result
