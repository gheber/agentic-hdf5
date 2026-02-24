"""
Decorator-based registry for HDF5 tool functions.

Usage:
    from tools.h5py.registry import hdf5_tool, TOOL_REGISTRY

    @hdf5_tool(
        category="inspection",
        keywords=["metadata", "inspect"],
        use_cases=["Exploring file structure"],
    )
    def my_tool(filepath: str) -> dict:
        ...
"""

TOOL_REGISTRY: list = []


def hdf5_tool(category: str, keywords: list[str], use_cases: list[str]):
    """Decorator that attaches HDF5 tool metadata and registers the function."""

    def decorator(func):
        meta = {
            "name": func.__name__,
            "category": category,
            "keywords": keywords,
            "use_cases": use_cases,
        }
        func._hdf5_tool_meta = meta
        # Guard against double-registration (e.g. module re-imported under different path)
        existing_names = {f._hdf5_tool_meta["name"] for f in TOOL_REGISTRY}
        if meta["name"] not in existing_names:
            TOOL_REGISTRY.append(func)
        return func

    return decorator
