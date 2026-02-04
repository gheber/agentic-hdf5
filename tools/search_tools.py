#!/usr/bin/env python3
"""
Command-line tool for searching the HDF5 agentic tools catalog.

Usage:
    python tools/search_tools.py "your search query"
    python -m tools.search_tools "chunk optimization"
"""

import sys
import json
import os
from pathlib import Path

# Maximum number of results to return
MAX_RESULTS = 5


def load_catalog(catalog_path=None):
    """
    Load the tool catalog JSON file.

    Args:
        catalog_path: Path to catalog file (if None, searches in standard locations)

    Returns:
        Dictionary containing the catalog data

    Raises:
        FileNotFoundError: If catalog file cannot be found
    """
    if catalog_path is None:
        # Try to find catalog in standard locations
        possible_paths = [
            Path(__file__).parent / "tool_catalog.json",
            Path.cwd() / "tools" / "tool_catalog.json",
            Path.cwd() / "tool_catalog.json",
        ]

        for path in possible_paths:
            if path.exists():
                catalog_path = path
                break

        if catalog_path is None:
            raise FileNotFoundError(
                "Could not find tool_catalog.json in standard locations. "
                "Please specify path explicitly."
            )

    with open(catalog_path, 'r') as f:
        return json.load(f)


def search_tools(query, catalog=None, max_results=None):
    """
    Search for tools matching the given query.

    Searches across tool names, descriptions, and keywords using case-insensitive
    substring matching. Returns tools ranked by relevance (number of matches).

    Args:
        query: Search query string
        catalog: Pre-loaded catalog dict (if None, loads from default location)
        max_results: Maximum number of results (defaults to MAX_RESULTS global)

    Returns:
        List of matching tool dictionaries, sorted by relevance

    Raises:
        ValueError: If query is empty or invalid
    """
    # Validate query
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string")

    query = query.strip()
    if not query:
        raise ValueError("Query must be a non-empty string")

    # Load catalog if not provided
    if catalog is None:
        catalog = load_catalog()

    if max_results is None:
        max_results = MAX_RESULTS

    # Normalize query for case-insensitive search
    query_lower = query.lower()
    query_terms = query_lower.split()

    results = []

    for tool in catalog.get('tools', []):
        score = 0

        # Search in tool name
        tool_name = tool.get('name', '').lower()
        if query_lower in tool_name:
            score += 10  # Exact match in name is high priority
        else:
            # Check individual terms
            for term in query_terms:
                if term in tool_name:
                    score += 5

        # Search in description
        description = tool.get('description', '').lower()
        if query_lower in description:
            score += 5
        else:
            for term in query_terms:
                if term in description:
                    score += 2

        # Search in detailed description
        detailed_desc = tool.get('detailed_description', '').lower()
        for term in query_terms:
            if term in detailed_desc:
                score += 1

        # Search in keywords (highest priority for exact matches)
        keywords = tool.get('search_keywords', [])
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if query_lower == keyword_lower:
                score += 20  # Exact keyword match
            elif query_lower in keyword_lower:
                score += 8
            else:
                for term in query_terms:
                    if term in keyword_lower:
                        score += 3

        # Search in use cases
        use_cases = tool.get('use_cases', [])
        for use_case in use_cases:
            use_case_lower = use_case.lower()
            for term in query_terms:
                if term in use_case_lower:
                    score += 1

        # If tool has any relevance, add it to results
        if score > 0:
            results.append({
                'tool': tool,
                'score': score
            })

    # Sort by score (descending)
    results.sort(key=lambda x: x['score'], reverse=True)

    # Return top max_results
    return [r['tool'] for r in results[:max_results]]


def format_tool_output(tool, index=None):
    """
    Format a tool for terminal output.

    Args:
        tool: Tool dictionary
        index: Optional index number (1-based) for numbered lists

    Returns:
        Formatted string
    """
    lines = []

    # Add number if provided
    if index is not None:
        lines.append(f"\n{index}. {tool['name']}")
    else:
        lines.append(f"\n{tool['name']}")

    lines.append("=" * len(tool['name']))
    lines.append(f"Description: {tool.get('description', 'N/A')}")
    lines.append(f"Category: {tool.get('category', 'N/A')}")
    lines.append(f"Import: {tool.get('import', 'N/A')}")

    # Add keywords
    keywords = tool.get('search_keywords', [])
    if keywords:
        lines.append(f"Keywords: {', '.join(keywords[:8])}")  # Show first 8 keywords

    return '\n'.join(lines)


def main():
    """Main CLI entry point."""
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python tools/search_tools.py <query>", file=sys.stderr)
        print("Example: python tools/search_tools.py 'chunk optimization'", file=sys.stderr)
        sys.exit(1)

    # Get query from command line
    query = ' '.join(sys.argv[1:])

    try:
        # Search for tools
        results = search_tools(query)

        # Display results
        if not results:
            print(f"No tools found matching query: '{query}'")
            sys.exit(0)

        print(f"Found {len(results)} tool(s) matching '{query}':")

        for i, tool in enumerate(results, start=1):
            print(format_tool_output(tool, index=i))

        sys.exit(0)

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
