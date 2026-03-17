"""
Convert a Markdown file to PDF via pandoc + pdflatex.

Handles LaTeX math ($...$, $$...$$), tables, code blocks, and all
standard Markdown features natively through the pandoc+LaTeX pipeline.

Can be used as a library function or as a CLI::

    python -m tools.markdown_to_pdf input.md              # -> input.pdf
    python -m tools.markdown_to_pdf input.md -o out.pdf
    python -m tools.markdown_to_pdf input.md --engine xelatex
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


def markdown_to_pdf(
    input_path: str,
    output_path: Optional[str] = None,
    engine: str = "pdflatex",
    extra_args: Optional[list[str]] = None,
) -> dict:
    """Convert a Markdown file to PDF using pandoc + a LaTeX engine.

    Parameters
    ----------
    input_path : str
        Path to the source ``.md`` file.
    output_path : str, optional
        Where to write the PDF.  Defaults to the input path with a
        ``.pdf`` extension.
    engine : str
        LaTeX engine for pandoc (``pdflatex``, ``xelatex``, ``lualatex``).
    extra_args : list[str], optional
        Additional command-line arguments passed straight to pandoc
        (e.g. ``["--toc", "-V", "geometry:margin=1in"]``).

    Returns
    -------
    dict
        ``{"status": "ok", "output_path": "<abs path>"}`` on success,
        or ``{"status": "error", "message": "..."}`` on failure.
    """
    # ── Validate prerequisites ──────────────────────────────────────
    if shutil.which("pandoc") is None:
        return {
            "status": "error",
            "message": (
                "pandoc not found on PATH. Install it: "
                "apt install pandoc (Debian/Ubuntu), "
                "brew install pandoc (macOS), "
                "or conda install pandoc."
            ),
        }

    input_file = Path(input_path).resolve()
    if not input_file.exists():
        return {"status": "error", "message": f"Input file not found: {input_file}"}

    if output_path is None:
        output_file = input_file.with_suffix(".pdf")
    else:
        output_file = Path(output_path).resolve()

    # ── Auto-select engine if the requested one is missing ─────────
    # Prefer pdflatex (best LaTeX math), fall back to weasyprint (HTML).
    if shutil.which(engine) is None:
        if engine != "weasyprint" and shutil.which("weasyprint") is not None:
            engine = "weasyprint"
        else:
            return {
                "status": "error",
                "message": (
                    f"{engine} not found on PATH. Install a TeX distribution "
                    "(apt install texlive-latex-recommended) or weasyprint "
                    "(pip install weasyprint)."
                ),
            }

    is_html_engine = engine in ("weasyprint", "prince", "wkhtmltopdf")

    # ── Build pandoc command ────────────────────────────────────────
    cmd = [
        "pandoc",
        str(input_file),
        "-o", str(output_file),
        f"--pdf-engine={engine}",
        "--standalone",
    ]

    if is_html_engine:
        # For HTML-based engines: use --webtex to render LaTeX math
        # as images (weasyprint/prince can't execute MathJax JS).
        # Also apply CSS for margins/fonts.
        css_path = Path(__file__).resolve().parent / "markdown_to_pdf.css"
        cmd.append("--webtex")
        if css_path.exists():
            cmd.extend(["--css", str(css_path)])
    else:
        # For LaTeX engines: use geometry/fontsize variables.
        cmd.extend([
            "-V", "geometry:margin=1in",
            "-V", "fontsize=11pt",
        ])

    if extra_args:
        cmd.extend(extra_args)

    # ── Run pandoc ──────────────────────────────────────────────────
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(input_file.parent),
        )
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "pandoc timed out after 120 seconds"}
    except FileNotFoundError:
        return {"status": "error", "message": "pandoc executable not found"}

    if result.returncode != 0:
        stderr = result.stderr.strip()
        return {
            "status": "error",
            "message": f"pandoc exited with code {result.returncode}:\n{stderr}",
        }

    return {"status": "ok", "output_path": str(output_file)}


# ── CLI entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Markdown to PDF via pandoc + LaTeX"
    )
    parser.add_argument("input", help="Path to the .md file")
    parser.add_argument("-o", "--output", help="Output PDF path (default: same name as input)")
    parser.add_argument("--engine", default="pdflatex", help="LaTeX engine (default: pdflatex)")
    args, extra = parser.parse_known_args()

    result = markdown_to_pdf(args.input, args.output, args.engine, extra or None)
    if result["status"] == "error":
        print(f"ERROR: {result['message']}", file=sys.stderr)
        sys.exit(1)
    print(result["output_path"])
