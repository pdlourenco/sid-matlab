#!/usr/bin/env python3
"""Validate Python docstrings against the CONTRIBUTING.md standard.

Checks library modules in python/sid/ and python/sid/_internal/.
Skips __init__.py, _results.py, _exceptions.py, and test files.

Exit codes:
    0 — all checks pass
    1 — one or more hard errors found
"""

import ast
import glob
import os
import re
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Directories to scan (relative to repo root)
SCAN_DIRS = ["python/sid", "python/sid/_internal"]

# Files to skip (no docstring enforcement)
SKIP_FILES = {"__init__.py", "_results.py", "_exceptions.py", "conftest.py"}

# Files exempt from specific sections
SPEC_EXEMPT: set[str] = set()
PARAMS_EXEMPT: set[str] = set()
RETURNS_EXEMPT: set[str] = set()

# Canonical section order — entries that may appear in the docstring.
# Each tuple: (section header regex, label, is_required)
# "required" means the section must appear in every function docstring
# (subject to per-file exemptions above).
SECTION_DEFS = [
    (r"^Parameters\s*$",  "Parameters",  True),
    (r"^Returns\s*$",     "Returns",     True),
    (r"^Raises\s*$",      "Raises",      False),
    (r"^Examples\s*$",    "Examples",    True),
    (r"^Notes\s*$",       "Notes",       False),
    (r"^References\s*$",  "References",  False),
    (r"^See Also\s*$",    "See Also",    True),
    (r"^Changelog\s*$",   "Changelog",   True),
]

# Singular headings that violate the plural rule
SINGULAR_PATTERNS = [
    (r"^Parameter\s*$",  "Parameter",  "Parameters"),
    (r"^Return\s*$",     "Return",     "Returns"),
    (r"^Example\s*$",    "Example",    "Examples"),
    (r"^Reference\s*$",  "Reference",  "References"),
]

# Copyright marker — must appear in file-level comments before module docstring
COPYRIGHT_PATTERN = re.compile(r"Copyright.*Pedro Lourenco", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_python_files(repo_root: str) -> list[str]:
    """Return sorted list of .py files in SCAN_DIRS (skip exempted files)."""
    files = []
    for d in SCAN_DIRS:
        pattern = os.path.join(repo_root, d, "*.py")
        for f in glob.glob(pattern):
            basename = os.path.basename(f)
            if basename not in SKIP_FILES:
                files.append(f)
    return sorted(files)


def get_file_comments(filepath: str) -> list[str]:
    """Return leading comment lines from a Python file (before code)."""
    comments = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#"):
                comments.append(stripped)
            elif stripped == "" or stripped.startswith('"""') or stripped.startswith("'"):
                # Allow blank lines and docstrings after comments
                if stripped == "":
                    continue
                break
            else:
                break
    return comments


def extract_public_functions(filepath: str) -> list[tuple[str, int, str | None]]:
    """Return list of (func_name, lineno, docstring) for public functions.

    Only considers module-level functions that don't start with '_'.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError:
        return []

    functions = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            docstring = ast.get_docstring(node)
            functions.append((node.name, node.lineno, docstring))

    return functions


def parse_docstring_sections(docstring: str) -> dict[str, int]:
    """Parse NumPy-style docstring and return {section_label: line_offset}.

    NumPy sections are identified by a header line followed by a line of
    dashes (e.g., "Parameters\\n----------").
    """
    if not docstring:
        return {}

    lines = docstring.split("\n")
    sections: dict[str, int] = {}

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Check if the next line is a line of dashes (NumPy section marker)
        if i + 1 < len(lines):
            next_stripped = lines[i + 1].strip()
            if next_stripped and all(c == "-" for c in next_stripped) and len(next_stripped) >= 3:
                # This line is a section header
                for pattern, label, _ in SECTION_DEFS:
                    if re.match(pattern, stripped):
                        if label not in sections:
                            sections[label] = i
                        break
                # Also check singular patterns
                for pattern, singular, plural in SINGULAR_PATTERNS:
                    if re.match(pattern, stripped):
                        if f"SINGULAR:{singular}" not in sections:
                            sections[f"SINGULAR:{singular}"] = i
                        break

    return sections


def check_file(filepath: str, repo_root: str) -> tuple[list, list]:
    """Check a single file. Return (errors, warnings)."""
    errors: list[tuple[int, str]] = []   # (line, message)
    warnings: list[tuple[int, str]] = []
    basename = os.path.basename(filepath)

    # --- Rule 0: Copyright comment at top of file ---
    file_comments = get_file_comments(filepath)
    has_copyright = any(COPYRIGHT_PATTERN.search(c) for c in file_comments)
    if not has_copyright:
        errors.append((1, "Missing copyright comment at top of file"))

    # --- Check each public function ---
    functions = extract_public_functions(filepath)

    if not functions:
        # File has no public functions — might be a module with only private
        # helpers. Skip docstring checks but copyright still required.
        return errors, warnings

    for func_name, func_lineno, docstring in functions:
        prefix = f"{func_name}(): "

        if docstring is None:
            errors.append((func_lineno, f"{prefix}Missing docstring"))
            continue

        # Parse sections
        sections = parse_docstring_sections(docstring)

        # --- Rule 1: No singular headings ---
        for pattern, singular, plural in SINGULAR_PATTERNS:
            key = f"SINGULAR:{singular}"
            if key in sections:
                offset = sections[key]
                errors.append((
                    func_lineno + offset,
                    f"{prefix}Singular heading '{singular}' -- use '{plural}'"
                ))

        # --- Rule 2: Required sections present ---
        for _, label, required in SECTION_DEFS:
            if required and label not in sections:
                # Per-file exemptions
                if label == "Parameters" and basename in PARAMS_EXEMPT:
                    continue
                if label == "Returns" and basename in RETURNS_EXEMPT:
                    continue
                errors.append((func_lineno, f"{prefix}Missing required section: {label}"))

        # --- Rule 3: Exactly one See Also ---
        see_also_count = sum(
            1 for line in docstring.split("\n")
            if re.match(r"^\s*See Also\s*$", line.strip())
        )
        if see_also_count > 1:
            errors.append((
                func_lineno,
                f"{prefix}Duplicate 'See Also' -- found {see_also_count}, expected 1"
            ))

        # --- Rule 4: Section order ---
        canonical_order = [label for _, label, _ in SECTION_DEFS]
        found_order = []
        for label in canonical_order:
            if label in sections:
                found_order.append((sections[label], label))
        found_order.sort(key=lambda x: x[0])

        for i in range(len(found_order) - 1):
            offset_a, label_a = found_order[i]
            offset_b, label_b = found_order[i + 1]
            idx_a = canonical_order.index(label_a)
            idx_b = canonical_order.index(label_b)
            if idx_a > idx_b:
                errors.append((
                    func_lineno + offset_b,
                    f"{prefix}Section order violation: '{label_b}' "
                    f"appears before '{label_a}'"
                ))

        # --- Rule 5 (warning): Missing SPEC.md reference in Notes ---
        if basename not in SPEC_EXEMPT:
            has_spec_ref = "SPEC.md" in docstring or "Specification" in docstring
            if not has_spec_ref:
                warnings.append((
                    func_lineno,
                    f"{prefix}No SPEC.md reference found in docstring"
                ))

    return errors, warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    # Determine repo root (script lives in .github/scripts/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.normpath(os.path.join(script_dir, "..", ".."))

    files = find_python_files(repo_root)
    if not files:
        print("No .py files found -- nothing to check.")
        return 0

    total_errors = 0
    total_warnings = 0

    for filepath in files:
        relpath = os.path.relpath(filepath, repo_root)
        file_errors, file_warnings = check_file(filepath, repo_root)

        for lineno, msg in file_errors:
            print(f"::error file={relpath},line={lineno}::{msg}")
            total_errors += 1
        for lineno, msg in file_warnings:
            print(f"::warning file={relpath},line={lineno}::{msg}")
            total_warnings += 1

    print()
    print(
        f"Python header check: {len(files)} file(s) analysed, "
        f"{total_errors} error(s), {total_warnings} warning(s)"
    )

    return 1 if total_errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
