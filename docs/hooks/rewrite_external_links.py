"""Rewrite broken relative `*.md` links in included content to GitHub URLs.

The `include-markdown` plugin pulls in README / SPEC / CONTRIBUTING etc.
from outside `docs/`. Those source files contain relative markdown links
that resolve against the SOURCE file's location, not the docs page they
get embedded in — so once included, the links no longer resolve.

This hook knows which docs pages are "includes" of which repo files, and
rewrites any relative `*.md` link on those pages to a `github.com/.../blob/main/...`
URL pointing at the equivalent repo path. Links that resolve cleanly to a
file inside `docs/` are left alone; HTTP(S), mailto, and anchor-only
links are left alone.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GITHUB_PREFIX = "https://github.com/pdlourenco/sid/blob/main"

# Docs page → original repo file it was included from.
INCLUDE_SOURCES: dict[str, str] = {
    "about/changelog.md": "python/RELEASE_NOTES.md",
    "about/contributing.md": "CONTRIBUTING.md",
    "spec/index.md": "spec/SPEC.md",
    "spec/examples-spec.md": "spec/EXAMPLES.md",
    "spec/cosmic/automatic-tuning.md": "spec/cosmic/automatic_tuning.md",
    "spec/cosmic/online-recursion.md": "spec/cosmic/online_recursion.md",
    "spec/cosmic/output.md": "spec/cosmic/output.md",
    "spec/cosmic/uncertainty-derivation.md": "spec/cosmic/uncertainty_derivation.md",
}

LINK_RE = re.compile(r"(\[[^\]]*\])\(([^)\s]+)\)")


def on_page_markdown(markdown, page, config, files):
    src_path = page.file.src_path.replace("\\", "/")
    source = INCLUDE_SOURCES.get(src_path)
    if source is None:
        return markdown
    source_dir = Path(source).parent

    def replace(match: re.Match[str]) -> str:
        label, target = match.group(1), match.group(2)
        if target.startswith(("http://", "https://", "mailto:", "#")):
            return match.group(0)
        anchor = ""
        if "#" in target:
            target, anchor = target.split("#", 1)
            anchor = "#" + anchor
        if not target:
            return match.group(0)
        try:
            repo_path = (REPO_ROOT / source_dir / target).resolve()
            rel_to_repo = repo_path.relative_to(REPO_ROOT)
        except (OSError, ValueError):
            return match.group(0)
        if not repo_path.exists():
            return match.group(0)
        return f"{label}({GITHUB_PREFIX}/{rel_to_repo}{anchor})"

    return LINK_RE.sub(replace, markdown)
