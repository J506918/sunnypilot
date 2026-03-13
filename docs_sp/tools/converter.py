"""MkDocs Material -> Discourse-compatible Markdown converter.

Converts MkDocs Material syntax to Discourse-friendly markdown:
1. Strip YAML front matter
2. Convert admonitions (!!!/???/???+) to Discourse callouts (> [!TYPE])
3. Convert Material tabs (=== "Tab Name") to bold headings + ---
4. Strip grid card HTML (<div class="grid cards" markdown>)
5. Convert Material emoji shortcodes (:material-*:) to Unicode or strip
6. Resolve internal .md links to docs site URLs
7. Clean up excessive blank lines
"""

from __future__ import annotations

import os
import re
from pathlib import Path

DOCS_BASE_URL = os.environ.get("DOCS_BASE_URL", "https://docs.sunnypilot.ai")

ADMONITION_MAP: dict[str, str] = {
    "note": "NOTE",
    "abstract": "ABSTRACT",
    "info": "INFO",
    "tip": "TIP",
    "success": "SUCCESS",
    "question": "QUESTION",
    "warning": "WARNING",
    "failure": "FAILURE",
    "danger": "DANGER",
    "bug": "BUG",
    "example": "EXAMPLE",
    "quote": "QUOTE",
}

EMOJI_MAP: dict[str, str] = {
    ":material-rocket-launch:": "",
    ":material-cog:": "",
    ":material-car:": "",
    ":material-shield:": "",
    ":material-download:": "",
    ":material-check:": "Y",
    ":material-close:": "N",
    ":material-alert:": "!",
    ":material-information:": "i",
    ":material-help-circle:": "?",
    ":material-star:": "*",
    ":material-link:": "",
    ":material-eye:": "",
    ":material-map:": "",
    ":material-wifi:": "",
    ":material-cellphone:": "",
    ":material-steering:": "",
    ":material-speedometer:": "",
}

_ADMONITION_RE = re.compile(r"^(\s*)(!{3}|\?{3}\+?)\s+(\w+)(?:\s+\"([^\"]*)\")?\s*$")
_TAB_RE = re.compile(r'^(\s*)===\s+"([^"]+)"\s*$')
_FRONT_MATTER_RE = re.compile(r"\A---\n.*?\n---\n*", re.DOTALL)
_GRID_CARD_OPEN_RE = re.compile(r'<div\s+class="grid\s+cards"\s*(?:markdown)?\s*>')
_GRID_CARD_CLOSE_RE = re.compile(r"</div>")
_INTERNAL_LINK_RE = re.compile(r"\]\(([^)]+\.md(?:#[^)]*)?)\)")
_EMOJI_SHORTCODE_RE = re.compile(r":material-[\w-]+:")
_EXCESSIVE_BLANKS_RE = re.compile(r"\n{4,}")


def convert(content: str, *, file_path: str, docs_base_url: str | None = None) -> str:
    """Convert MkDocs Material markdown to Discourse-compatible markdown.

    Args:
        content: Raw markdown content.
        file_path: Path to the source file (relative to docs_sp/ or absolute).
            Used for resolving relative internal links.
        docs_base_url: Base URL for the docs site. Defaults to DOCS_BASE_URL env var.

    Returns:
        Converted markdown string.
    """
    base_url = docs_base_url or DOCS_BASE_URL
    result = content

    result = strip_front_matter(result)
    result = convert_admonitions(result)
    result = convert_tabs(result)
    result = convert_grid_cards(result)
    result = convert_emoji_shortcodes(result)
    result = resolve_internal_links(result, file_path=file_path, docs_base_url=base_url)
    result = clean_blank_lines(result)

    return result.strip() + "\n"


def strip_front_matter(content: str) -> str:
    """Remove YAML front matter (--- ... ---) from the start of content."""
    return _FRONT_MATTER_RE.sub("", content)


def convert_admonitions(content: str) -> str:
    """Convert MkDocs admonitions to Discourse/Obsidian callouts.

    Input:
        !!! warning "Title"
            Content line 1
            Content line 2

    Output:
        > [!WARNING] Title
        > Content line 1
        > Content line 2
    """
    lines = content.splitlines(keepends=True)
    result: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        m = _ADMONITION_RE.match(line)

        if m:
            indent = m.group(1)
            marker = m.group(2)
            ad_type = m.group(3).lower()
            title = m.group(4)

            callout_type = ADMONITION_MAP.get(ad_type, ad_type.upper())

            header = f"{indent}> [!{callout_type}]"
            if title:
                header += f" {title}"

            if marker.startswith("???"):
                collapsed = "+" not in marker
                if not title:
                    action = "expand" if collapsed else "collapse"
                    header += f" *(click to {action})*"

            result.append(header + "\n")
            i += 1

            content_indent = indent + "    "
            while i < len(lines):
                content_line = lines[i]
                if content_line.startswith(content_indent):
                    stripped = content_line[len(content_indent):]
                    result.append(f"{indent}> {stripped}")
                    i += 1
                elif content_line.strip() == "":
                    # Check if blank line is internal to the admonition
                    j = i + 1
                    while j < len(lines) and lines[j].strip() == "":
                        j += 1
                    if j < len(lines) and lines[j].startswith(content_indent):
                        result.append(f"{indent}>\n")
                        i += 1
                    else:
                        break
                else:
                    break
        else:
            result.append(line)
            i += 1

    return "".join(result)


def convert_tabs(content: str) -> str:
    """Convert Material tabs to bold headings with horizontal rules.

    Input:
        === "Tab Name"
            Content

    Output:
        **Tab Name**

        Content

        ---
    """
    lines = content.splitlines(keepends=True)
    result: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        m = _TAB_RE.match(line)

        if m:
            indent = m.group(1)
            tab_name = m.group(2)

            result.append(f"{indent}**{tab_name}**\n")
            result.append("\n")
            i += 1

            content_indent = indent + "    "
            while i < len(lines):
                content_line = lines[i]
                if content_line.startswith(content_indent) or content_line.strip() == "":
                    if content_line.strip() == "":
                        result.append("\n")
                    else:
                        stripped = content_line[len(content_indent):]
                        result.append(f"{indent}{stripped}")
                    i += 1
                else:
                    break

            # Ensure blank line before the horizontal rule
            if result and result[-1] != "\n":
                result.append("\n")
            result.append(f"{indent}---\n")
            result.append("\n")
        else:
            result.append(line)
            i += 1

    return "".join(result)


def convert_grid_cards(content: str) -> str:
    """Strip grid card HTML wrappers (Discourse doesn't support them)."""
    result = _GRID_CARD_OPEN_RE.sub("", content)
    result = _GRID_CARD_CLOSE_RE.sub("", result)
    return result


def convert_emoji_shortcodes(content: str) -> str:
    """Convert Material emoji shortcodes to plain text equivalents."""
    result = content
    for shortcode, replacement in EMOJI_MAP.items():
        result = result.replace(shortcode, replacement)
    # Strip any remaining :material-*: shortcodes not in the map
    result = _EMOJI_SHORTCODE_RE.sub("", result)
    return result


def resolve_internal_links(
    content: str, *, file_path: str, docs_base_url: str
) -> str:
    """Resolve internal .md links to docs site URLs.

    Converts: [text](../features/icbm.md) -> [text](https://docs.sunnypilot.ai/features/icbm/)
    Converts: [text](../features/icbm.md#section) -> [text](https://docs.sunnypilot.ai/features/icbm/#section)
    """
    current_dir = str(Path(file_path).parent)

    def _replace_link(match: re.Match[str]) -> str:
        raw_path = match.group(1)
        # Skip external URLs
        if raw_path.startswith("http"):
            return match.group(0)

        # Split path and optional anchor
        anchor = ""
        if "#" in raw_path:
            raw_path, anchor = raw_path.rsplit("#", 1)
            anchor = f"#{anchor}"

        # Resolve relative path from current file's directory
        resolved = Path(current_dir, raw_path).resolve()
        resolved_str = str(resolved)

        # Extract docs-relative path
        docs_sp_idx = resolved_str.find("docs_sp/")
        if docs_sp_idx >= 0:
            docs_relative = resolved_str[docs_sp_idx + len("docs_sp/"):]
        else:
            # Fallback: use the resolved path relative to file
            docs_relative = raw_path

        # Convert to URL: remove .md, add trailing slash
        url_path = re.sub(r"\.md$", "/", docs_relative)
        return f"]({docs_base_url}/{url_path}{anchor})"

    return _INTERNAL_LINK_RE.sub(_replace_link, content)


def clean_blank_lines(content: str) -> str:
    """Collapse 4+ consecutive blank lines down to 3."""
    return _EXCESSIVE_BLANKS_RE.sub("\n\n\n", content)
