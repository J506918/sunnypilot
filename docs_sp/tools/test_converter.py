#!/usr/bin/env python3
"""Test the MkDocs → Discourse admonition conversion logic.

This mirrors the Ruby converter's logic to verify correctness.
Run: python3 docs_sp/tools/test_converter.py
"""

import re
import sys

ADMONITION_MAP = {
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


def convert_admonitions(content: str) -> str:
    """Convert MkDocs admonitions to Obsidian/Discourse callouts."""
    lines = content.splitlines(keepends=True)
    result = []
    i = 0

    pattern = re.compile(r'^(\s*)(!{3}|\?{3}\+?) (\w+)(?: "([^"]*)")?')

    while i < len(lines):
        line = lines[i]
        m = pattern.match(line)

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
                    header += f" *(click to {'expand' if collapsed else 'collapse'})*"

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
                    # Blank line: only treat as part of admonition if the
                    # next non-blank line is still indented at content level
                    j = i + 1
                    while j < len(lines) and lines[j].strip() == "":
                        j += 1
                    if j < len(lines) and lines[j].startswith(content_indent):
                        result.append(f"{indent}>\n")
                        i += 1
                    else:
                        # Blank line ends the admonition
                        break
                else:
                    break
        else:
            result.append(line)
            i += 1

    return "".join(result)


# ---- Tests ----

def test_basic_warning():
    input_text = '''!!! warning "Important"
    sunnypilot is a **driver assistance** system.
    Always pay attention.
'''
    expected = '''> [!WARNING] Important
> sunnypilot is a **driver assistance** system.
> Always pay attention.
'''
    result = convert_admonitions(input_text)
    assert result == expected, f"FAIL basic_warning:\n{result!r}\n!=\n{expected!r}"
    print("  PASS: basic_warning")


def test_info_no_title():
    input_text = '''!!! info
    Content line 1
    Content line 2
'''
    expected = '''> [!INFO]
> Content line 1
> Content line 2
'''
    result = convert_admonitions(input_text)
    assert result == expected, f"FAIL info_no_title:\n{result!r}\n!=\n{expected!r}"
    print("  PASS: info_no_title")


def test_info_with_title():
    input_text = '''!!! info "Requirements"
    - Longitudinal control must be available
    - ICBM must be enabled
'''
    expected = '''> [!INFO] Requirements
> - Longitudinal control must be available
> - ICBM must be enabled
'''
    result = convert_admonitions(input_text)
    assert result == expected, f"FAIL info_with_title:\n{result!r}\n!=\n{expected!r}"
    print("  PASS: info_with_title")


def test_danger():
    input_text = '''!!! danger "Important"
    sunnypilot is a **driver assistance** system. It is **NOT** a self-driving system.
'''
    expected = '''> [!DANGER] Important
> sunnypilot is a **driver assistance** system. It is **NOT** a self-driving system.
'''
    result = convert_admonitions(input_text)
    assert result == expected, f"FAIL danger:\n{result!r}\n!=\n{expected!r}"
    print("  PASS: danger")


def test_tip():
    input_text = '''!!! tip
    The more detail you provide, the faster we can diagnose and fix the issue.
'''
    expected = '''> [!TIP]
> The more detail you provide, the faster we can diagnose and fix the issue.
'''
    result = convert_admonitions(input_text)
    assert result == expected, f"FAIL tip:\n{result!r}\n!=\n{expected!r}"
    print("  PASS: tip")


def test_multiline_with_blank():
    input_text = '''!!! warning
    Line 1

    Line 2 after blank
'''
    expected = '''> [!WARNING]
> Line 1
>
> Line 2 after blank
'''
    result = convert_admonitions(input_text)
    assert result == expected, f"FAIL multiline_with_blank:\n{result!r}\n!=\n{expected!r}"
    print("  PASS: multiline_with_blank")


def test_collapsible():
    input_text = '''??? warning "Click to see"
    Hidden content
'''
    expected = '''> [!WARNING] Click to see
> Hidden content
'''
    result = convert_admonitions(input_text)
    assert result == expected, f"FAIL collapsible:\n{result!r}\n!=\n{expected!r}"
    print("  PASS: collapsible")


def test_collapsible_open():
    input_text = '''???+ info "Open by default"
    Visible content
'''
    expected = '''> [!INFO] Open by default
> Visible content
'''
    result = convert_admonitions(input_text)
    assert result == expected, f"FAIL collapsible_open:\n{result!r}\n!=\n{expected!r}"
    print("  PASS: collapsible_open")


def test_surrounded_by_content():
    input_text = '''Some text before.

!!! note "Note Title"
    Note content here.

Some text after.
'''
    expected = '''Some text before.

> [!NOTE] Note Title
> Note content here.

Some text after.
'''
    result = convert_admonitions(input_text)
    assert result == expected, f"FAIL surrounded:\n{result!r}\n!=\n{expected!r}"
    print("  PASS: surrounded_by_content")


def test_multiple_admonitions():
    input_text = '''!!! info "Requirements"
    - Req 1
    - Req 2

!!! warning "Vehicle Restrictions"
    - Tesla: disabled on release
    - Rivian: always disabled
'''
    expected = '''> [!INFO] Requirements
> - Req 1
> - Req 2

> [!WARNING] Vehicle Restrictions
> - Tesla: disabled on release
> - Rivian: always disabled
'''
    result = convert_admonitions(input_text)
    assert result == expected, f"FAIL multiple:\n{result!r}\n!=\n{expected!r}"
    print("  PASS: multiple_admonitions")


def test_real_doc_snippet():
    """Test with an actual snippet from docs_sp/settings/speed-limit.md."""
    input_text = '''## Speed Limit Mode

| Property | Value |
|----------|-------|
| **Param** | `SpeedLimitMode` |
| **Type** | Multi-button selector |

!!! info "Requirements"
    - Longitudinal control must be available, **or** ICBM must be enabled

!!! warning "Vehicle Restrictions"
    - **Tesla:** Speed Limit Assist mode is disabled on release branches
    - **Rivian:** Speed Limit Assist mode is always disabled

---
'''
    expected = '''## Speed Limit Mode

| Property | Value |
|----------|-------|
| **Param** | `SpeedLimitMode` |
| **Type** | Multi-button selector |

> [!INFO] Requirements
> - Longitudinal control must be available, **or** ICBM must be enabled

> [!WARNING] Vehicle Restrictions
> - **Tesla:** Speed Limit Assist mode is disabled on release branches
> - **Rivian:** Speed Limit Assist mode is always disabled

---
'''
    result = convert_admonitions(input_text)
    assert result == expected, f"FAIL real_doc:\n{result!r}\n!=\n{expected!r}"
    print("  PASS: real_doc_snippet")


if __name__ == "__main__":
    print("Testing MkDocs → Discourse admonition conversion:")
    tests = [
        test_basic_warning,
        test_info_no_title,
        test_info_with_title,
        test_danger,
        test_tip,
        test_multiline_with_blank,
        test_collapsible,
        test_collapsible_open,
        test_surrounded_by_content,
        test_multiple_admonitions,
        test_real_doc_snippet,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    sys.exit(1 if failed > 0 else 0)
