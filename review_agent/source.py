from __future__ import annotations


def source_lines(code: str) -> list[str]:
    """Split LF-normalized Python source without splitting string characters."""
    if not code:
        return []
    lines = code.split("\n")
    return lines[:-1] if code.endswith("\n") else lines
