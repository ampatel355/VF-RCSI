"""Compatibility wrapper that forwards legacy CLI usage to Code/main.py."""

from __future__ import annotations

try:
    from main import main
except ModuleNotFoundError:
    from Code.main import main


if __name__ == "__main__":
    main()
