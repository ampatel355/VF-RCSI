"""Primary CLI entrypoint for the Virtu Fortuna research workflows."""

from __future__ import annotations

try:
    from aaamain import main
except ModuleNotFoundError:
    from Code.aaamain import main


if __name__ == "__main__":
    main()
