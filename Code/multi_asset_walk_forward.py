"""Load the preserved compiled walk-forward implementation."""

from __future__ import annotations

import importlib.util
from importlib.machinery import SourcelessFileLoader
import os
from pathlib import Path
import subprocess
import sys


def _normalize_python_script_command(command, code_dir: Path) -> tuple[object, bool]:
    """Resolve relative Python script paths against the Code directory."""
    if not isinstance(command, (list, tuple)) or len(command) < 2:
        return command, False

    script_name = command[1]
    if not isinstance(script_name, str):
        return command, False

    script_path = Path(script_name)
    if script_path.is_absolute() or script_path.suffix.lower() != ".py":
        return command, False

    candidate_path = code_dir / script_path
    if not candidate_path.exists():
        return command, False

    normalized_command = list(command)
    normalized_command[1] = str(candidate_path)
    return normalized_command, True


def _patch_subprocess_run(code_dir: Path) -> None:
    """Make the preserved implementation launch sibling scripts from Code/."""
    original_run = subprocess.run

    def run(*popenargs, **kwargs):
        if popenargs:
            normalized_command, rewritten = _normalize_python_script_command(popenargs[0], code_dir)
            if rewritten:
                kwargs.setdefault("cwd", str(code_dir))
                popenargs = (normalized_command, *popenargs[1:])
            return original_run(*popenargs, **kwargs)

        if "args" in kwargs:
            normalized_command, rewritten = _normalize_python_script_command(kwargs["args"], code_dir)
            if rewritten:
                kwargs["args"] = normalized_command
                kwargs.setdefault("cwd", str(code_dir))

        return original_run(**kwargs)

    subprocess.run = run


def _configured_min_trades(module) -> int | None:
    """Return an optional panel trade-count override for the preserved workflow."""
    override_text = os.environ.get("WALK_FORWARD_MIN_TRADES_PER_PANEL", "").strip()
    if override_text:
        return max(int(override_text), 1)

    tickers = getattr(module, "TICKERS", [])
    if isinstance(tickers, str):
        tickers = [ticker.strip() for ticker in tickers.split(",") if ticker.strip()]

    if any(str(ticker).strip().upper().endswith("=X") for ticker in tickers):
        # Forex panels can be much sparser because Yahoo does not expose exchange volume
        # and the preserved walk-forward windows are strict about completed trades.
        return 1

    return None


def _configure_impl(module) -> None:
    """Adjust preserved implementation settings for the current runtime."""
    min_trades = _configured_min_trades(module)
    if min_trades is not None and hasattr(module, "MIN_TRADES_PER_PANEL"):
        module.MIN_TRADES_PER_PANEL = min_trades

    original_load_fold_market_data = getattr(module, "load_fold_market_data", None)
    if callable(original_load_fold_market_data):
        def load_fold_market_data_with_ticker(ticker: str):
            market_df = original_load_fold_market_data(ticker)
            market_df.attrs["ticker"] = str(ticker).strip().upper()
            return market_df

        module.load_fold_market_data = load_fold_market_data_with_ticker


def _load_compiled_module():
    """Load the legacy walk-forward implementation from its tracked bytecode file."""
    code_dir = Path(__file__).resolve().parent
    pyc_path = (
        code_dir
        / "__pycache__"
        / "multi_asset_walk_forward_impl.cpython-314.pyc"
    )
    if not pyc_path.exists():
        raise FileNotFoundError(
            "The compiled walk-forward implementation is missing from Code/__pycache__."
        )

    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    _patch_subprocess_run(code_dir)

    loader = SourcelessFileLoader("_multi_asset_walk_forward_impl", str(pyc_path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise ImportError(f"Unable to build an import spec for {pyc_path}.")

    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    _configure_impl(module)
    return module


_IMPL = _load_compiled_module()
__doc__ = _IMPL.__doc__

for _name in dir(_IMPL):
    if _name.startswith("__") and _name not in {"__all__"}:
        continue
    globals()[_name] = getattr(_IMPL, _name)


if __name__ == "__main__":
    _IMPL.main()
