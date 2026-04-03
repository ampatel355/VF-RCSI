"""Shared helpers for paths and optional figure output."""

from pathlib import Path
import os
import subprocess
import sys

try:
    from strategy_config import AGENT_ORDER
    from timeframe_config import timeframe_output_suffix
except ModuleNotFoundError:
    from Code.strategy_config import AGENT_ORDER
    from Code.timeframe_config import timeframe_output_suffix

# Every script can read the active ticker from the same environment variable.
ticker = os.environ.get("TICKER", "SPY")

# SHOW_PLOTS controls whether matplotlib figures are displayed interactively.
show_plots = os.environ.get("SHOW_PLOTS", "1") == "1"
save_outputs = os.environ.get("SAVE_OUTPUTS", "0") == "1"


def project_root() -> Path:
    """Return the root folder of the project."""
    return Path(__file__).resolve().parents[1]


def code_dir() -> Path:
    """Return the folder that contains the Python scripts."""
    return project_root() / "Code"


def resolve_named_dir(lowercase_name: str, uppercase_name: str) -> Path:
    """Return the canonical project folder, preferring the uppercase location.

    The repo standardized on ``Data_Clean``, ``Data_Raw``, and ``Charts``.
    Older lowercase variants may still exist from previous runs, but mixing
    them can make the app and the pipeline disagree about which artifacts are
    current.
    """
    root = project_root()
    lowercase_dir = root / lowercase_name
    uppercase_dir = root / uppercase_name

    if uppercase_dir.exists():
        return uppercase_dir
    if lowercase_dir.exists():
        return lowercase_dir

    uppercase_dir.mkdir(parents=True, exist_ok=True)
    return uppercase_dir


def data_raw_dir() -> Path:
    """Return the raw-data folder.

    Raw data is shared across timeframes (downloads are interval-stamped) so
    no timeframe suffix is applied here.
    """
    return resolve_named_dir("data_raw", "Data_Raw")


def data_clean_dir() -> Path:
    """Return the clean-data folder for the active timeframe.

    On daily data (the default) this returns ``Data_Clean`` for backward
    compatibility.  On other timeframes it returns e.g. ``Data_Clean_1h``
    so artifacts from different intervals never collide.
    """
    suffix = timeframe_output_suffix()
    return resolve_named_dir(f"data_clean{suffix}", f"Data_Clean{suffix}")


def charts_dir() -> Path:
    """Return the charts folder for the active timeframe."""
    suffix = timeframe_output_suffix()
    return resolve_named_dir(f"charts{suffix}", f"Charts{suffix}")


def raw_prices_path(current_ticker: str | None = None) -> Path:
    """Return the path to the raw price CSV for one ticker."""
    current_ticker = current_ticker or ticker
    return data_raw_dir() / f"{current_ticker}.csv"


def features_path(current_ticker: str | None = None) -> Path:
    """Return the path to the features CSV for one ticker."""
    current_ticker = current_ticker or ticker
    return data_clean_dir() / f"{current_ticker}_features.csv"


def regimes_path(current_ticker: str | None = None) -> Path:
    """Return the path to the regimes CSV for one ticker."""
    current_ticker = current_ticker or ticker
    return data_clean_dir() / f"{current_ticker}_regimes.csv"


def trades_path(agent_name: str, current_ticker: str | None = None) -> Path:
    """Return the path to the trades CSV for one agent and ticker."""
    current_ticker = current_ticker or ticker
    return data_clean_dir() / f"{current_ticker}_{agent_name}_trades.csv"


def metrics_path(agent_name: str, current_ticker: str | None = None) -> Path:
    """Return the path to the metrics CSV for one agent and ticker."""
    current_ticker = current_ticker or ticker
    return data_clean_dir() / f"{current_ticker}_{agent_name}_metrics.csv"


def regime_analysis_path(current_ticker: str | None = None) -> Path:
    """Return the path to the regime-analysis CSV."""
    current_ticker = current_ticker or ticker
    return data_clean_dir() / f"{current_ticker}_regime_analysis.csv"


def monte_carlo_summary_path(current_ticker: str | None = None) -> Path:
    """Return the path to the Monte Carlo summary CSV."""
    current_ticker = current_ticker or ticker
    return data_clean_dir() / f"{current_ticker}_monte_carlo_summary.csv"


def monte_carlo_returns_path(agent_name: str, current_ticker: str | None = None) -> Path:
    """Return the path to the detailed Monte Carlo results CSV for one agent."""
    current_ticker = current_ticker or ticker
    return data_clean_dir() / f"{current_ticker}_{agent_name}_monte_carlo_results.csv"


def rcsi_path(current_ticker: str | None = None) -> Path:
    """Return the path to the RCSI CSV."""
    current_ticker = current_ticker or ticker
    return data_clean_dir() / f"{current_ticker}_rcsi.csv"


def chart_path(filename: str) -> Path:
    """Return the generic chart path that existing scripts already use."""
    return charts_dir() / filename


def ticker_chart_path(filename: str, current_ticker: str | None = None) -> Path:
    """Return a ticker-labeled chart path for the same chart."""
    current_ticker = current_ticker or ticker
    return charts_dir() / f"{current_ticker}_{filename}"


def save_figure(fig, filename: str, current_ticker: str | None = None) -> tuple[Path, Path]:
    """Optionally save both the legacy chart name and ticker-labeled chart name.

    The simplified research workflow does not persist chart files unless
    SAVE_OUTPUTS=1 is explicitly enabled.
    """
    generic_path = chart_path(filename)
    labeled_path = ticker_chart_path(filename, current_ticker)

    if save_outputs:
        generic_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(generic_path, dpi=300, bbox_inches="tight")
        fig.savefig(labeled_path, dpi=300, bbox_inches="tight")

    return generic_path, labeled_path


def finish_figure(fig) -> None:
    """Show or close a figure depending on whether interactive plots are enabled."""
    if show_plots:
        import matplotlib.pyplot as plt

        plt.show()
    else:
        import matplotlib.pyplot as plt

        plt.close(fig)


def open_file(path: Path) -> None:
    """Open one file using the operating system's default application."""
    if not path.exists():
        return

    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", str(path)], check=False)
    elif os.name == "nt":
        subprocess.run(["cmd", "/c", "start", "", str(path)], check=False)


def open_files(paths: list[Path]) -> None:
    """Open a list of files, skipping duplicates and missing files."""
    seen = set()

    for path in paths:
        resolved = Path(path)
        if resolved in seen or not resolved.exists():
            continue

        seen.add(resolved)
        open_file(resolved)


def pipeline_chart_paths(current_ticker: str | None = None) -> list[Path]:
    """Return the ticker-labeled charts created by the full pipeline."""
    current_ticker = current_ticker or ticker
    filenames = [
        "skill_luck_summary.png",
        "rcsi_bar_chart.png",
        "regime_sharpe_chart.png",
        "regime_heatmap.png",
        "equity_curve.png",
        "monte_carlo_robustness_rcsi.png",
        "monte_carlo_robustness_percentile.png",
        "p_value_chart.png",
    ]
    monte_carlo_filenames = [f"{agent_name}_monte_carlo.png" for agent_name in AGENT_ORDER]
    filenames = filenames[:5] + monte_carlo_filenames + filenames[5:]
    return [ticker_chart_path(filename, current_ticker) for filename in filenames]
