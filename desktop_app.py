"""PySide6 desktop interface for the Virtu Fortuna research workflows."""

from __future__ import annotations

from datetime import datetime
import importlib
from pathlib import Path
import sys
import traceback

import matplotlib.pyplot as plt
import pandas as pd

try:
    from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
    from PySide6.QtGui import QAction, QFont
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QFrame,
        QGroupBox,
        QHeaderView,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSplitter,
        QStatusBar,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PySide6 is required for the desktop application. "
        "Install it with: pip install -r requirements.txt"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent
CODE_DIR = PROJECT_ROOT / "Code"
for candidate in (PROJECT_ROOT, CODE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from Code.desktop_ui_support import (  # noqa: E402
    DataFramePane,
    MatplotlibFigurePane,
    SummaryMetricsWidget,
    open_path,
    reveal_path,
)
from Code.comparison_conclusion import (  # noqa: E402
    build_comparison_conclusion_text,
    can_build_comparison_conclusion,
)
from Code.pipeline_utils import charts_dir, data_clean_dir, data_raw_dir, pipeline_chart_paths  # noqa: E402
from Code.strategy_config import AGENT_DISPLAY_NAMES, AGENT_ORDER, BENCHMARK_NAME, format_strategy_name  # noqa: E402
from Code.workflow_runner import (  # noqa: E402
    WorkflowEvent,
    WorkflowRunResult,
    benchmark_curve_path,
    ensure_combined_chart_pdf,
    existing_tickers,
    read_csv_if_exists,
    run_single_ticker_pipeline,
    ticker_chart_files,
    ticker_data_files,
    ticker_trade_files,
)


APP_TITLE = "Virtu Fortuna Desktop Research App"
DEFAULT_SINGLE_TICKER = "SPY"
PLAIN_STYLESHEET = """
QMainWindow, QWidget {
    background: #f7f7f7;
    color: #000000;
    font-family: "Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif", serif;
    font-size: 13px;
}
QGroupBox {
    background: #ffffff;
    border: 1px solid #8e8e8e;
    border-radius: 0px;
    margin-top: 0px;
    padding: 24px 8px 8px 8px;
}
QGroupBox::title {
    subcontrol-origin: padding;
    subcontrol-position: top left;
    left: 8px;
    top: 5px;
    padding: 0;
    background: #ffffff;
}
QPushButton {
    background: #efefef;
    border: 1px solid #8a8a8a;
    border-radius: 0px;
    padding: 4px 10px;
    min-height: 24px;
}
QPushButton:pressed {
    background: #dddddd;
}
QLineEdit, QComboBox, QTableView, QTableWidget {
    background: #ffffff;
    border: 1px solid #8f8f8f;
    border-radius: 0px;
}
QLineEdit, QComboBox {
    min-height: 24px;
    padding: 2px 4px;
}
QHeaderView::section {
    background: #ececec;
    border: 1px solid #b5b5b5;
    padding: 4px;
}
QTableView, QTableWidget {
    gridline-color: #c5c5c5;
}
QMenuBar, QMenu, QStatusBar {
    background: #efefef;
}
QSplitter::handle {
    background: #d0d0d0;
}
QScrollArea {
    background: #f7f7f7;
    border: none;
}
QScrollBar:vertical {
    background: #efefef;
    width: 18px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #b0b0b0;
    min-height: 36px;
    border: 1px solid #8a8a8a;
}
QScrollBar:horizontal {
    background: #efefef;
    height: 16px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: #b0b0b0;
    min-width: 36px;
    border: 1px solid #8a8a8a;
}
QScrollBar::add-line, QScrollBar::sub-line {
    background: #efefef;
    border: 1px solid #8a8a8a;
}
"""

SPECIAL_ARTIFACT_LABELS = {
    "full_comparison": "Full Comparison",
    "agent_comparison": "Agent Comparison",
    "monte_carlo_summary": "Monte Carlo Summary",
    "monte_carlo_robustness_summary": "Monte Carlo Robustness Summary",
    "monte_carlo_robustness_runs": "Monte Carlo Robustness Runs",
    "rcsi": "RCSI",
    "regime_analysis": "Regime Analysis",
    "buy_hold_metrics": "Buy and Hold Metrics",
    "buy_hold_curve": "Buy and Hold Curve",
    "features": "Feature Table",
    "regimes": "Regime Table",
    "monte_carlo_results": "Monte Carlo Results",
    "monte_carlo_returns": "Monte Carlo Returns",
    "momentum_relative_strength_universe": "Relative Strength Universe",
}

COMPARISON_FILE_SUFFIXES = [
    "full_comparison",
    "agent_comparison",
    "monte_carlo_summary",
    "monte_carlo_robustness_summary",
    "monte_carlo_robustness_runs",
    "rcsi",
    "regime_analysis",
    "buy_hold_metrics",
]


def format_duration(seconds: float | None) -> str:
    """Format a runtime in seconds."""
    if seconds is None:
        return "n/a"
    return f"{seconds:.1f}s"


def format_timestamp(path: Path) -> str:
    """Return a human-friendly modification time."""
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        return "n/a"


def file_size_text(path: Path) -> str:
    """Format a file size in kilobytes."""
    try:
        return f"{path.stat().st_size / 1024.0:,.1f} KB"
    except OSError:
        return "n/a"


def best_strategy_row(comparison_df: pd.DataFrame | None):
    """Return the strongest strategy row by cumulative return."""
    if comparison_df is None or comparison_df.empty or "agent" not in comparison_df.columns:
        return None

    working_df = comparison_df.copy()
    working_df = working_df.loc[working_df["agent"].astype(str) != BENCHMARK_NAME].copy()
    if working_df.empty or "cumulative_return" not in working_df.columns:
        return None

    working_df["cumulative_return"] = pd.to_numeric(
        working_df["cumulative_return"],
        errors="coerce",
    )
    working_df = working_df.dropna(subset=["cumulative_return"])
    if working_df.empty:
        return None

    return working_df.sort_values("cumulative_return", ascending=False).iloc[0]


def lowest_p_value_row(comparison_df: pd.DataFrame | None):
    """Return the row with the lowest available p-value."""
    if comparison_df is None or comparison_df.empty or "p_value" not in comparison_df.columns:
        return None

    working_df = comparison_df.copy()
    working_df["p_value"] = pd.to_numeric(working_df["p_value"], errors="coerce")
    working_df = working_df.dropna(subset=["p_value"])
    if working_df.empty:
        return None

    return working_df.sort_values("p_value", ascending=True).iloc[0]


def artifact_label(path: Path, ticker: str | None = None) -> str:
    """Convert a saved file path into a readable UI label."""
    stem = path.stem
    if ticker:
        normalized_ticker = ticker.upper()
        if stem == normalized_ticker:
            return "Raw Price History"
        prefix = f"{normalized_ticker}_"
        if stem.startswith(prefix):
            stem = stem[len(prefix):]

    if stem in SPECIAL_ARTIFACT_LABELS:
        return SPECIAL_ARTIFACT_LABELS[stem]

    for agent_name in sorted(AGENT_DISPLAY_NAMES.keys(), key=len, reverse=True):
        prefix = f"{agent_name}_"
        if stem.startswith(prefix):
            remainder = stem[len(prefix):]
            suffix = SPECIAL_ARTIFACT_LABELS.get(remainder, remainder.replace("_", " ").title())
            return f"{format_strategy_name(agent_name, short=True)} {suffix}"

    return stem.replace("_", " ").title()


def comparison_entries_for_ticker(ticker: str) -> list[tuple[str, Path]]:
    """Return the main comparison and metrics tables for one ticker."""
    normalized_ticker = ticker.upper()
    entries: list[tuple[str, Path]] = []
    for suffix in COMPARISON_FILE_SUFFIXES:
        path = data_clean_dir() / f"{normalized_ticker}_{suffix}.csv"
        if path.exists():
            entries.append((artifact_label(path, normalized_ticker), path))
    return entries


def ordered_trade_entries(ticker: str) -> list[tuple[str, Path]]:
    """Return labeled trade logs for one ticker."""
    normalized_ticker = ticker.upper()
    entries = [
        (artifact_label(path, normalized_ticker), path)
        for path in ticker_trade_files(normalized_ticker)
    ]
    buy_hold_path = benchmark_curve_path(normalized_ticker)
    if buy_hold_path.exists():
        entries.append((artifact_label(buy_hold_path, normalized_ticker), buy_hold_path))
    return entries


def ordered_chart_entries(ticker: str) -> list[tuple[str, Path]]:
    """Return the standard chart set first, followed by any extra saved chart files."""
    normalized_ticker = ticker.upper()
    entries: list[tuple[str, Path]] = []
    seen_paths: set[Path] = set()
    allowed_suffixes = {".png", ".jpg", ".jpeg"}

    for path in pipeline_chart_paths(normalized_ticker):
        if path.exists():
            resolved = path.resolve()
            seen_paths.add(resolved)
            entries.append((artifact_label(path, normalized_ticker), path))

    for path in ticker_chart_files(normalized_ticker):
        if path.suffix.lower() not in allowed_suffixes:
            continue
        resolved = path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        entries.append((artifact_label(path, normalized_ticker), path))

    return entries


def chart_suffix_for_ticker(path: Path, ticker: str) -> str:
    """Return the chart suffix after the ticker prefix."""
    normalized_ticker = ticker.upper()
    stem = path.stem
    prefix = f"{normalized_ticker}_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return stem


def _render_with_patched_plot_hooks(module_name: str, ticker: str, callback):
    """Create one matplotlib figure without opening standalone chart windows."""
    module = importlib.import_module(module_name)
    plot_config = importlib.import_module("plot_config")

    original_module_ticker = getattr(module, "ticker", None)
    original_module_show = getattr(module, "show_chart", None)
    original_module_save = getattr(module, "save_chart", None)
    original_plot_show = plot_config.show_chart
    original_plot_save = plot_config.save_chart

    existing_numbers = set(plt.get_fignums())

    try:
        if hasattr(module, "ticker"):
            module.ticker = ticker
        if original_module_show is not None:
            module.show_chart = lambda: None
        if original_module_save is not None:
            module.save_chart = lambda fig, filename: Path(filename)
        plot_config.show_chart = lambda: None
        plot_config.save_chart = lambda fig, filename: Path(filename)
        callback(module)
        new_numbers = [number for number in plt.get_fignums() if number not in existing_numbers]
        if not new_numbers:
            return None
        figure = plt.figure(new_numbers[-1])
        optimize_figure_for_display(figure)
        return figure
    finally:
        if hasattr(module, "ticker"):
            module.ticker = original_module_ticker
        if original_module_show is not None:
            module.show_chart = original_module_show
        if original_module_save is not None:
            module.save_chart = original_module_save
        plot_config.show_chart = original_plot_show
        plot_config.save_chart = original_plot_save


def build_direct_chart_figure(ticker: str, chart_path: Path):
    """Render one saved chart directly in-app when a native matplotlib renderer exists."""
    suffix = chart_suffix_for_ticker(chart_path, ticker)

    if suffix == "equity_curve":
        return _render_with_patched_plot_hooks(
            "equity_curve",
            ticker,
            lambda module: module.main(),
        )
    if suffix == "rcsi_bar_chart":
        return _render_with_patched_plot_hooks(
            "rcsi_plot",
            ticker,
            lambda module: module.main(),
        )
    if suffix == "regime_sharpe_chart":
        return _render_with_patched_plot_hooks(
            "regime_plot",
            ticker,
            lambda module: module.main(),
        )
    if suffix == "regime_heatmap":
        return _render_with_patched_plot_hooks(
            "rcsi_heatmap",
            ticker,
            lambda module: module.main(),
        )
    if suffix == "p_value_chart":
        return _render_with_patched_plot_hooks(
            "p_value_plot",
            ticker,
            lambda module: module.main(),
        )
    if suffix == "skill_luck_summary":
        return _render_with_patched_plot_hooks(
            "strategy_verdict_plot",
            ticker,
            lambda module: module.main(),
        )
    if suffix == "monte_carlo_robustness_rcsi":
        return _render_with_patched_plot_hooks(
            "monte_carlo_robustness_plot",
            ticker,
            lambda module: module.create_rcsi_stability_chart(module.load_robustness_data()[1]),
        )
    if suffix == "monte_carlo_robustness_percentile":
        return _render_with_patched_plot_hooks(
            "monte_carlo_robustness_plot",
            ticker,
            lambda module: module.create_percentile_stability_chart(module.load_robustness_data()[1]),
        )
    if suffix.endswith("_monte_carlo"):
        agent_name = suffix[: -len("_monte_carlo")]
        if agent_name in AGENT_ORDER:
            return _render_with_patched_plot_hooks(
                "monte_carlo_plot",
                ticker,
                lambda module: module.create_monte_carlo_chart(agent_name),
            )

    return None


def optimize_figure_for_display(figure) -> None:
    """Give embedded charts more breathing room than the saved-file defaults."""
    figure.set_dpi(110)
    figure.set_size_inches(13.8, 9.0, forward=True)
    figure.patch.set_facecolor("#ffffff")

    for axis in figure.axes:
        try:
            axis.set_facecolor("#ffffff")
        except Exception:
            pass
        try:
            axis.title.set_wrap(True)
            axis.title.set_y(1.01)
        except Exception:
            pass

        try:
            axis.xaxis.labelpad = max(axis.xaxis.labelpad, 10)
            axis.yaxis.labelpad = max(axis.yaxis.labelpad, 10)
        except Exception:
            pass

        tick_labels = [label.get_text() for label in axis.get_xticklabels() if label.get_text()]
        if tick_labels:
            longest_label = max(len(label_text) for label_text in tick_labels)
            if len(tick_labels) >= 5 or longest_label > 12:
                for label in axis.get_xticklabels():
                    label.set_rotation(28)
                    label.set_ha("right")

    try:
        figure.tight_layout(rect=(0.03, 0.05, 0.97, 0.95), pad=1.6)
    except Exception:
        try:
            figure.subplots_adjust(left=0.08, right=0.97, bottom=0.11, top=0.91)
        except Exception:
            pass


class WorkflowWorker(QObject):
    """Run the single-ticker workflow in a background thread."""

    event_emitted = Signal(object)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, ticker: str) -> None:
        super().__init__()
        self.ticker = ticker

    @Slot()
    def run(self) -> None:
        """Execute the real single-ticker workflow."""
        try:
            result = run_single_ticker_pipeline(
                ticker=self.ticker,
                env_overrides={},
                event_callback=self._handle_event,
                interactive_chart_open=False,
            )
            self.completed.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())

    def _handle_event(self, event: WorkflowEvent) -> None:
        self.event_emitted.emit(event)


class MainWindow(QMainWindow):
    """Simplified desktop application window."""

    def __init__(self) -> None:
        super().__init__()
        app = QApplication.instance()
        if app is not None:
            app.setStyle("Fusion")
            serif_font = QFont("Times New Roman", 11)
            serif_font.setStyleHint(QFont.StyleHint.Serif)
            app.setFont(serif_font)
            app.setStyleSheet(PLAIN_STYLESHEET)

        self.setWindowTitle(APP_TITLE)
        self.resize(1720, 1080)

        self.last_result: WorkflowRunResult | None = None
        self.worker_thread: QThread | None = None
        self.worker: WorkflowWorker | None = None
        self.is_running = False
        self.current_comparison_entries: list[tuple[str, Path]] = []
        self.current_chart_entries: list[tuple[str, Path]] = []
        self.current_trade_entries: list[tuple[str, Path]] = []
        self.current_raw_entries: list[tuple[str, str, Path]] = []
        self.current_chart_pdf: Path | None = None
        self._initial_refresh_done = False

        self._build_ui()
        self._build_menu()

    def _build_ui(self) -> None:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        dashboard_widget = QWidget()
        outer_layout = QVBoxLayout(dashboard_widget)
        outer_layout.setContentsMargins(10, 10, 10, 10)
        outer_layout.setSpacing(8)

        outer_layout.addWidget(self._build_input_section())

        summary_section = self._build_summary_section()
        summary_section.setMaximumHeight(190)
        comparison_section = self._build_comparison_section()
        comparison_section.setMinimumWidth(680)
        comparison_section.setMinimumHeight(500)
        trades_section = self._build_trades_section()
        raw_output_section = self._build_raw_output_section()

        left_splitter = QSplitter(Qt.Orientation.Vertical)
        left_splitter.addWidget(summary_section)
        left_splitter.addWidget(comparison_section)
        left_splitter.setChildrenCollapsible(False)
        left_splitter.setStretchFactor(0, 0)
        left_splitter.setStretchFactor(1, 1)
        left_splitter.setSizes([150, 730])

        charts_section = self._build_charts_section()

        upper_splitter = QSplitter(Qt.Orientation.Horizontal)
        upper_splitter.addWidget(left_splitter)
        upper_splitter.addWidget(charts_section)
        upper_splitter.setChildrenCollapsible(False)
        upper_splitter.setStretchFactor(0, 1)
        upper_splitter.setStretchFactor(1, 2)
        upper_splitter.setSizes([620, 1260])

        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        bottom_splitter.addWidget(trades_section)
        bottom_splitter.addWidget(raw_output_section)
        bottom_splitter.setChildrenCollapsible(False)
        bottom_splitter.setStretchFactor(0, 1)
        bottom_splitter.setStretchFactor(1, 1)
        bottom_splitter.setSizes([980, 700])

        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(upper_splitter)
        main_splitter.addWidget(bottom_splitter)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 0)
        main_splitter.setSizes([860, 180])
        outer_layout.addWidget(main_splitter, 1)

        dashboard_widget.setMinimumHeight(1420)
        scroll_area.setWidget(dashboard_widget)
        self.setCentralWidget(scroll_area)

        status_bar = QStatusBar()
        status_bar.showMessage("Ready")
        self.setStatusBar(status_bar)

    def showEvent(self, event) -> None:
        """Load saved outputs after the window has a real layout size."""
        super().showEvent(event)
        if not self._initial_refresh_done:
            self._initial_refresh_done = True
            QTimer.singleShot(0, self.refresh_all_views)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")

        open_raw_action = QAction("Open Raw Data Folder", self)
        open_raw_action.triggered.connect(lambda: open_path(data_raw_dir()))
        file_menu.addAction(open_raw_action)

        open_clean_action = QAction("Open Clean Data Folder", self)
        open_clean_action.triggered.connect(lambda: open_path(data_clean_dir()))
        file_menu.addAction(open_clean_action)

        open_charts_action = QAction("Open Charts Folder", self)
        open_charts_action.triggered.connect(lambda: open_path(charts_dir()))
        file_menu.addAction(open_charts_action)

    def _build_input_section(self) -> QWidget:
        section = QGroupBox("Run Pipeline")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Ticker"))

        self.ticker_input = QLineEdit()
        available_tickers = existing_tickers()
        self.ticker_input.setText(available_tickers[0] if available_tickers else DEFAULT_SINGLE_TICKER)
        self.ticker_input.setPlaceholderText("SPY")
        self.ticker_input.editingFinished.connect(self.refresh_all_views)
        self.ticker_input.returnPressed.connect(self.start_run)
        top_row.addWidget(self.ticker_input, 1)

        self.run_button = QPushButton("Run")
        self.run_button.setMinimumWidth(90)
        self.run_button.clicked.connect(self.start_run)
        top_row.addWidget(self.run_button)

        layout.addLayout(top_row)

        self.status_label = QLabel(
            "Enter a ticker and run the full research pipeline. Existing saved outputs load automatically for the current ticker."
        )
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        return section

    def _build_summary_section(self) -> QWidget:
        section = QGroupBox("Summary Metrics")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.summary_metrics_widget = SummaryMetricsWidget()
        layout.addWidget(self.summary_metrics_widget)
        return section

    def _build_comparison_section(self) -> QWidget:
        section = QGroupBox("Comparison Table")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Result Table"))
        self.comparison_combo = QComboBox()
        self.comparison_combo.currentIndexChanged.connect(self.update_comparison_view)
        selector_row.addWidget(self.comparison_combo, 1)
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_comparison_entries)
        selector_row.addWidget(refresh_button)
        layout.addLayout(selector_row)

        self.comparison_panel = DataFramePane("No comparison table is available for the current ticker.")
        self.comparison_panel.table_view.setMinimumHeight(420)
        layout.addWidget(self.comparison_panel, 1)

        self.comparison_conclusion_label = QLabel(
            "Conclusion and Interpretation will appear here after the Full Comparison results are available."
        )
        self.comparison_conclusion_label.setWordWrap(True)
        self.comparison_conclusion_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.comparison_conclusion_label.setStyleSheet("color: #202020;")
        layout.addWidget(self.comparison_conclusion_label)
        return section

    def _build_charts_section(self) -> QWidget:
        section = QGroupBox("Charts")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Chart"))

        self.charts_combo = QComboBox()
        self.charts_combo.currentIndexChanged.connect(self.update_chart_view)
        selector_row.addWidget(self.charts_combo, 1)

        self.chart_previous_button = QPushButton("Previous")
        self.chart_previous_button.clicked.connect(self.show_previous_chart)
        selector_row.addWidget(self.chart_previous_button)

        self.chart_next_button = QPushButton("Next")
        self.chart_next_button.clicked.connect(self.show_next_chart)
        selector_row.addWidget(self.chart_next_button)

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_chart_entries)
        selector_row.addWidget(refresh_button)
        layout.addLayout(selector_row)

        action_row = QHBoxLayout()
        self.open_chart_button = QPushButton("Open File")
        self.open_chart_button.clicked.connect(self.open_selected_chart_file)
        action_row.addWidget(self.open_chart_button)

        self.reveal_chart_button = QPushButton("Reveal")
        self.reveal_chart_button.clicked.connect(self.reveal_selected_chart_file)
        action_row.addWidget(self.reveal_chart_button)

        self.open_chart_pdf_button = QPushButton("Open Combined PDF")
        self.open_chart_pdf_button.clicked.connect(self.open_chart_pdf)
        action_row.addWidget(self.open_chart_pdf_button)

        action_row.addStretch(1)
        layout.addLayout(action_row)

        self.chart_status_label = QLabel(
            "Direct chart view."
        )
        self.chart_status_label.setWordWrap(False)
        layout.addWidget(self.chart_status_label)

        self.charts_widget = MatplotlibFigurePane(
            "No charts are available for the current ticker."
        )
        self.charts_widget.setMinimumHeight(920)
        layout.addWidget(self.charts_widget, 1)
        return section

    def _build_trades_section(self) -> QWidget:
        section = QGroupBox("Trades Log")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Trade File"))
        self.trades_combo = QComboBox()
        self.trades_combo.currentIndexChanged.connect(self.update_trades_view)
        selector_row.addWidget(self.trades_combo, 1)
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_trade_entries)
        selector_row.addWidget(refresh_button)
        layout.addLayout(selector_row)

        self.trades_panel = DataFramePane("No trade log is available for the current ticker.")
        self.trades_panel.table_view.setMinimumHeight(260)
        layout.addWidget(self.trades_panel, 1)
        return section

    def _build_raw_output_section(self) -> QWidget:
        section = QGroupBox("Raw Output")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        button_row = QHBoxLayout()
        self.open_selected_file_button = QPushButton("Open Selected")
        self.open_selected_file_button.clicked.connect(self.open_selected_raw_file)
        button_row.addWidget(self.open_selected_file_button)

        self.reveal_selected_file_button = QPushButton("Reveal Selected")
        self.reveal_selected_file_button.clicked.connect(self.reveal_selected_raw_file)
        button_row.addWidget(self.reveal_selected_file_button)

        button_row.addStretch(1)

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_raw_file_table)
        button_row.addWidget(refresh_button)
        layout.addLayout(button_row)

        self.raw_files_table = QTableWidget(0, 6)
        self.raw_files_table.setHorizontalHeaderLabels(
            ["Category", "Label", "File", "Modified", "Size", "Path"]
        )
        self.raw_files_table.verticalHeader().setVisible(False)
        self.raw_files_table.horizontalHeader().setStretchLastSection(False)
        self.raw_files_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.raw_files_table.horizontalHeader().setSectionsMovable(True)
        self.raw_files_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.raw_files_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.raw_files_table.setWordWrap(False)
        self.raw_files_table.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.raw_files_table.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self.raw_files_table.setVerticalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self.raw_files_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.raw_files_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.raw_files_table.setMinimumHeight(220)
        self.raw_files_table.cellDoubleClicked.connect(
            lambda _row, _column: self.open_selected_raw_file()
        )
        layout.addWidget(self.raw_files_table, 1)
        return section

    def current_ticker(self) -> str:
        ticker = self.ticker_input.text().strip().upper()
        return ticker or DEFAULT_SINGLE_TICKER

    @Slot()
    def start_run(self) -> None:
        """Launch the real single-ticker workflow in a worker thread."""
        if self.is_running:
            return

        ticker = self.current_ticker()
        if not ticker:
            QMessageBox.warning(self, "Missing Ticker", "Enter a ticker before running the pipeline.")
            return

        self.is_running = True
        self.run_button.setEnabled(False)
        self.ticker_input.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.status_label.setText(
            "Running the full single-ticker research pipeline. The window will stay responsive."
        )
        self.statusBar().showMessage("Running full pipeline...")

        self.worker_thread = QThread(self)
        self.worker = WorkflowWorker(ticker=ticker)
        self.worker.moveToThread(self.worker_thread)
        self.worker.event_emitted.connect(self.handle_workflow_event)
        self.worker.completed.connect(self.handle_workflow_completed)
        self.worker.failed.connect(self.handle_workflow_failed)
        self.worker_thread.started.connect(self.worker.run)
        self.worker_thread.start()

    @Slot(object)
    def handle_workflow_event(self, event: WorkflowEvent) -> None:
        """Update the compact top status area while the worker runs."""
        if event.kind == "workflow_started":
            self.status_label.setText(event.message)
            self.statusBar().showMessage(event.message)
            return

        if event.kind == "step_started":
            total_steps = max(event.total_steps or 1, 1)
            current_step = max((event.step_index or 1) - 1, 0)
            self.progress_bar.setValue(int((current_step / total_steps) * 100))
            self.status_label.setText(
                f"Running step {event.step_index}/{total_steps}: {event.step_label or event.script_name or ''}"
            )
            self.statusBar().showMessage(event.message)
            return

        if event.kind == "step_finished":
            total_steps = max(event.total_steps or 1, 1)
            current_step = max(event.step_index or total_steps, 0)
            self.progress_bar.setValue(int((current_step / total_steps) * 100))
            self.status_label.setText(event.message)
            self.statusBar().showMessage(event.message)
            return

        if event.kind == "workflow_finished":
            self.status_label.setText(event.message)
            self.statusBar().showMessage(event.message)

    @Slot(object)
    def handle_workflow_completed(self, result: WorkflowRunResult) -> None:
        """Refresh the visible sections after a workflow finishes."""
        self.last_result = result
        if result.metadata.get("ticker"):
            self.ticker_input.setText(result.metadata["ticker"])
        self.refresh_all_views()
        self.finish_worker()

        self.status_label.setText(
            "Run completed successfully." if result.success else (result.error_message or "Run failed.")
        )
        self.statusBar().showMessage(self.status_label.text())
        if not result.success:
            QMessageBox.warning(
                self,
                "Pipeline Finished With Errors",
                result.error_message or "The pipeline stopped before completion.",
            )

    @Slot(str)
    def handle_workflow_failed(self, traceback_text: str) -> None:
        """Handle an unexpected worker exception."""
        self.finish_worker()
        self.status_label.setText("The desktop runner hit an unexpected error.")
        self.statusBar().showMessage("Pipeline crashed")
        QMessageBox.critical(
            self,
            "Unexpected Error",
            "The desktop runner hit an unexpected error.\n\n"
            f"{traceback_text}",
        )

    def finish_worker(self) -> None:
        """Restore the top input row after a background run ends."""
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread.deleteLater()
        if self.worker is not None:
            self.worker.deleteLater()

        self.worker_thread = None
        self.worker = None
        self.is_running = False
        self.run_button.setEnabled(True)
        self.ticker_input.setEnabled(True)
        self.progress_bar.hide()
        self.progress_bar.setValue(0)

    def refresh_all_views(self) -> None:
        """Reload every visible section for the current ticker."""
        normalized_ticker = self.current_ticker()
        if self.ticker_input.text() != normalized_ticker:
            self.ticker_input.blockSignals(True)
            self.ticker_input.setText(normalized_ticker)
            self.ticker_input.blockSignals(False)
        self.refresh_summary()
        self.refresh_comparison_entries()
        self.refresh_chart_entries()
        self.refresh_trade_entries()
        self.refresh_raw_file_table()

    def refresh_summary(self) -> None:
        """Refresh the summary metrics section."""
        ticker = self.current_ticker()
        comparison_df = read_csv_if_exists(data_clean_dir() / f"{ticker}_full_comparison.csv")
        robustness_df = read_csv_if_exists(data_clean_dir() / f"{ticker}_monte_carlo_robustness_summary.csv")

        current_result = None
        if self.last_result and self.last_result.metadata.get("ticker") == ticker:
            current_result = self.last_result

        metrics: list[tuple[str, str]] = [
            ("Ticker", ticker),
            (
                "Run Status",
                "Success" if current_result and current_result.success else "Saved Results",
            ),
            ("Run Time", format_duration(current_result.duration_seconds if current_result else None)),
            ("Charts Found", str(len(ticker_chart_files(ticker)))),
        ]

        if comparison_df is not None:
            best_row = best_strategy_row(comparison_df)
            if best_row is not None:
                metrics.append(
                    (
                        "Best Strategy by Return",
                        f"{format_strategy_name(str(best_row['agent']), short=True)} | "
                        f"{float(best_row['cumulative_return']):.2%}",
                    )
                )
            p_value_row = lowest_p_value_row(comparison_df)
            if p_value_row is not None:
                metrics.append(
                    (
                        "Lowest p-value",
                        f"{format_strategy_name(str(p_value_row['agent']), short=True)} | "
                        f"{float(p_value_row['p_value']):.4f}",
                    )
                )

        if robustness_df is not None and "mean_p_value" in robustness_df.columns and not robustness_df.empty:
            working_df = robustness_df.copy()
            working_df["mean_p_value"] = pd.to_numeric(working_df["mean_p_value"], errors="coerce")
            working_df = working_df.dropna(subset=["mean_p_value"])
            if not working_df.empty:
                best_robust = working_df.sort_values("mean_p_value", ascending=True).iloc[0]
                metrics.append(
                    (
                        "Strongest Robustness Case",
                        f"{format_strategy_name(str(best_robust['agent']), short=True)} | "
                        f"mean p={float(best_robust['mean_p_value']):.4f}",
                    )
                )

        self.summary_metrics_widget.set_metrics(metrics)

    def refresh_comparison_entries(self) -> None:
        """Refresh the main comparison table selector."""
        entries = comparison_entries_for_ticker(self.current_ticker())
        self.current_comparison_entries = entries
        self._populate_combo(self.comparison_combo, entries)
        self.update_comparison_view()

    def update_comparison_view(self) -> None:
        """Display the selected comparison/metrics table."""
        entry = self._selected_entry(self.comparison_combo, self.current_comparison_entries)
        if entry is None:
            self.comparison_panel.set_content(
                title="Comparison Results",
                dataframe=None,
                message="No comparison table is available for the current ticker.",
            )
            self.comparison_conclusion_label.setText(
                "Conclusion and Interpretation will appear here after the Full Comparison results are available."
            )
            return

        label, path = entry
        dataframe = read_csv_if_exists(path)
        self.comparison_panel.set_content(
            title=label,
            dataframe=dataframe,
            path=path,
            message=f"{label} is not available yet.",
        )

        if can_build_comparison_conclusion(dataframe):
            self.comparison_conclusion_label.setText(build_comparison_conclusion_text(dataframe))
        else:
            self.comparison_conclusion_label.setText(
                "Conclusion and Interpretation is generated from the Full Comparison table once those results are available."
            )

    def refresh_chart_entries(self) -> None:
        """Refresh the chart selector and viewer."""
        ticker = self.current_ticker()
        self.current_chart_entries = ordered_chart_entries(ticker)
        self.current_chart_pdf = ensure_combined_chart_pdf(ticker)
        self._populate_combo(self.charts_combo, self.current_chart_entries)
        self.update_chart_view()

    def update_chart_view(self) -> None:
        """Display the currently selected chart using a direct matplotlib canvas when possible."""
        entry = self._selected_entry(self.charts_combo, self.current_chart_entries)
        chart_path = entry[1] if entry is not None else None
        self.open_chart_button.setEnabled(chart_path is not None and chart_path.exists())
        self.reveal_chart_button.setEnabled(chart_path is not None and chart_path.exists())
        self.open_chart_pdf_button.setEnabled(
            self.current_chart_pdf is not None and self.current_chart_pdf.exists()
        )

        if entry is None:
            self.chart_status_label.setText("No charts are available for the current ticker.")
            self.charts_widget.set_message(
                title="Chart View",
                message="No charts are available for the current ticker.",
            )
            return

        label, path = entry
        try:
            figure = build_direct_chart_figure(
                self.current_ticker(),
                path,
            )
        except Exception as exc:
            self.chart_status_label.setText(
                f"Direct rendering did not complete for {path.name}. You can still open the saved chart file."
            )
            self.charts_widget.set_message(
                title=label,
                message=(
                    "This chart is available as a saved export, but it could not be redrawn directly "
                    f"in the application.\n\nReason: {exc}"
                ),
            )
            return

        if figure is None:
            self.chart_status_label.setText(
                f"{path.name} is available as a saved export. Use Open File or Open Combined PDF for the full-quality view."
            )
            self.charts_widget.set_message(
                title=label,
                message=(
                    "This chart does not currently have an in-app direct renderer.\n"
                    "Use Open File or Open Combined PDF for the saved full-quality export."
                ),
            )
            return

        self.chart_status_label.setText(
            "Direct chart view."
        )
        self.charts_widget.set_figure(
            title=label,
            figure=figure,
            message="",
        )

    def refresh_trade_entries(self) -> None:
        """Refresh the trade log selector."""
        entries = ordered_trade_entries(self.current_ticker())
        self.current_trade_entries = entries
        self._populate_combo(self.trades_combo, entries)
        self.update_trades_view()

    def update_trades_view(self) -> None:
        """Display the selected trade log."""
        entry = self._selected_entry(self.trades_combo, self.current_trade_entries)
        if entry is None:
            self.trades_panel.set_content(
                title="Trades Log",
                dataframe=None,
                message="No trade log is available for the current ticker.",
            )
            return

        label, path = entry
        self.trades_panel.set_content(
            title=label,
            dataframe=read_csv_if_exists(path),
            path=path,
            message=f"{label} is not available yet.",
        )

    def show_previous_chart(self) -> None:
        """Move to the previous available chart."""
        count = self.charts_combo.count()
        if count <= 1:
            return
        self.charts_combo.setCurrentIndex((self.charts_combo.currentIndex() - 1) % count)

    def show_next_chart(self) -> None:
        """Move to the next available chart."""
        count = self.charts_combo.count()
        if count <= 1:
            return
        self.charts_combo.setCurrentIndex((self.charts_combo.currentIndex() + 1) % count)

    def open_selected_chart_file(self) -> None:
        """Open the selected chart export."""
        entry = self._selected_entry(self.charts_combo, self.current_chart_entries)
        if entry is not None:
            open_path(entry[1])

    def reveal_selected_chart_file(self) -> None:
        """Reveal the selected chart export."""
        entry = self._selected_entry(self.charts_combo, self.current_chart_entries)
        if entry is not None:
            reveal_path(entry[1])

    def open_chart_pdf(self) -> None:
        """Open the combined chart PDF when it exists."""
        if self.current_chart_pdf is not None and self.current_chart_pdf.exists():
            open_path(self.current_chart_pdf)

    def refresh_raw_file_table(self) -> None:
        """Refresh the raw output file list."""
        ticker = self.current_ticker()
        rows: list[tuple[str, str, Path]] = []

        for path in ticker_data_files(ticker):
            rows.append(("Data", artifact_label(path, ticker), path))
        for path in ticker_chart_files(ticker):
            rows.append(("Charts", path.name, path))

        combined_pdf = ensure_combined_chart_pdf(ticker)
        if combined_pdf is not None and combined_pdf.exists():
            rows.append(("Charts", combined_pdf.name, combined_pdf))

        unique_rows: list[tuple[str, str, Path]] = []
        seen_paths: set[Path] = set()
        for category, label, path in rows:
            if not path.exists() or path in seen_paths:
                continue
            seen_paths.add(path)
            unique_rows.append((category, label, path))

        self.current_raw_entries = unique_rows
        self.raw_files_table.setRowCount(len(unique_rows))
        for row_index, (category, label, path) in enumerate(unique_rows):
            values = [
                category,
                label,
                path.name,
                format_timestamp(path),
                file_size_text(path),
                str(path),
            ]
            for column_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, str(path))
                self.raw_files_table.setItem(row_index, column_index, item)
        self.raw_files_table.resizeColumnsToContents()
        self.raw_files_table.resizeRowsToContents()

    def open_selected_raw_file(self) -> None:
        """Open the selected raw-output file."""
        path = self._selected_raw_file_path()
        if path is not None:
            open_path(path)

    def reveal_selected_raw_file(self) -> None:
        """Reveal the selected raw-output file in the file manager."""
        path = self._selected_raw_file_path()
        if path is not None:
            reveal_path(path)

    def _selected_raw_file_path(self) -> Path | None:
        row = self.raw_files_table.currentRow()
        if row < 0:
            return None
        item = self.raw_files_table.item(row, 0)
        if item is None:
            return None
        stored_path = item.data(Qt.ItemDataRole.UserRole)
        if not stored_path:
            return None
        return Path(str(stored_path))

    def _populate_combo(self, combo: QComboBox, entries: list[tuple[str, Path]]) -> None:
        current_path = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for label, path in entries:
            combo.addItem(label, str(path))
        if current_path:
            index = combo.findData(current_path)
            if index >= 0:
                combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def _selected_entry(
        self,
        combo: QComboBox,
        entries: list[tuple[str, Path]],
    ) -> tuple[str, Path] | None:
        stored_path = combo.currentData()
        if not stored_path:
            return entries[0] if entries else None
        for label, path in entries:
            if str(path) == stored_path:
                return (label, path)
        return entries[0] if entries else None


def main() -> None:
    """Launch the desktop application."""
    QApplication.setApplicationName(APP_TITLE)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
 