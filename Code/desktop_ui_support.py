"""Reusable Qt widgets and helpers for the desktop application."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSize, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTableView,
    QVBoxLayout,
    QWidget,
)


def clear_layout(layout) -> None:
    """Delete every widget and nested layout from a Qt layout."""
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child_layout = item.layout()
        if widget is not None:
            widget.hide()
            widget.setParent(None)
            widget.deleteLater()
        elif child_layout is not None:
            clear_layout(child_layout)
            child_layout.deleteLater()


def scalar_to_text(value) -> str:
    """Convert one DataFrame scalar into a readable cell string."""
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def open_path(path: Path) -> None:
    """Open a file or folder with the operating system default application."""
    resolved = Path(path)
    if not resolved.exists():
        return
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(resolved)))


def reveal_path(path: Path) -> None:
    """Reveal a file in the platform file manager when possible."""
    resolved = Path(path)
    if not resolved.exists():
        return

    if sys.platform == "darwin":
        subprocess.run(["open", "-R", str(resolved)], check=False)
        return
    if sys.platform.startswith("linux"):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(resolved.parent)))
        return
    if sys.platform.startswith("win"):
        subprocess.run(["explorer", "/select,", str(resolved)], check=False)
        return

    QDesktopServices.openUrl(QUrl.fromLocalFile(str(resolved.parent)))


class PandasTableModel(QAbstractTableModel):
    """Expose a pandas DataFrame inside a QTableView."""

    def __init__(self, dataframe: pd.DataFrame | None = None) -> None:
        super().__init__()
        self._dataframe = pd.DataFrame() if dataframe is None else dataframe.copy()

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return the currently displayed DataFrame."""
        return self._dataframe

    def set_dataframe(self, dataframe: pd.DataFrame | None) -> None:
        """Replace the underlying DataFrame."""
        self.beginResetModel()
        self._dataframe = pd.DataFrame() if dataframe is None else dataframe.copy()
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._dataframe.index)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._dataframe.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        value = self._dataframe.iat[index.row(), index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return scalar_to_text(value)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if isinstance(value, (int, float)) and not pd.isna(value):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return scalar_to_text(self._dataframe.columns[section])
        return scalar_to_text(self._dataframe.index[section])


class SummaryMetricsWidget(QWidget):
    """Render summary metrics in a plain two-column table."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(False)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setWordWrap(True)
        self.table.setTextElideMode(Qt.TextElideMode.ElideNone)
        layout.addWidget(self.table)

    def set_metrics(self, metrics: list[tuple[str, str]]) -> None:
        """Populate the summary table."""
        self.table.setRowCount(len(metrics))
        if not metrics:
            self.table.setRowCount(1)
            self.table.setItem(0, 0, QTableWidgetItem("Status"))
            self.table.setItem(0, 1, QTableWidgetItem("No summary metrics are available yet."))
            self.table.resizeColumnsToContents()
            self.table.resizeRowsToContents()
            return

        for row_index, (label_text, value_text) in enumerate(metrics):
            label_item = QTableWidgetItem(label_text)
            value_item = QTableWidgetItem(value_text)
            label_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            value_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(row_index, 0, label_item)
            self.table.setItem(row_index, 1, value_item)

        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()


class DataFramePane(QWidget):
    """A reusable table panel with file actions and inline status text."""

    def __init__(self, empty_message: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._empty_message = empty_message
        self._active_path: Path | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        self.title_label = QLabel("")
        self.title_label.setStyleSheet("font-size: 13px; font-weight: 600;")
        self.title_label.setWordWrap(True)

        header_layout.addWidget(self.title_label, 1)

        self.open_button = QPushButton("Open File")
        self.open_button.clicked.connect(self._open_current_path)
        self.open_button.setEnabled(False)
        header_layout.addWidget(self.open_button)

        self.reveal_button = QPushButton("Reveal")
        self.reveal_button.clicked.connect(self._reveal_current_path)
        self.reveal_button.setEnabled(False)
        header_layout.addWidget(self.reveal_button)

        layout.addLayout(header_layout)

        self.path_label = QLabel("")
        self.path_label.setWordWrap(False)
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.path_label.setStyleSheet("color: #404040;")
        layout.addWidget(self.path_label)

        self.message_label = QLabel(self._empty_message)
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet("color: #404040;")
        layout.addWidget(self.message_label)

        self.table_view = QTableView()
        self.table_view.setAlternatingRowColors(True)
        self.table_view.verticalHeader().setVisible(False)
        self.table_view.verticalHeader().setDefaultSectionSize(24)
        self.table_view.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.table_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.table_view.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.table_view.setWordWrap(False)
        self.table_view.setHorizontalScrollMode(QTableView.ScrollMode.ScrollPerPixel)
        self.table_view.setVerticalScrollMode(QTableView.ScrollMode.ScrollPerPixel)
        self.table_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.table_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.model = PandasTableModel()
        self.table_view.setModel(self.model)
        self.table_view.setSortingEnabled(False)
        self.table_view.horizontalHeader().setStretchLastSection(False)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table_view.horizontalHeader().setSectionsMovable(True)
        self.table_view.horizontalHeader().setMinimumSectionSize(80)
        layout.addWidget(self.table_view, 1)

    def set_content(
        self,
        *,
        title: str,
        dataframe: pd.DataFrame | None,
        path: Path | None = None,
        message: str | None = None,
    ) -> None:
        """Replace the displayed table and file actions."""
        self._active_path = path if path is not None and path.exists() else None
        self.title_label.setText(title)
        if path is not None:
            self.path_label.setText(f"Saved file: {path.name}")
            self.path_label.setToolTip(str(path))
        else:
            self.path_label.setText("")
            self.path_label.setToolTip("")
        self.open_button.setEnabled(self._active_path is not None)
        self.reveal_button.setEnabled(self._active_path is not None)

        if dataframe is None:
            self.model.set_dataframe(pd.DataFrame())
            self.message_label.setText(message or self._empty_message)
            self.message_label.show()
            return

        self.model.set_dataframe(dataframe)
        self.message_label.setText(message or "")
        self.message_label.setVisible(bool(message))
        self.table_view.resizeColumnsToContents()
        self.table_view.verticalScrollBar().setValue(0)

    def _open_current_path(self) -> None:
        if self._active_path is not None:
            open_path(self._active_path)

    def _reveal_current_path(self) -> None:
        if self._active_path is not None:
            reveal_path(self._active_path)


class MatplotlibFigurePane(QWidget):
    """Show one direct matplotlib figure using a stable embedded canvas."""

    def __init__(self, empty_message: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._empty_message = empty_message
        self._figure = None
        self._canvas: FigureCanvasQTAgg | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.title_label = QLabel("Chart View")
        self.title_label.setStyleSheet("font-size: 13px; font-weight: 600;")
        self.title_label.setWordWrap(True)
        layout.addWidget(self.title_label)

        self.message_label = QLabel(empty_message)
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet("color: #404040;")
        layout.addWidget(self.message_label)

        self.canvas_host = QWidget()
        self.canvas_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas_host.setStyleSheet("background: #ffffff;")
        self.canvas_layout = QVBoxLayout(self.canvas_host)
        self.canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas_layout.setSpacing(0)
        layout.addWidget(self.canvas_host, 1)

    def set_figure(self, *, title: str, figure, message: str = "") -> None:
        """Display one interactive matplotlib figure."""
        self._clear_canvas()
        self.title_label.setText(title)
        self.title_label.hide()
        self.message_label.setText(message)
        self.message_label.setVisible(bool(message))

        self._figure = figure
        self._canvas = FigureCanvasQTAgg(figure)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._canvas.setStyleSheet("background: #ffffff;")
        self._canvas.setMinimumSize(QSize(940, 720))
        self.canvas_layout.addWidget(self._canvas, 1)
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._canvas.setFocus()
        self.canvas_layout.activate()
        self._canvas.draw()
        self._canvas.repaint()

    def set_message(self, *, title: str, message: str) -> None:
        """Show a plain status message instead of a figure."""
        self._clear_canvas()
        self.title_label.setText(title)
        self.title_label.show()
        self.message_label.setText(message)
        self.message_label.show()

    def _clear_canvas(self) -> None:
        current_figure = self._figure
        self._figure = None
        self._canvas = None
        clear_layout(self.canvas_layout)
        self.canvas_host.update()
        if current_figure is not None:
            plt.close(current_figure)


class ImageGalleryWidget(QWidget):
    """Scroll through saved chart images and related exported files."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area, 1)

        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(8, 8, 8, 8)
        self.container_layout.setSpacing(16)
        self.scroll_area.setWidget(self.container)

    def set_artifacts(self, image_paths: list[Path], other_paths: list[Path]) -> None:
        """Populate the gallery with saved images and file actions."""
        clear_layout(self.container_layout)

        if other_paths:
            pdf_title = QLabel("Exported Files")
            pdf_title.setStyleSheet("font-size: 16px; font-weight: 700;")
            self.container_layout.addWidget(pdf_title)
            for path in other_paths:
                self.container_layout.addWidget(self._build_file_row(path))

        if not image_paths:
            placeholder = QLabel("No chart images are available yet.")
            placeholder.setWordWrap(True)
            self.container_layout.addWidget(placeholder)
            self.container_layout.addStretch(1)
            return

        images_title = QLabel("Charts")
        images_title.setStyleSheet("font-size: 16px; font-weight: 700;")
        self.container_layout.addWidget(images_title)

        for path in image_paths:
            chart_box = QFrame()
            chart_box.setFrameShape(QFrame.Shape.Box)
            chart_layout = QVBoxLayout(chart_box)
            chart_layout.setContentsMargins(12, 12, 12, 12)
            chart_layout.setSpacing(10)

            header_layout = QHBoxLayout()
            title_label = QLabel(path.name)
            header_layout.addWidget(title_label, 1)

            open_button = QPushButton("Open")
            open_button.clicked.connect(lambda _checked=False, current_path=path: open_path(current_path))
            header_layout.addWidget(open_button)

            reveal_button = QPushButton("Reveal")
            reveal_button.clicked.connect(
                lambda _checked=False, current_path=path: reveal_path(current_path)
            )
            header_layout.addWidget(reveal_button)

            chart_layout.addLayout(header_layout)

            pixmap = QPixmap(str(path))
            image_label = QLabel()
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            if pixmap.isNull():
                image_label.setText(f"Unable to load {path.name}")
            else:
                scaled = pixmap.scaled(
                    QSize(1100, 900),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                image_label.setPixmap(scaled)
            chart_layout.addWidget(image_label)
            self.container_layout.addWidget(chart_box)

        self.container_layout.addStretch(1)

    def _build_file_row(self, path: Path) -> QWidget:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)

        name_label = QLabel(path.name)
        name_label.setWordWrap(True)
        row_layout.addWidget(name_label, 1)

        open_button = QPushButton("Open")
        open_button.clicked.connect(lambda _checked=False, current_path=path: open_path(current_path))
        row_layout.addWidget(open_button)

        reveal_button = QPushButton("Reveal")
        reveal_button.clicked.connect(
            lambda _checked=False, current_path=path: reveal_path(current_path)
        )
        row_layout.addWidget(reveal_button)

        return row_widget
