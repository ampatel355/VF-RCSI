"""Open the current ticker's pipeline charts as one combined PDF."""

import os
from pathlib import Path
import subprocess

from pipeline_utils import pipeline_chart_paths
from PIL import Image


# Every script reads the active ticker from the same environment variable.
ticker = os.environ.get("TICKER", "SPY")
interactive_open = os.environ.get("OPEN_CHARTS_INTERACTIVE", "1") == "1"


def find_chart_files() -> list[Path]:
    """Return the standard chart set created by the pipeline for one ticker."""
    chart_files = [chart_path for chart_path in pipeline_chart_paths(ticker) if chart_path.exists()]

    if not chart_files:
        raise FileNotFoundError(
            f"No pipeline chart files were found for ticker '{ticker}'."
        )

    return chart_files


def combined_chart_pdf_path() -> Path:
    """Return the path for the single combined chart PDF."""
    chart_files = pipeline_chart_paths(ticker)
    if chart_files:
        return chart_files[0].parent / f"{ticker}_pipeline_charts.pdf"
    return Path("Charts") / f"{ticker}_pipeline_charts.pdf"


def build_combined_pdf(chart_files: list[Path]) -> Path:
    """Create one multi-page PDF containing every saved chart."""
    pdf_path = combined_chart_pdf_path()

    image_pages = []
    for chart_path in chart_files:
        with Image.open(chart_path) as image:
            image_pages.append(image.convert("RGB"))

    first_page, *remaining_pages = image_pages
    first_page.save(pdf_path, save_all=True, append_images=remaining_pages)
    return pdf_path


def main() -> None:
    """Open one combined PDF containing each saved pipeline chart."""
    chart_files = find_chart_files()
    pdf_path = build_combined_pdf(chart_files)

    if not interactive_open:
        return

    preview_result = subprocess.run(
        ["open", "-a", "Preview", str(pdf_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    if preview_result.returncode == 0:
        return

    default_result = subprocess.run(
        ["open", str(pdf_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    if default_result.returncode != 0:
        print("Warning: The combined chart PDF could not be opened automatically.")
        print(f"- {pdf_path}")


if __name__ == "__main__":
    main()
