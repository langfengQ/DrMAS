"""Regenerate website figures from the supplied paper (requires pymupdf)."""

import shutil
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "public" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

with pymupdf.open(ROOT / "Dr__MAS_final.pdf") as document:
    # PDF coordinates are in points; exclude captions and surrounding body text.
    for name, page, bounds in [
        ("overview", 0, (108, 518, 503, 665)),
        ("framework", 5, (106, 69, 506, 167)),
        ("training-dynamics", 7, (106, 320, 506, 394)),
    ]:
        document[page].get_pixmap(
            matrix=pymupdf.Matrix(5, 5),
            clip=pymupdf.Rect(bounds),
            alpha=False,
        ).save(FIGURES / f"{name}.png")

shutil.copyfile(ROOT / "Dr__MAS_final.pdf", ROOT / "public" / "paper.pdf")
print("Exported overview, framework, training dynamics, and paper PDF.")
