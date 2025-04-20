#!/usr/bin/env python3

import fitz
import tabula
import pandas as pd
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────
PDF_PATH = "ltimindtree_annual_report.pdf"  # change this when needed
OUTPUT_DIR = Path("extracted")
TEXT_FILE = OUTPUT_DIR / "all_text.txt"
TABLES_DIR = OUTPUT_DIR / "tables"
# ──────────────────────────────────────────────────────────────────────────────

def extract_text():
    doc = fitz.open(PDF_PATH)
    OUTPUT_DIR.mkdir(exist_ok=True)
    TEXT_FILE.touch(exist_ok=True)

    with open(TEXT_FILE, "a", encoding="utf-8") as f:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            f.write(f"\n\n--- FILE: {Path(PDF_PATH).name} | PAGE {page_num} ---\n\n")
            f.write(text or "")
    print(f"Appended text from {PDF_PATH} to {TEXT_FILE}")

def extract_tables():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    all_dfs = []

    print("Extracting tables with lattice mode...")
    dfs_lattice = tabula.read_pdf(
        PDF_PATH,
        pages="all",
        multiple_tables=True,
        lattice=True,
        guess=False
    )
    all_dfs.extend(dfs_lattice)

    print("Extracting tables with stream mode...")
    dfs_stream = tabula.read_pdf(
        PDF_PATH,
        pages="all",
        multiple_tables=True,
        stream=True,
        guess=False
    )
    all_dfs.extend(dfs_stream)

    seen = set()
    count = 0
    pdf_stem = Path(PDF_PATH).stem
    for df in all_dfs:
        if df.dropna(how="all").empty:
            continue
        signature = tuple(
            df.fillna("").astype(str)
              .apply(lambda row: "|".join(row), axis=1)
        )
        if signature in seen:
            continue
        seen.add(signature)
        count += 1
        out_path = TABLES_DIR / f"{pdf_stem}_table_{count}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved table #{count} to {out_path}")

    if count == 0:
        print("No tables found.")
    else:
        print(f"Extracted {count} tables to {TABLES_DIR}/")

if __name__ == "__main__":
    if not Path(PDF_PATH).is_file():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")
    print(f"Processing {PDF_PATH}")
    extract_text()
    extract_tables()
    print("Done.")