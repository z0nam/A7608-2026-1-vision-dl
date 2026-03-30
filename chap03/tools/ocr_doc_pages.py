from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC_DIR = ROOT / "book" / "processed"
OCR_DIR = ROOT / "book" / "ocr"


def ensure_dirs() -> None:
    OCR_DIR.mkdir(parents=True, exist_ok=True)


def run_tesseract(image_path: Path, out_base: Path) -> None:
    subprocess.run(
        [
            "tesseract",
            str(image_path),
            str(out_base),
            "-l",
            "eng+kor",
            "--psm",
            "6",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def build_manifest(entries: list[tuple[str, str]]) -> None:
    manifest = ROOT / "book" / "page_index.md"
    lines = [
        "# Page Index",
        "",
        "| Page | OCR Text |",
        "| --- | --- |",
    ]
    for page_name, txt_name in entries:
        lines.append(f"| `{page_name}` | `{txt_name}` |")
    lines.append("")
    manifest.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    entries: list[tuple[str, str]] = []
    for src in sorted(DOC_DIR.glob("*.jpg")):
        page_id = src.stem
        text_base = OCR_DIR / page_id

        run_tesseract(src, text_base)
        entries.append((src.name, f"{page_id}.txt"))

    build_manifest(entries)
    print(f"Processed {len(entries)} pages.")


if __name__ == "__main__":
    main()
