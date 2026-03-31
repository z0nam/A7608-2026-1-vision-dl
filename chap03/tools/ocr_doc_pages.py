from __future__ import annotations

import subprocess
from pathlib import Path

from doc_image_utils import detect_rotation, rotate_copy, select_ocr_language

ROOT = Path(__file__).resolve().parents[1]
DOC_DIR = ROOT / "book" / "processed"
OCR_DIR = ROOT / "book" / "ocr"
PAGE_OVERRIDES: dict[str, dict[str, int]] = {
    "KakaoTalk_20260329_001527225_03": {"rotate": 180, "psm": 4},
    "KakaoTalk_20260329_001527225_07": {"rotate": 180, "psm": 4},
    "KakaoTalk_20260329_001527225_09": {"rotate": 180, "psm": 4},
    "KakaoTalk_20260329_001527225_14": {"rotate": 180, "psm": 4},
    "KakaoTalk_20260329_001527225_16": {"rotate": 180, "psm": 4},
    "KakaoTalk_20260329_001527225_17": {"rotate": 180, "psm": 4},
    "KakaoTalk_20260329_001527225_20": {"rotate": 180, "psm": 6},
}


def ensure_dirs() -> None:
    OCR_DIR.mkdir(parents=True, exist_ok=True)


def run_tesseract(image_path: Path, out_base: Path, language: str, psm: int) -> None:
    subprocess.run(
        [
            "tesseract",
            str(image_path),
            str(out_base),
            "-l",
            language,
            "--psm",
            str(psm),
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
    language = select_ocr_language()
    entries: list[tuple[str, str]] = []
    for src in sorted(DOC_DIR.glob("*.jpg")):
        page_id = src.stem
        text_base = OCR_DIR / page_id
        override = PAGE_OVERRIDES.get(page_id, {})
        rotation = override.get("rotate", detect_rotation(src))
        psm = override.get("psm", 6)
        ocr_src = src
        rotated_copy: Path | None = None
        if rotation:
            rotated_copy = rotate_copy(src, rotation)
            ocr_src = rotated_copy

        try:
            run_tesseract(ocr_src, text_base, language, psm)
        finally:
            if rotated_copy is not None and rotated_copy.exists():
                rotated_copy.unlink()

        entries.append((src.name, f"{page_id}.txt"))

    build_manifest(entries)
    print(f"Processed {len(entries)} pages.")


if __name__ == "__main__":
    main()
