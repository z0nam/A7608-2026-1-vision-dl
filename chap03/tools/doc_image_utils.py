from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


def available_tesseract_languages() -> set[str]:
    result = subprocess.run(
        ["tesseract", "--list-langs"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and not line.startswith("List of available languages")
    }


def select_ocr_language() -> str:
    langs = available_tesseract_languages()
    if {"eng", "kor"}.issubset(langs):
        return "eng+kor"
    if "eng" in langs:
        return "eng"
    raise RuntimeError("tesseract 'eng' language data is not available.")


def detect_rotation(image_path: Path) -> int:
    result = subprocess.run(
        ["tesseract", str(image_path), "stdout", "--psm", "0"],
        capture_output=True,
        text=True,
        errors="replace",
    )
    if result.returncode != 0:
        return 0

    for line in result.stdout.splitlines():
        if line.startswith("Rotate:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return 0
    return 0


def rotate_copy(image_path: Path, degrees: int) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=image_path.suffix, delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

    subprocess.run(
        ["sips", "-r", str(degrees), str(image_path), "--out", str(tmp_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return tmp_path


def normalize_image_in_place(image_path: Path, degrees: int) -> None:
    rotated = rotate_copy(image_path, degrees)
    rotated.replace(image_path)
