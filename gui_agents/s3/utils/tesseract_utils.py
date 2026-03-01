from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Optional

import pytesseract


def _candidate_tesseract_paths() -> list[Path]:
    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    program_files_x86 = Path(
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    )
    local_app_data = Path(os.environ.get("LOCALAPPDATA", ""))

    return [
        program_files / "Tesseract-OCR" / "tesseract.exe",
        program_files_x86 / "Tesseract-OCR" / "tesseract.exe",
        local_app_data / "Programs" / "Tesseract-OCR" / "tesseract.exe",
    ]


def configure_pytesseract() -> Optional[str]:
    detected = shutil.which("tesseract")
    if detected:
        pytesseract.pytesseract.tesseract_cmd = detected
        return detected

    configured = getattr(pytesseract.pytesseract, "tesseract_cmd", None)
    if configured and Path(configured).exists():
        return configured

    for candidate in _candidate_tesseract_paths():
        if candidate.exists():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            return str(candidate)

    return None


__all__ = ["configure_pytesseract"]
