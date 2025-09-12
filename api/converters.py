import os
import subprocess
import tempfile
from pathlib import Path

OFFICE_EXTS = {".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"}


def is_office_file(path: str) -> bool:
    return Path(path).suffix.lower() in OFFICE_EXTS


def to_pdf(input_path: str, outdir: str) -> str:
    """Convert an Office document to PDF using LibreOffice headless.

    Returns the output PDF path.
    """
    os.makedirs(outdir, exist_ok=True)
    cmd = [
        "soffice",
        "--headless",
        "--norestore",
        "--nolockcheck",
        "--nodefault",
        "--nofirststartwizard",
        "--convert-to",
        "pdf",
        "--outdir",
        outdir,
        input_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"LibreOffice conversion failed: {e.stderr.decode(errors='ignore')}"
        )

    stem = Path(input_path).stem
    out_pdf = Path(outdir) / f"{stem}.pdf"
    if not out_pdf.exists():
        # LibreOffice sometimes outputs with uppercase extension or altered name; fallback search
        candidates = list(Path(outdir).glob("*.pdf"))
        if not candidates:
            raise FileNotFoundError("Converted PDF not found after LibreOffice run")
        # Choose the most recent
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        out_pdf = candidates[0]
    return str(out_pdf)
