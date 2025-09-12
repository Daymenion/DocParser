from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional
import os
import io
import tempfile
import shutil
from pathlib import Path
import requests

from dots_ocr.parser import DotsOCRParser
from dots_ocr.utils.doc_utils import load_images_from_pdf
from PIL import Image
from .converters import is_office_file, to_pdf
import zipfile

app = FastAPI(title="az doc-parser API", version="1.0")

# Singleton parser in HF mode
MODEL_PATH = os.environ.get("MODEL_PATH", "./weights/DotsOCR")
parser = DotsOCRParser(use_hf=True, dpi=200, model_path=MODEL_PATH)


@app.get("/health")
def health():
    info = {
        "backend": "hf" if parser.use_hf else "vllm",
        "dpi": parser.dpi,
        "min_pixels": parser.min_pixels,
        "max_pixels": parser.max_pixels,
    }
    return info


def _save_upload_to_tmp(upload: UploadFile) -> str:
    suffix = Path(upload.filename).suffix or ""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return path


@app.post("/parse")
def parse(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    prompt_mode: str = Form("prompt_layout_all_en"),
    fitz_preprocess: bool = Form(True),
    output: str = Form("md"),
    inline_md: bool = Form(False),
):
    if output not in {"md", "zip", "md+zip"}:
        raise HTTPException(status_code=400, detail="Invalid output param")

    tmp_input = None
    tmp_dir = None
    try:
        if not file and not url:
            raise HTTPException(status_code=400, detail="Provide either file or url")
        if file and url:
            raise HTTPException(status_code=400, detail="Provide only one of file or url")

        if file:
            tmp_input = _save_upload_to_tmp(file)
            name = Path(file.filename).stem
        else:
            # Download from URL with basic size guard
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            suffix = Path(url).suffix or ""
            fd, path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            with open(path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            tmp_input = path
            name = Path(url).stem or "document"
        tmp_dir = tempfile.mkdtemp(prefix="dotsocr_api_")

        ext = Path(tmp_input).suffix.lower()
        # Office conversion
        if is_office_file(tmp_input):
            tmp_pdf = to_pdf(tmp_input, tmp_dir)
            tmp_input = tmp_pdf
            ext = ".pdf"

        if ext == ".pdf":
            results = parser.parse_pdf(tmp_input, name, prompt_mode, tmp_dir)
            combined_md = []
            for res in results:
                md_path = res.get("md_content_path")
                if md_path and os.path.exists(md_path):
                    with open(md_path, "r", encoding="utf-8") as f:
                        combined_md.append(f.read())
            md = "\n\n---\n\n".join(combined_md) if combined_md else ""
            # persist combined md for zip
            combined_md_path = os.path.join(tmp_dir, f"{name}_combined.md")
            with open(combined_md_path, "w", encoding="utf-8") as f:
                f.write(md)
        else:
            # Treat as image
            image = Image.open(tmp_input)
            results = parser.parse_image(tmp_input, name, prompt_mode, tmp_dir, fitz_preprocess=fitz_preprocess)
            res = results[0]
            md_path = res.get("md_content_path")
            md = ""
            if md_path and os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8") as f:
                    md = f.read()
            # also write combined for single image for consistency
            combined_md_path = os.path.join(tmp_dir, f"{name}_combined.md")
            with open(combined_md_path, "w", encoding="utf-8") as f:
                f.write(md)

        # Return Markdown
        if output in {"md", "md+zip"}:
            if inline_md:
                # Small docs can be returned inline as JSON
                return JSONResponse({"filename": f"{name}.md", "md_content": md})
            else:
                md_bytes = md.encode("utf-8")
                headers = {"Content-Disposition": f"attachment; filename=\"{name}.md\""}
                return StreamingResponse(io.BytesIO(md_bytes), media_type="text/markdown; charset=utf-8", headers=headers)
        else:
            # Build zip of all artifacts within tmp_dir
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(tmp_dir):
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        # avoid nested zips
                        if not fname.lower().endswith('.zip'):
                            arcname = os.path.relpath(fpath, tmp_dir)
                            zipf.write(fpath, arcname)
                # include a canonical combined.md name at top-level
                # already present as {name}_combined.md; duplicate under combined.md for convenience
                combined_src = os.path.join(tmp_dir, f"{name}_combined.md")
                if os.path.exists(combined_src):
                    with open(combined_src, 'rb') as cf:
                        zipf.writestr('combined.md', cf.read())
            zip_buf.seek(0)
            headers = {"Content-Disposition": f"attachment; filename=\"{name}_artifacts.zip\""}
            return StreamingResponse(zip_buf, media_type="application/zip", headers=headers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file and not file.file.closed:
            file.file.close()
        # Do not delete tmp_dir immediately if we returned streaming content from it; here we only used md text
        if tmp_input and os.path.exists(tmp_input):
            try:
                os.remove(tmp_input)
            except Exception:
                pass
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
