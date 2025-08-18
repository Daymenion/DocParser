import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dots_ocr.parser import DotsOCRParser

# Optional: mount existing Gradio UI onto FastAPI
try:
    import gradio as gr
    from demo.demo_gradio import create_gradio_interface
    HAS_GRADIO = True
except Exception:
    HAS_GRADIO = False


APP_PORT = int(os.getenv("APP_PORT", "7860"))
VLLM_HOST = os.getenv("VLLM_HOST", "127.0.0.1")
VLLM_PORT = int(os.getenv("VLLM_PORT", "9998"))
MIN_PIXELS = os.getenv("MIN_PIXELS")
MAX_PIXELS = os.getenv("MAX_PIXELS")


def _to_int_or_none(val: Optional[str]) -> Optional[int]:
    if val is None or val == "":
        return None
    try:
        return int(val)
    except Exception:
        return None


default_min_pixels = _to_int_or_none(MIN_PIXELS)
default_max_pixels = _to_int_or_none(MAX_PIXELS)


app = FastAPI(title="dots.ocr API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _make_parser(min_pixels: Optional[int], max_pixels: Optional[int]) -> DotsOCRParser:
    return DotsOCRParser(
        ip=VLLM_HOST,
        port=VLLM_PORT,
        dpi=200,
        min_pixels=min_pixels if min_pixels is not None else default_min_pixels,
        max_pixels=max_pixels if max_pixels is not None else default_max_pixels,
    )


@app.get("/health")
def health():
    return {"status": "ok", "vllm": {"host": VLLM_HOST, "port": VLLM_PORT}}


@app.post("/api/parse")
async def parse_document(
    file: UploadFile = File(...),
    prompt_mode: str = Form("prompt_layout_all_en"),
    fitz_preprocess: bool = Form(True),
    return_markdown: bool = Form(True),
    min_pixels: Optional[int] = Form(None),
    max_pixels: Optional[int] = Form(None),
):
    """
    Accepts a PDF or image and returns structured data.

    - If PDF: returns combined cells_data, combined markdown, and per-page metadata.
    - If Image: returns cells_data and markdown for the image.
    """
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in [".pdf", ".png", ".jpg", ".jpeg"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload PDF or PNG/JPG images.")

    # Persist upload to a temp file for processing
    tmp_dir = Path(tempfile.mkdtemp(prefix="dotsocr_api_"))
    tmp_path = tmp_dir / ("input" + suffix)
    try:
        data = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(data)

        parser = _make_parser(min_pixels, max_pixels)

        if suffix == ".pdf":
            filename = tmp_path.stem
            results = parser.parse_pdf(str(tmp_path), filename, prompt_mode, str(tmp_dir))

            combined_cells = []
            pages = []
            md_parts = []
            for i, r in enumerate(results):
                page_cells = None
                if r.get("layout_info_path") and os.path.exists(r["layout_info_path"]):
                    try:
                        with open(r["layout_info_path"], "r", encoding="utf-8") as jf:
                            page_cells = json.load(jf)
                            if isinstance(page_cells, list):
                                combined_cells.extend(page_cells)
                            else:
                                combined_cells.append(page_cells)
                    except Exception:
                        page_cells = None

                page_md = None
                if return_markdown and r.get("md_content_path") and os.path.exists(r["md_content_path"]):
                    try:
                        with open(r["md_content_path"], "r", encoding="utf-8") as mf:
                            page_md = mf.read()
                            md_parts.append(page_md)
                    except Exception:
                        page_md = None

                pages.append({
                    "page_no": r.get("page_no", i),
                    "input_height": r.get("input_height"),
                    "input_width": r.get("input_width"),
                    "filtered": r.get("filtered", False),
                    "cells_data": page_cells,
                })

            payload = {
                "type": "pdf",
                "total_pages": len(results),
                "cells_data": combined_cells,
                "markdown": ("\n\n---\n\n".join(md_parts) if return_markdown else None),
                "pages": pages,
                "vllm": {"host": VLLM_HOST, "port": VLLM_PORT},
            }
            return JSONResponse(content=payload)

        else:
            # Image processing path
            filename = tmp_path.stem
            results = parser.parse_image(str(tmp_path), filename, prompt_mode, str(tmp_dir), fitz_preprocess=fitz_preprocess)
            if not results:
                raise HTTPException(status_code=500, detail="No results returned from parser")

            r = results[0]
            cells_data = None
            if r.get("layout_info_path") and os.path.exists(r["layout_info_path"]):
                try:
                    with open(r["layout_info_path"], "r", encoding="utf-8") as jf:
                        cells_data = json.load(jf)
                except Exception:
                    cells_data = None

            md_content = None
            if return_markdown and r.get("md_content_path") and os.path.exists(r["md_content_path"]):
                try:
                    with open(r["md_content_path"], "r", encoding="utf-8") as mf:
                        md_content = mf.read()
                except Exception:
                    md_content = None

            payload = {
                "type": "image",
                "input_height": r.get("input_height"),
                "input_width": r.get("input_width"),
                "filtered": r.get("filtered", False),
                "cells_data": cells_data,
                "markdown": md_content if return_markdown else None,
                "vllm": {"host": VLLM_HOST, "port": VLLM_PORT},
            }
            return JSONResponse(content=payload)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")
    finally:
        # Clean temp directory
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


# Mount Gradio UI at root path if available
if HAS_GRADIO:
    demo: gr.Blocks = create_gradio_interface()
    # Mount at "/" so both UI and API share the same port
    app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=APP_PORT, reload=False)
