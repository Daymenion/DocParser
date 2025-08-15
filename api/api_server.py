"""
FastAPI Server for Document Parsing API

This server provides REST API endpoints for parsing PDF and image documents.
It extracts and reuses the core business logic from demo_gradio.py while providing
a clean API interface.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import shutil
import os
import json
import uuid
import zipfile
from pathlib import Path
from typing import Optional, List, Dict, Any
from PIL import Image
import uvicorn

# Import core components from the existing codebase
from dots_ocr.parser import DotsOCRParser
from dots_ocr.utils import dict_promptmode_to_prompt
from dots_ocr.utils.consts import MIN_PIXELS, MAX_PIXELS

# API Configuration
app = FastAPI(
    title="DocParser Document Parsing API",
    description="REST API for parsing PDF and image documents using DotsOCR engine",
    version="1.0.0"
)

# CORS middleware for web client compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global configuration
DEFAULT_CONFIG = {
    'ip': "127.0.0.1",
    'port_vllm': 9998,
    'min_pixels': MIN_PIXELS,
    'max_pixels': MAX_PIXELS,
}

# Initialize the parser
dots_parser = DotsOCRParser(
    ip=DEFAULT_CONFIG['ip'],
    port=DEFAULT_CONFIG['port_vllm'],
    dpi=200,
    min_pixels=DEFAULT_CONFIG['min_pixels'],
    max_pixels=DEFAULT_CONFIG['max_pixels']
)

class DocumentParsingService:
    """
    Service class that encapsulates the document parsing business logic.
    This extracts the core functionality from demo_gradio.py.
    """
    
    @staticmethod
    def create_temp_session_dir():
        """Creates a unique temporary directory for processing"""
        session_id = uuid.uuid4().hex[:8]
        temp_dir = os.path.join(tempfile.gettempdir(), f"dots_ocr_api_{session_id}")
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir, session_id

    @staticmethod
    def parse_image_document(parser, image_path: str, prompt_mode: str, fitz_preprocess: bool = False):
        """Parse a single image document"""
        temp_dir, session_id = DocumentParsingService.create_temp_session_dir()
        
        try:
            # Load image
            image = Image.open(image_path)
            
            # Use the high-level API parse_image
            filename = f"api_{session_id}"
            results = parser.parse_image(
                input_path=image,
                filename=filename,
                prompt_mode=prompt_mode,
                save_dir=temp_dir,
                fitz_preprocess=fitz_preprocess
            )
            
            if not results:
                raise ValueError("No results returned from parser")
            
            result = results[0]
            
            # Process results
            response_data = {
                'session_id': session_id,
                'document_type': 'image',
                'filtered': result.get('filtered', False),
                'input_dimensions': {
                    'width': result.get('input_width', 0),
                    'height': result.get('input_height', 0)
                },
                'original_dimensions': {
                    'width': image.width,
                    'height': image.height
                }
            }
            
            # Read structured data
            if 'layout_info_path' in result and os.path.exists(result['layout_info_path']):
                with open(result['layout_info_path'], 'r', encoding='utf-8') as f:
                    response_data['layout_data'] = json.load(f)
            
            # Read markdown content
            if 'md_content_path' in result and os.path.exists(result['md_content_path']):
                with open(result['md_content_path'], 'r', encoding='utf-8') as f:
                    response_data['markdown_content'] = f.read()
            
            # Create ZIP file for download
            zip_path = os.path.join(temp_dir, f"results_{session_id}.zip")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        if not file.endswith('.zip'):
                            file_path = os.path.join(root, file)
                            zipf.write(file_path, os.path.relpath(file_path, temp_dir))
            
            response_data['download_url'] = f"/api/download/{session_id}"
            response_data['temp_dir'] = temp_dir
            
            return response_data
            
        except Exception as e:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise e

    @staticmethod
    def parse_pdf_document(parser, pdf_path: str, prompt_mode: str):
        """Parse a PDF document"""
        temp_dir, session_id = DocumentParsingService.create_temp_session_dir()
        
        try:
            # Use the high-level API parse_pdf
            filename = f"api_{session_id}"
            results = parser.parse_pdf(
                input_path=pdf_path,
                filename=filename,
                prompt_mode=prompt_mode,
                save_dir=temp_dir
            )
            
            if not results:
                raise ValueError("No results returned from parser")
            
            # Process multi-page results
            all_layout_data = []
            all_md_content = []
            page_results = []
            
            for i, result in enumerate(results):
                page_data = {
                    'page_no': result.get('page_no', i),
                    'filtered': result.get('filtered', False)
                }
                
                # Read page layout data
                if 'layout_info_path' in result and os.path.exists(result['layout_info_path']):
                    with open(result['layout_info_path'], 'r', encoding='utf-8') as f:
                        page_layout = json.load(f)
                        page_data['layout_data'] = page_layout
                        all_layout_data.extend(page_layout)
                
                # Read page markdown content
                if 'md_content_path' in result and os.path.exists(result['md_content_path']):
                    with open(result['md_content_path'], 'r', encoding='utf-8') as f:
                        page_content = f.read()
                        page_data['markdown_content'] = page_content
                        all_md_content.append(page_content)
                
                page_results.append(page_data)
            
            # Create response data
            response_data = {
                'session_id': session_id,
                'document_type': 'pdf',
                'total_pages': len(results),
                'pages': page_results,
                'combined_layout_data': all_layout_data,
                'combined_markdown_content': "\n\n---\n\n".join(all_md_content) if all_md_content else "",
                'total_elements': len(all_layout_data)
            }
            
            # Create ZIP file for download
            zip_path = os.path.join(temp_dir, f"results_{session_id}.zip")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        if not file.endswith('.zip'):
                            file_path = os.path.join(root, file)
                            zipf.write(file_path, os.path.relpath(file_path, temp_dir))
            
            response_data['download_url'] = f"/api/download/{session_id}"
            response_data['temp_dir'] = temp_dir
            
            return response_data
            
        except Exception as e:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise e

# Store session data for download endpoints
session_storage = {}

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "DotsOCR Document Parsing API",
        "version": "1.0.0",
        "docs": "/docs",
        "available_endpoints": [
            "/api/parse/image",
            "/api/parse/pdf", 
            "/api/config",
            "/api/health"
        ]
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "dotsocr-api"}

@app.get("/api/config")
async def get_config():
    """Get current API configuration"""
    return {
        "vllm_server": f"{dots_parser.ip}:{dots_parser.port}",
        "min_pixels": dots_parser.min_pixels,
        "max_pixels": dots_parser.max_pixels,
        "available_prompt_modes": list(dict_promptmode_to_prompt.keys())
    }

@app.post("/api/parse/image")
async def parse_image(
    file: UploadFile = File(...),
    prompt_mode: str = Form(default="prompt_layout_all_en"),
    fitz_preprocess: bool = Form(default=False),
    server_ip: Optional[str] = Form(default=None),
    server_port: Optional[int] = Form(default=None),
    min_pixels: Optional[int] = Form(default=None),
    max_pixels: Optional[int] = Form(default=None)
):
    """
    Parse an image document
    
    - **file**: Image file to parse (PNG, JPG, JPEG)
    - **prompt_mode**: Parsing mode (prompt_layout_all_en, prompt_layout_only_en, prompt_ocr)
    - **fitz_preprocess**: Enable fitz preprocessing for low DPI images
    - **server_ip**: Override vLLM server IP
    - **server_port**: Override vLLM server port
    - **min_pixels**: Override minimum pixels
    - **max_pixels**: Override maximum pixels
    """
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate prompt mode
    if prompt_mode not in dict_promptmode_to_prompt:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid prompt_mode. Available: {list(dict_promptmode_to_prompt.keys())}"
        )
    
    # Update parser configuration if provided
    if server_ip:
        dots_parser.ip = server_ip
    if server_port:
        dots_parser.port = server_port
    if min_pixels:
        dots_parser.min_pixels = min_pixels
    if max_pixels:
        dots_parser.max_pixels = max_pixels
    
    # Save uploaded file
    temp_input_file = None
    try:
        # Create temporary file for input
        temp_input_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}")
        content = await file.read()
        temp_input_file.write(content)
        temp_input_file.close()
        
        # Parse the document
        result = DocumentParsingService.parse_image_document(
            dots_parser, temp_input_file.name, prompt_mode, fitz_preprocess
        )
        
        # Store session for download
        session_storage[result['session_id']] = result['temp_dir']
        
        # Remove temp_dir from response (internal use only)
        result.pop('temp_dir', None)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Clean up input file
        if temp_input_file and os.path.exists(temp_input_file.name):
            os.unlink(temp_input_file.name)

@app.post("/api/parse/pdf")
async def parse_pdf(
    file: UploadFile = File(...),
    prompt_mode: str = Form(default="prompt_layout_all_en"),
    server_ip: Optional[str] = Form(default=None),
    server_port: Optional[int] = Form(default=None),
    min_pixels: Optional[int] = Form(default=None),
    max_pixels: Optional[int] = Form(default=None)
):
    """
    Parse a PDF document
    
    - **file**: PDF file to parse
    - **prompt_mode**: Parsing mode (prompt_layout_all_en, prompt_layout_only_en, prompt_ocr)
    - **server_ip**: Override vLLM server IP
    - **server_port**: Override vLLM server port
    - **min_pixels**: Override minimum pixels
    - **max_pixels**: Override maximum pixels
    """
    
    # Validate file type
    if not file.content_type == 'application/pdf':
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Validate prompt mode
    if prompt_mode not in dict_promptmode_to_prompt:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid prompt_mode. Available: {list(dict_promptmode_to_prompt.keys())}"
        )
    
    # Update parser configuration if provided
    if server_ip:
        dots_parser.ip = server_ip
    if server_port:
        dots_parser.port = server_port
    if min_pixels:
        dots_parser.min_pixels = min_pixels
    if max_pixels:
        dots_parser.max_pixels = max_pixels
    
    # Save uploaded file
    temp_input_file = None
    try:
        # Create temporary file for input
        temp_input_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        content = await file.read()
        temp_input_file.write(content)
        temp_input_file.close()
        
        # Parse the document
        result = DocumentParsingService.parse_pdf_document(
            dots_parser, temp_input_file.name, prompt_mode
        )
        
        # Store session for download
        session_storage[result['session_id']] = result['temp_dir']
        
        # Remove temp_dir from response (internal use only)
        result.pop('temp_dir', None)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Clean up input file
        if temp_input_file and os.path.exists(temp_input_file.name):
            os.unlink(temp_input_file.name)

@app.get("/api/download/{session_id}")
async def download_results(session_id: str):
    """
    Download the complete parsing results as a ZIP file
    
    - **session_id**: Session ID from parsing response
    """
    if session_id not in session_storage:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    temp_dir = session_storage[session_id]
    zip_path = os.path.join(temp_dir, f"results_{session_id}.zip")
    
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="Results file not found")
    
    return FileResponse(
        zip_path,
        media_type='application/zip',
        filename=f"docparser_results_{session_id}.zip"
    )

@app.delete("/api/session/{session_id}")
async def cleanup_session(session_id: str):
    """
    Clean up session data and temporary files
    
    - **session_id**: Session ID to clean up
    """
    if session_id not in session_storage:
        raise HTTPException(status_code=404, detail="Session not found")
    
    temp_dir = session_storage[session_id]
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        del session_storage[session_id]
        return {"message": f"Session {session_id} cleaned up successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

if __name__ == "__main__":
    import sys
    
    # Default configuration
    host = "0.0.0.0"
    port = 7860
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}. Using default port {port}")
    
    print(f"Starting DocParser API Server on {host}:{port}")
    print(f"API Documentation available at: http://{host}:{port}/docs")
    print(f"vLLM Server: {dots_parser.ip}:{dots_parser.port}")
    
    uvicorn.run(app, host=host, port=port)
