# DotsOCR API Server

A FastAPI-based REST API server for document parsing using DotsOCR.

## Features

- **Image Parsing**: Upload and parse PNG, JPG, JPEG images
- **PDF Parsing**: Upload and parse multi-page PDF documents  
- **Multiple Prompt Modes**: Support for different parsing strategies
- **Structured Output**: JSON responses with layout data and markdown content
- **File Downloads**: Download complete parsing results as ZIP files
- **Configuration**: Runtime configuration of parsing parameters

## Setup

### Prerequisites

1. **vLLM Server Running**: The API server requires a vLLM server with the DotsOCR model
2. **Python Environment**: Python 3.8+ with required dependencies

### Installation

```bash
# Install API server dependencies
pip install -r api/requirements.txt

# Or install individual packages
pip install fastapi uvicorn[standard] python-multipart
```

### Start the API Server

```bash
# Start with default port (7860)
python api/api_server.py

# Start with custom port
python api/api_server.py 7865

# Or use uvicorn directly
uvicorn api.api_server:app --host 0.0.0.0 --port 7860
```

## API Endpoints

### Base URLs
- **API Server**: `http://localhost:7860`
- **Documentation**: `http://localhost:7860/docs`
- **OpenAPI Spec**: `http://localhost:7860/openapi.json`

### Core Endpoints

#### 1. Health Check
```bash
GET /api/health
```

#### 2. Get Configuration
```bash
GET /api/config
```

#### 3. Parse Image
```bash
POST /api/parse/image
```

**Parameters:**
- `file` (required): Image file (PNG, JPG, JPEG)
- `prompt_mode` (optional): Parsing mode (default: "prompt_layout_all_en")
- `fitz_preprocess` (optional): Enable fitz preprocessing (default: false)
- `server_ip` (optional): Override vLLM server IP
- `server_port` (optional): Override vLLM server port

#### 4. Parse PDF
```bash
POST /api/parse/pdf
```

**Parameters:**
- `file` (required): PDF file
- `prompt_mode` (optional): Parsing mode (default: "prompt_layout_all_en")
- `server_ip` (optional): Override vLLM server IP
- `server_port` (optional): Override vLLM server port

#### 5. Download Results
```bash
GET /api/download/{session_id}
```

#### 6. Cleanup Session
```bash
DELETE /api/session/{session_id}
```

## Usage Examples

### Python Client Example

```python
import requests
import json

# API base URL
API_BASE = "http://localhost:7860"

# Parse an image
def parse_image(image_path, prompt_mode="prompt_layout_all_en"):
    with open(image_path, "rb") as f:
        files = {"file": f}
        data = {"prompt_mode": prompt_mode}
        
        response = requests.post(f"{API_BASE}/api/parse/image", files=files, data=data)
        
    if response.status_code == 200:
        result = response.json()
        print(f"Session ID: {result['session_id']}")
        print(f"Layout elements found: {len(result.get('layout_data', []))}")
        return result
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Parse a PDF
def parse_pdf(pdf_path, prompt_mode="prompt_layout_all_en"):
    with open(pdf_path, "rb") as f:
        files = {"file": f}
        data = {"prompt_mode": prompt_mode}
        
        response = requests.post(f"{API_BASE}/api/parse/pdf", files=files, data=data)
        
    if response.status_code == 200:
        result = response.json()
        print(f"Session ID: {result['session_id']}")
        print(f"Total pages: {result['total_pages']}")
        print(f"Total elements: {result['total_elements']}")
        return result
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Download results
def download_results(session_id, output_path):
    response = requests.get(f"{API_BASE}/api/download/{session_id}")
    
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Results downloaded to: {output_path}")
    else:
        print(f"Download failed: {response.status_code}")

# Example usage
if __name__ == "__main__":
    # Parse an image
    result = parse_image("demo/demo_image1.jpg")
    if result:
        # Download the complete results
        download_results(result['session_id'], f"results_{result['session_id']}.zip")

    # Parse a PDF
    pdf_result = parse_pdf("demo/demo_pdf1.pdf")
    if pdf_result:
        # Access structured data
        print("Combined markdown content:")
        print(pdf_result['combined_markdown_content'][:500] + "...")
```

### cURL Examples

```bash
# Health check
curl http://localhost:7860/api/health

# Get configuration
curl http://localhost:7860/api/config

# Parse an image
curl -X POST "http://localhost:7860/api/parse/image" \
  -F "file=@demo/demo_image1.jpg" \
  -F "prompt_mode=prompt_layout_all_en"

# Parse a PDF
curl -X POST "http://localhost:7860/api/parse/pdf" \
  -F "file=@demo/demo_pdf1.pdf" \
  -F "prompt_mode=prompt_layout_all_en"

# Download results (replace SESSION_ID with actual session ID)
curl -O -J "http://localhost:7860/api/download/SESSION_ID"
```

### JavaScript/Node.js Example

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const API_BASE = 'http://localhost:7860';

async function parseImage(imagePath, promptMode = 'prompt_layout_all_en') {
    const form = new FormData();
    form.append('file', fs.createReadStream(imagePath));
    form.append('prompt_mode', promptMode);
    
    try {
        const response = await axios.post(`${API_BASE}/api/parse/image`, form, {
            headers: form.getHeaders()
        });
        
        console.log('Session ID:', response.data.session_id);
        console.log('Layout elements:', response.data.layout_data?.length || 0);
        return response.data;
    } catch (error) {
        console.error('Parse failed:', error.response?.data || error.message);
        return null;
    }
}

// Usage
parseImage('demo/demo_image1.jpg').then(result => {
    if (result) {
        console.log('Parsing successful!');
        // Process the result...
    }
});
```

## Response Format

### Image Parsing Response
```json
{
    "session_id": "abc12345",
    "document_type": "image",
    "filtered": false,
    "input_dimensions": {
        "width": 1024,
        "height": 768
    },
    "original_dimensions": {
        "width": 2048,
        "height": 1536
    },
    "layout_data": [
        {
            "category": "Title",
            "bbox": [100, 50, 400, 100],
            "text": "Document Title"
        },
        {
            "category": "Text",
            "bbox": [100, 120, 400, 200],
            "text": "Document content..."
        }
    ],
    "markdown_content": "# Document Title\n\nDocument content...",
    "download_url": "/api/download/abc12345"
}
```

### PDF Parsing Response
```json
{
    "session_id": "def67890",
    "document_type": "pdf",
    "total_pages": 3,
    "total_elements": 25,
    "pages": [
        {
            "page_no": 0,
            "filtered": false,
            "layout_data": [...],
            "markdown_content": "..."
        }
    ],
    "combined_layout_data": [...],
    "combined_markdown_content": "...",
    "download_url": "/api/download/def67890"
}
```

## Prompt Modes

Available prompt modes for different parsing strategies:

- **`prompt_layout_all_en`**: Full layout detection and content recognition (default)
- **`prompt_layout_only_en`**: Layout detection only, no text content
- **`prompt_ocr`**: Text extraction only, excluding headers/footers

## Configuration

### Environment Variables
- `API_KEY`: vLLM API key (if required)

### Runtime Configuration
You can override vLLM server settings per request:
- `server_ip`: vLLM server IP address
- `server_port`: vLLM server port
- `min_pixels`/`max_pixels`: Image processing constraints

## Error Handling

The API returns standard HTTP status codes:

- **200**: Success
- **400**: Bad Request (invalid parameters, unsupported file type)
- **404**: Not Found (session expired, file not found)
- **500**: Internal Server Error (processing failed)

Error responses include a `detail` field with the error message.

## Production Considerations

1. **File Size Limits**: Configure appropriate file upload limits
2. **Session Cleanup**: Implement automatic cleanup of old sessions
3. **Rate Limiting**: Add rate limiting for production use
4. **Security**: Configure CORS appropriately for your domain
5. **Monitoring**: Add logging and monitoring for production deployment
6. **Load Balancing**: Use multiple API server instances if needed

## Comparison with Web Interface

| Feature | Web Interface (Gradio) | API Server |
|---------|----------------------|------------|
| Usage | Interactive UI | Programmatic |
| File Upload | Browser upload | HTTP POST |
| Results | Visual display | JSON response |
| Integration | Manual use | System integration |
| Batch Processing | Manual | Automated |
| Downloads | ZIP files | ZIP files |
