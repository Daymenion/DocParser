# 📄 DocParser - Advanced Document Parsing Application

DocParser is a powerful document parsing application that combines AI-powered OCR with modern web technologies. Built on top of DotsOCR engine, it provides both REST API and web interface for parsing PDF documents and images.

## 🚀 Features

- **Multi-format Support**: PDF documents and image files
- **REST API**: Clean, well-documented API endpoints
- **Web Interface**: User-friendly Gradio interface
- **Docker Support**: Easy deployment with multi-container setup
- **Flexible Modes**: Combined, API-only, UI-only, or separate services
- **Health Monitoring**: Built-in health checks and monitoring
- **Session Management**: Temporary file handling and cleanup

## 🏗️ Architecture

### Services
- **vLLM Server** (Port 9998): AI model serving with DotsOCR
- **DocParser App** (Port 7860): Main application with API + UI

### Deployment Modes
1. **Combined**: API + UI in single service
2. **API Only**: REST API service only
3. **UI Only**: Web interface only
4. **Separate**: API and UI in different ports

## 🐳 Quick Start with Docker

### Prerequisites
- Docker & Docker Compose
- At least 8GB RAM
- GPU support (recommended)

### One-Command Setup
```bash
make quick-start
```

This will:
1. Download DotsOCR model weights (~6GB)
2. Build Docker images
3. Start all services
4. Run health checks

### Manual Steps
```bash
# 1. Build and start services
make build
make up

# 2. Check status
make status

# 3. View logs
make logs

# 4. Access services
# API + UI: http://localhost:7860
# API Docs: http://localhost:7860/docs
# vLLM: http://localhost:9998
```

## 📡 API Usage

### Health Check
```bash
curl http://localhost:7860/api/health
```

### Parse PDF
```bash
curl -X POST "http://localhost:7860/api/parse/pdf" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

### Parse Image
```bash
curl -X POST "http://localhost:7860/api/parse/image" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

### Download Results
```bash
curl -O -J "http://localhost:7860/api/download/{session_id}"
```

## 🛠️ Development

### Local Development
```bash
# Setup environment
make setup-dev

# Start in development mode
make dev

# Start only API
make dev-api

# Start only UI
make dev-ui
```

### Available Commands
```bash
make help          # Show all commands
make build          # Build Docker images
make up             # Start services
make down           # Stop services
make logs           # View all logs
make logs-app       # View app logs only
make logs-vllm      # View vLLM logs only
make status         # Show service status
make health         # Check service health
make shell          # Open app container shell
make shell-vllm     # Open vLLM container shell
make clean          # Clean containers
make clean-all      # Remove everything
```

## 🔧 Configuration

### Environment Variables
```bash
# App Configuration
APP_MODE=combined    # combined|api|ui|separate
APP_HOST=0.0.0.0
APP_PORT=7860

# vLLM Configuration
VLLM_SERVER=vllm-server:9998
VLLM_PORT=9998
```

### Docker Compose Override
Create `docker-compose.override.yml` for custom settings:
```yaml
version: '3.8'
services:
  docparser-app:
    environment:
      - APP_MODE=api
    ports:
      - "8080:7860"
```

## 📊 Health Monitoring

### Service Health
```bash
make health
```

### Individual Checks
```bash
# App health
curl http://localhost:7860/health

# vLLM health  
curl http://localhost:9998/health
```

## 🗂️ Project Structure

```
DocParser/
├── app.py                    # Main application
├── api/
│   └── api_server.py        # REST API server
├── docker/
│   ├── docker-compose.yml   # Multi-service setup
│   ├── Dockerfile          # App container
│   └── entrypoint.sh       # Container startup
├── weights/
│   └── DotsOCR/            # Model weights (auto-downloaded)
├── Makefile                # Build automation
└── requirements.txt        # Python dependencies
```

## 🐛 Troubleshooting

### Common Issues

1. **Model not found**: Run `make download-model`
2. **Port conflicts**: Check ports 7860 and 9998
3. **GPU issues**: Ensure Docker GPU support
4. **Memory**: Requires at least 8GB RAM

### Logs
```bash
# All services
make logs

# Specific service
make logs-app
make logs-vllm
```

### Reset Everything
```bash
make clean-all
```

## 📄 License

This project uses the DotsOCR model. Please refer to the original license agreement.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Make your changes
4. Test with `make test`
5. Submit a pull request

---

**DocParser** - Powered by DotsOCR | Built with ❤️ for document processing
