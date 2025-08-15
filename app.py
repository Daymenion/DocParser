#!/usr/bin/env python3
"""
DocParser Main Application Server

This script serves both the REST API and Web Interface in a unified application.
It can run API-only, UI-only, or both services together.
"""

import os
import sys
import argparse
import threading
import time
import signal
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import uvicorn
    import gradio as gr
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import RedirectResponse
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Please install required packages:")
    print("   pip install -r requirements.txt")
    DEPENDENCIES_AVAILABLE = False
    sys.exit(1)

# Import our services
from api.api_server import app as api_app, DocumentParsingService, dots_parser
from demo.demo_gradio import create_gradio_interface


class DocParserApp:
    """Main application class that manages both API and UI services"""
    
    def __init__(self):
        self.api_thread = None
        self.ui_thread = None
        self.api_server = None
        self.ui_server = None
        self.running = False
        
    def create_combined_app(self, api_port=7860, ui_port=7860):
        """Create a combined FastAPI app that serves both API and UI"""
        
        # Create main FastAPI app
        main_app = FastAPI(
            title="DocParser Application",
            description="Document parsing with both REST API and Web Interface",
            version="1.0.0"
        )
        
        # Mount the API app
        main_app.mount("/api", api_app)
        
        # Create Gradio interface
        gradio_app = create_gradio_interface()
        
        # Mount Gradio app
        gradio_fastapi = gr.mount_gradio_app(main_app, gradio_app, path="/ui")
        
        # Root redirect to UI
        @main_app.get("/")
        async def root():
            return RedirectResponse(url="/ui")
        
        # Health check for the combined app
        @main_app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "services": {
                    "api": "available at /api",
                    "ui": "available at /ui",
                    "docs": "available at /docs"
                }
            }
        
        return main_app
    
    def start_api_only(self, host="0.0.0.0", port=7860):
        """Start only the API server"""
        print(f"🚀 Starting API server on http://{host}:{port}")
        print(f"📚 API docs: http://{host}:{port}/docs")
        
        self.running = True
        uvicorn.run(api_app, host=host, port=port, log_level="info")
    
    def start_ui_only(self, host="0.0.0.0", port=7860):
        """Start only the UI server"""
        print(f"🎨 Starting Web UI on http://{host}:{port}")
        
        gradio_app = create_gradio_interface()
        self.running = True
        gradio_app.launch(
            server_name=host,
            server_port=port,
            share=False,
            debug=False
        )
    
    def start_combined(self, host="0.0.0.0", port=7860):
        """Start combined API + UI server"""
        print(f"🚀 Starting DotsOCR Application on http://{host}:{port}")
        print(f"🎨 Web Interface: http://{host}:{port}/ui")
        print(f"📚 API Documentation: http://{host}:{port}/docs")
        print(f"🔧 API Endpoints: http://{host}:{port}/api/...")
        
        combined_app = self.create_combined_app()
        self.running = True
        uvicorn.run(combined_app, host=host, port=port, log_level="info")
    
    def start_separate_services(self, api_host="0.0.0.0", api_port=7860, 
                               ui_host="0.0.0.0", ui_port=7860):
        """Start API and UI as separate services"""
        print(f"🚀 Starting DotsOCR with separate services:")
        print(f"   🔧 API Server: http://{api_host}:{api_port}")
        print(f"   🎨 Web Interface: http://{ui_host}:{ui_port}")
        
        # Start API in a separate thread
        def run_api():
            uvicorn.run(api_app, host=api_host, port=api_port, log_level="info")
        
        # Start UI in a separate thread  
        def run_ui():
            gradio_app = create_gradio_interface()
            gradio_app.launch(
                server_name=ui_host,
                server_port=ui_port,
                share=False,
                debug=False
            )
        
        self.api_thread = threading.Thread(target=run_api, daemon=True)
        self.ui_thread = threading.Thread(target=run_ui, daemon=True)
        
        self.api_thread.start()
        time.sleep(2)  # Give API time to start
        self.ui_thread.start()
        
        self.running = True
        
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop all services"""
        print("\n🛑 Stopping DotsOCR Application...")
        self.running = False


def check_vllm_server(ip="127.0.0.1", port=9998):
    """Check if vLLM server is running"""
    try:
        import requests
        response = requests.get(f"http://{ip}:{port}/health", timeout=3)
        return response.status_code == 200
    except:
        return False


def print_startup_info():
    """Print startup information"""
    print("=" * 60)
    print("🔍 DotsOCR - Document Layout Parsing Application")
    print("=" * 60)
    print("Features:")
    print("  🎨 Web Interface - Interactive document parsing")
    print("  🔧 REST API - Programmatic access")
    print("  📄 PDF & Image Support - Multiple document formats")
    print("  🌐 Multiple Prompt Modes - Different parsing strategies")
    print("=" * 60)


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="DotsOCR Application Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start combined API + UI server (recommended)
  python app.py --mode combined --port 7860
  
  # Start only API server
  python app.py --mode api --port 7860
  
  # Start only Web UI
  python app.py --mode ui --port 7860
  
  # Start separate API and UI servers
  python app.py --mode separate --api-port 7860 --ui-port 7861
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["combined", "api", "ui", "separate"],
        default="combined",
        help="Service mode (default: combined)"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=7860, help="Server port (for combined/api/ui mode)")
    parser.add_argument("--api-port", type=int, default=7860, help="API server port (for separate mode)")
    parser.add_argument("--ui-port", type=int, default=7860, help="UI server port (for separate mode)")
    parser.add_argument("--vllm-ip", default="127.0.0.1", help="vLLM server IP")
    parser.add_argument("--vllm-port", type=int, default=9998, help="vLLM server port")
    parser.add_argument("--check-vllm", action="store_true", help="Check vLLM server before starting")
    parser.add_argument("--dev", action="store_true", help="Development mode with auto-reload")
    
    args = parser.parse_args()
    
    print_startup_info()
    
    # Check if we're in the project directory
    if not os.path.exists("dots_ocr"):
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Update parser configuration
    dots_parser.ip = args.vllm_ip
    dots_parser.port = args.vllm_port
    
    # Check vLLM server if requested
    if args.check_vllm:
        print(f"🔍 Checking vLLM server at {args.vllm_ip}:{args.vllm_port}...")
        if check_vllm_server(args.vllm_ip, args.vllm_port):
            print("✅ vLLM server is reachable")
        else:
            print("⚠️  vLLM server is not reachable")
            print("   Start vLLM server with:")
            print(f"   CUDA_VISIBLE_DEVICES=0 vllm serve ./weights/DotsOCR \\")
            print(f"     --tensor-parallel-size 1 --gpu-memory-utilization 0.95 \\")
            print(f"     --chat-template-content-format string --served-model-name model \\")
            print(f"     --trust-remote-code --port {args.vllm_port}")
            
            if not args.dev:
                response = input("\nContinue anyway? (y/N): ")
                if response.lower() != 'y':
                    sys.exit(1)
    
    # Create and start the application
    app = DocParserApp()
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        app.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if args.mode == "combined":
            app.start_combined(args.host, args.port)
        elif args.mode == "api":
            app.start_api_only(args.host, args.port)
        elif args.mode == "ui":
            app.start_ui_only(args.host, args.port)
        elif args.mode == "separate":
            app.start_separate_services(args.host, args.api_port, args.host, args.ui_port)
    
    except KeyboardInterrupt:
        app.stop()
    except Exception as e:
        print(f"❌ Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
