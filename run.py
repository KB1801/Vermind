"""
VeriMind Master Orchestrator & Multi-Process Launcher
Handles system verification, directory creation, dependency checks, and service startup.
"""
import sys
import os
import subprocess
import time
from pathlib import Path

# Fix Windows console UTF-8 printing
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def print_banner():
    banner = """
    ====================================================================
      VERIMIND v2.2.0 - Hybrid RAG & Agentic AI Enterprise Platform
      Dual-Engine | Atomic Verification | 4 Native ML Models | 8GB Optimized
    ====================================================================
    """
    print(banner)

def verify_environment():
    print("[1/3] Verifying runtime environment...")
    data_dir = Path("data/models")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    print("  ✓ Directory structure confirmed.")

def check_services():
    print("[2/3] Checking background service accessibility...")
    import socket
    
    def is_port_open(host, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            return s.connect_ex((host, port)) == 0

    qdrant_online = is_port_open("localhost", 6333)
    redis_online = is_port_open("localhost", 6379)
    
    print(f"  * Qdrant (6333): {'ONLINE' if qdrant_online else 'OFFLINE (Will use in-memory vector store fallback)'}")
    print(f"  * Redis (6379):  {'ONLINE' if redis_online else 'OFFLINE (Will use in-memory caching fallback)'}")

def start_backend():
    print("[3/3] Starting VeriMind FastAPI Backend on http://localhost:8000...")
    cmd = [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[!] VeriMind Server Shutdown Requested.")

if __name__ == "__main__":
    print_banner()
    verify_environment()
    check_services()
    start_backend()