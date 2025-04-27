import subprocess
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def run_servers():
    # Start FastAPI server
    fastapi_process = subprocess.Popen(
        ["python3", "api/run.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Start MCP server
    mcp_process = subprocess.Popen(
        ["python3", "mcp_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        print("Both servers are running. Press Ctrl+C to stop.")
        fastapi_process.wait()
        mcp_process.wait()
    except KeyboardInterrupt:
        print("\nStopping servers...")
        fastapi_process.terminate()
        mcp_process.terminate()
        fastapi_process.wait()
        mcp_process.wait()
        print("Servers stopped.")

if __name__ == "__main__":
    run_servers() 