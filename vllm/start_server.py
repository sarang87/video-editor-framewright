import os
import subprocess
import sys

def start_vllm():
    # Environment variables for stability
    os.environ["VLLM_USE_V1"] = "0"
    
    # Optimized command for RTX 5070 Ti (16GB VRAM)
    # Using Qwen3-VL-8B-Instruct-FP8 for memory efficiency
    cmd = [
        "vllm", "serve", "Qwen/Qwen3-VL-8B-Instruct-FP8",
        "--trust-remote-code",
        "--max-model-len", "14112",
        "--gpu-memory-utilization", "0.9",
        "--enforce-eager",
        "--limit-mm-per-prompt", '{"video": 1}',
        "--allowed-local-media-path", "/opt/project_root"
    ]
    
    print(f"Starting vLLM server with command: {' '.join(cmd)}")
    
    try:
        # Using subprocess.run to keep the process alive
        # The container will stay alive as long as this process is running
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"vLLM server exited with error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("vLLM server stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    start_vllm()
