import os
import subprocess
import sys

def start_vllm():
    # Environment variables for stability
    os.environ["VLLM_USE_V1"] = "0"
    
    # Add project root to path to import shared logger
    sys.path.append("/opt/project_root")
    try:
        from app.utils.logger import setup_logging
        logger = setup_logging("vllm_service")
    except ImportError:
        # Fallback if import fails (e.g. testing without mount)
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("vllm_service")

    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct-FP8")
    max_model_len = os.getenv("MAX_MODEL_LEN", "8192")
    gpu_mem_util = os.getenv("GPU_MEMORY_UTILIZATION", "0.85")

    # Optimized command for RTX 5070 Ti (16GB VRAM)
    cmd = [
        "vllm", "serve", model_name,
        "--trust-remote-code",
        "--max-model-len", max_model_len,
        "--gpu-memory-utilization", gpu_mem_util,
        "--enforce-eager",
        "--limit-mm-per-prompt", '{"video": 1}',
        "--allowed-local-media-path", "/opt/project_root",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "llama3_json"
    ]
    
    logger.info(f"Starting vLLM server with command: {' '.join(cmd)}")
    
    try:
        # Popen to capture stdout/stderr and log it
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )

        # Stream output to logger
        for line in process.stdout:
            logger.info(line.strip())
            
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"vLLM server exited with error code: {process.returncode}")
            sys.exit(process.returncode)
            
    except Exception as e:
        logger.error(f"vLLM server failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("vLLM server stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    start_vllm()
