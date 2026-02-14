import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

try:
    from app.core.config import settings
    print(f"Successfully loaded settings.")
    print(f"VLLM_BASE_URL: {settings.VLLM_BASE_URL}")
    print(f"MODEL_NAME: {settings.MODEL_NAME}")
    print(f"OLLAMA_BASE_URL: {settings.OLLAMA_BASE_URL}")

    # Test connection
    import requests
    print("\nTesting VLLM connection...")
    try:
        # Check health endpoint (assuming it's at root or /health, not /v1/health usually)
        # settings.VLLM_BASE_URL is http://vllm:8000/v1
        base = settings.VLLM_BASE_URL.replace("/v1", "")
        resp = requests.get(f"{base}/health", timeout=5)
        print(f"VLLM Health Check: {resp.status_code}")
    except Exception as e:
        print(f"VLLM Connection Failed: {e}")

except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error loading settings: {e}")
