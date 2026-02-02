
import os
import requests
import json
import base64

VLLM_URL = "http://vllm:8000/v1/chat/completions"
# Use a known file that exists in the container
VIDEO_PATH = "/opt/project_root/videos/proxies/C0011_proxy.mp4"
MODEL = "Qwen/Qwen3-VL-8B-Instruct-FP8"

def test_payload(name, messages):
    print(f"\n--- Testing {name} ---")
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 300
    }
    try:
        response = requests.post(VLLM_URL, json=payload)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("Response:", response.json()['choices'][0]['message']['content'][:200] + "...")
        else:
            print("Error:", response.text)
    except Exception as e:
        print(f"Exception: {e}")

# 1. Text Tag Approach
print("Checking text tag approach...")
msg_text = [
    {"role": "user", "content": f"<|video|>{VIDEO_PATH}\nDescribe this video in detail."}
]
test_payload("Text Tag", msg_text)

# 2. Structured Video Type (Standard OpenAI)
print("Checking structured video type...")
msg_video = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this video."},
            {"type": "video_url", "video_url": {"url": f"file://{VIDEO_PATH}"}} 
        ]
    }
]
# Note: OpenAI standard is 'image_url', some extensions use 'video_url', vLLM docs can allow 'video' type in custom chat_template
test_payload("Structured Video URL", msg_video)

# 3. Custom 'video' type (vLLM specific?)
msg_custom = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this video."},
            {"type": "video", "video": VIDEO_PATH}
        ]
    }
]
test_payload("Custom Video Type", msg_custom)
