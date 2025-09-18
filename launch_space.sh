#!/bin/bash

apt update
apt install -y ffmpeg
pip install ultralytics

# Start vLLM server in background
./vllm_serve.sh &

# Wait briefly to ensure vLLM is up before Gradio tries to connect
sleep 60

# Display fancy startup message
echo "
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║                   🚀 Gradio Space Starting! 🚀                 ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
"

# Start Gradio app
python3 execute_demo_v3.py
