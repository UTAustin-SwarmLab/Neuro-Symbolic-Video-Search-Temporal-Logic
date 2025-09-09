#!/bin/bash

# Start vLLM server in background
# ./vllm_serve.sh &

# Wait briefly to ensure vLLM is up before Gradio tries to connect
# sleep 60

# Display fancy startup message
echo "
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║                   🚀 Gradio Space Starting! 🚀                 ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
"

# Start Gradio app
uv run execute_demo.py
