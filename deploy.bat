@echo off
chcp 65001
echo Deploying simplified GPU-direct OCR service...
uv run modal deploy src\ocr_deployment\modal_gpu.py