#!/bin/bash
# OCR GUI 启动脚本

# 进入项目目录
cd "$(dirname "$0")"

# 激活虚拟环境
source opencv_env/bin/activate

# 运行 GUI (使用 Tesseract 版本，避免 PyTorch 兼容性问题)
python ocr_gui_tesseract.py
