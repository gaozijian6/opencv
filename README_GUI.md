# OCR GUI 使用说明

## 环境配置

已创建虚拟环境 `opencv_env`，包含以下依赖：
- Python 3.9.6
- opencv-python 4.13.0
- EasyOCR 1.7.2
- PyTorch 2.8.0
- Pillow 11.3.0
- NumPy 2.0.2

## 启动方法

### 方法 1：使用启动脚本（推荐）

```bash
./run_gui.sh
```

### 方法 2：手动激活虚拟环境

```bash
# 进入项目目录
cd /Users/gaogaozijian/Desktop/opencv

# 激活虚拟环境
source opencv_env/bin/activate

# 运行GUI
python ocr_gui_simple.py

# 退出虚拟环境（使用完成后）
deactivate
```

## 可用的 GUI 程序

1. **ocr_gui_simple.py** （推荐）
   - 简化版 GUI
   - 延迟初始化 EasyOCR（避免启动时崩溃）
   - 支持图片选择和数字识别

2. **ocr_gui.py**
   - 完整版 GUI
   - 包含正方形检测和识别结果窗口
   - 功能更丰富

3. **ocr_gui_nolib.py**
   - 无 OCR 库版本
   - 仅用于图片分析，不进行 OCR 识别
   - 适合测试环境问题

## 使用步骤

1. 双击 `run_gui.sh` 或在终端运行启动脚本
2. 点击"选择图片"按钮，选择要识别的图片
3. 点击"识别数字"按钮（首次使用会下载 EasyOCR 模型，需要几分钟）
4. 查看识别结果

## 注意事项

- 首次运行时 EasyOCR 会自动下载模型文件（约 100MB），需要网络连接
- 模型下载后会缓存，下次运行会更快
- 虚拟环境隔离了系统库，避免版本冲突

## 故障排除

如果遇到问题：

1. 确认虚拟环境已激活（命令行前会显示 `(opencv_env)`）
2. 检查依赖是否安装完整：
   ```bash
   source opencv_env/bin/activate
   pip list
   ```
3. 如需重新安装依赖：
   ```bash
   source opencv_env/bin/activate
   pip install --force-reinstall easyocr opencv-python pillow numpy
   ```

## 文件说明

- `opencv_env/` - Python 虚拟环境目录
- `run_gui.sh` - 启动脚本
- `ocr_gui_simple.py` - 简化版 GUI（推荐）
- `ocr_gui.py` - 完整版 GUI
- `ocr_gui_nolib.py` - 无 OCR 库测试版
- `index.py` - 原有的命令行版本
- `README_GUI.md` - 本说明文档
