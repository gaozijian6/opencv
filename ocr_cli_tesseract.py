#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命令行版本的 OCR 识别（使用 Tesseract，避免 GUI 兼容性问题）
"""
import cv2
import pytesseract
import sys
import os

def recognize_numbers_tesseract(image_path):
    """使用 Tesseract 识别图片中的数字"""
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图片: {image_path}")
        return None
    
    print(f"📸 已加载图片: {os.path.basename(image_path)}")
    print(f"   尺寸: {image.shape[1]} x {image.shape[0]}")
    
    # 缩放到60x60（参考 Swift 代码）
    resized = cv2.resize(image, (60, 60))
    
    # 预处理
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized
    
    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 二值化
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    print("🔍 正在识别...")
    
    # 使用 Tesseract 识别（只识别数字）
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=123456789'
    text = pytesseract.image_to_string(binary, config=custom_config)
    
    # 处理结果
    recognized_numbers = []
    text = text.strip()
    
    # 特殊处理
    if text in ["00", "0O", "O0"]:
        text = "8"
    
    # 只保留1-9的数字
    for char in text:
        if char.isdigit():
            digit = int(char)
            if 1 <= digit <= 9:
                recognized_numbers.append(str(digit))
    
    return recognized_numbers


def main():
    if len(sys.argv) < 2:
        print("📋 使用方法:")
        print(f"   {sys.argv[0]} <图片路径>")
        print()
        print("示例:")
        print(f"   {sys.argv[0]} image1.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"❌ 文件不存在: {image_path}")
        sys.exit(1)
    
    # 识别数字
    numbers = recognize_numbers_tesseract(image_path)
    
    if numbers:
        result = "  ".join(numbers)
        print()
        print("=" * 50)
        print(f"✅ 识别结果: {result}")
        print(f"   找到 {len(numbers)} 个数字")
        print("=" * 50)
    else:
        print()
        print("⚠️  未识别到数字")


if __name__ == "__main__":
    main()
