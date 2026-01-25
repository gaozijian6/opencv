# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

# 从 index.py 导入识别函数
import sys
sys.path.insert(0, os.path.dirname(__file__))

class SimpleOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("数字 OCR 识别")
        self.root.geometry("700x750")
        
        # 延迟加载 EasyOCR（避免初始化时崩溃）
        self.reader = None
        self.reader_initialized = False
        
        # 状态变量
        self.current_image = None
        self.current_image_path = None
        
        self.setup_ui()
    
    def init_reader(self):
        """延迟初始化 EasyOCR"""
        if not self.reader_initialized:
            try:
                self.update_status("正在初始化 EasyOCR（首次使用需要下载模型，请稍候）...")
                self.root.update()
                
                import easyocr
                self.reader = easyocr.Reader(['en', 'ch_sim'], gpu=True)
                self.reader_initialized = True
                self.update_status("EasyOCR 初始化完成")
                return True
            except Exception as e:
                messagebox.showerror("错误", f"EasyOCR 初始化失败:\n{str(e)}\n\n请检查依赖是否正确安装")
                self.update_status(f"初始化失败: {str(e)}")
                return False
        return True
    
    def setup_ui(self):
        # 标题
        title_label = tk.Label(
            self.root,
            text="数字 OCR 识别（简化版）",
            font=("Arial", 20, "bold"),
            pady=15
        )
        title_label.pack()
        
        # 图片显示区域
        self.image_frame = tk.Frame(
            self.root,
            bg="#f0f0f0",
            relief=tk.RAISED,
            borderwidth=2,
            width=660,
            height=300
        )
        self.image_frame.pack(pady=15, padx=20)
        self.image_frame.pack_propagate(False)
        
        self.image_label = tk.Label(
            self.image_frame,
            text="📷\n点击下方按钮选择图片",
            font=("Arial", 14),
            bg="#f0f0f0",
            fg="#666666"
        )
        self.image_label.pack(expand=True)
        
        # 按钮组
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        self.select_btn = tk.Button(
            button_frame,
            text="选择图片",
            command=self.select_image,
            font=("Arial", 12),
            width=15,
            height=2,
            bg="#4CAF50",
            fg="white"
        )
        self.select_btn.pack(side=tk.LEFT, padx=5)
        
        self.recognize_btn = tk.Button(
            button_frame,
            text="识别数字",
            command=self.recognize_image,
            font=("Arial", 12),
            width=15,
            height=2,
            bg="#2196F3",
            fg="white",
            state=tk.DISABLED
        )
        self.recognize_btn.pack(side=tk.LEFT, padx=5)
        
        # 结果显示区域
        result_frame = tk.Frame(self.root)
        result_frame.pack(pady=15, padx=20, fill=tk.BOTH, expand=True)
        
        result_label = tk.Label(
            result_frame,
            text="识别结果:",
            font=("Arial", 14, "bold"),
            anchor="w"
        )
        result_label.pack(fill=tk.X, pady=(0, 10))
        
        self.result_text = tk.Text(
            result_frame,
            font=("Arial", 28, "bold"),
            height=4,
            wrap=tk.WORD,
            relief=tk.SUNKEN,
            borderwidth=2,
            bg="#ffffff"
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.insert("1.0", "等待选择图片...")
        self.result_text.config(state=tk.DISABLED)
        
        # 状态栏
        self.status_label = tk.Label(
            self.root,
            text="就绪",
            font=("Arial", 10),
            anchor="w",
            relief=tk.SUNKEN,
            bd=1
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def select_image(self):
        """选择图片"""
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[
                ("图片文件", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("所有文件", "*.*")
            ]
        )
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, image_path):
        """加载图片"""
        try:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                messagebox.showerror("错误", "无法加载图片")
                return
            
            self.current_image = image
            self.current_image_path = image_path
            
            # 显示缩略图
            self.show_thumbnail(image)
            
            # 更新按钮状态
            self.recognize_btn.config(state=tk.NORMAL)
            
            # 更新状态
            self.update_status(f"已加载: {os.path.basename(image_path)}")
            self.update_result("请点击'识别数字'按钮", "#0000aa")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载图片失败: {str(e)}")
    
    def show_thumbnail(self, image):
        """显示图片缩略图"""
        # 计算缩略图尺寸
        max_width = 640
        max_height = 280
        
        h, w = image.shape[:2]
        scale = min(max_width / w, max_height / h, 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 调整大小
        resized = cv2.resize(image, (new_w, new_h))
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 转换为 PIL Image
        pil_image = Image.fromarray(resized_rgb)
        photo = ImageTk.PhotoImage(pil_image)
        
        # 显示图片
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo
    
    def recognize_image(self):
        """识别图片中的数字"""
        if self.current_image is None:
            messagebox.showwarning("警告", "请先选择图片")
            return
        
        # 初始化 reader
        if not self.init_reader():
            return
        
        self.recognize_btn.config(state=tk.DISABLED)
        self.update_status("正在识别...")
        self.update_result("识别中...", "#ff8800")
        self.root.update()
        
        try:
            # 使用简单的识别方法
            numbers = self.recognize_numbers_simple(self.current_image)
            
            if numbers:
                result_text = "  ".join(numbers)
                self.update_result(result_text, "#00aa00")
                self.update_status(f"识别完成: 找到 {len(numbers)} 个数字")
                messagebox.showinfo("识别完成", f"识别结果: {result_text}")
            else:
                self.update_result("⚠️ 未识别到数字", "#aa0000")
                self.update_status("识别完成: 未找到数字")
                
        except Exception as e:
            self.update_result(f"❌ 识别失败", "#aa0000")
            self.update_status(f"识别失败: {str(e)}")
            messagebox.showerror("错误", f"识别失败:\n{str(e)}")
        finally:
            self.recognize_btn.config(state=tk.NORMAL)
    
    def recognize_numbers_simple(self, image):
        """简单的数字识别（缩放到60x60）"""
        # 缩放到60x60
        resized = cv2.resize(image, (60, 60))
        
        # 转换为RGB
        if len(resized.shape) == 2:
            resized_rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        else:
            resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 使用 EasyOCR 识别
        results = self.reader.readtext(
            resized_rgb,
            allowlist='123456789',
            paragraph=False
        )
        
        recognized_numbers = []
        
        for (bbox, text, confidence) in results:
            if confidence > 0.3:
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
    
    def update_result(self, text, color="#000000"):
        """更新结果显示"""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", text)
        self.result_text.config(state=tk.DISABLED, fg=color)
    
    def update_status(self, text):
        """更新状态栏"""
        self.status_label.config(text=text)


def main():
    root = tk.Tk()
    app = SimpleOCRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
