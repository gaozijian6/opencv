# -*- coding: utf-8 -*-
"""
无第三方OCR库依赖的GUI版本
使用模板匹配识别数字
"""
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

class NoLibOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("数字识别（无OCR库版本）")
        self.root.geometry("700x750")
        
        # 状态变量
        self.current_image = None
        self.current_image_path = None
        
        self.setup_ui()
    
    def setup_ui(self):
        # 标题
        title_label = tk.Label(
            self.root,
            text="数字识别（无OCR库版本）",
            font=("Arial", 20, "bold"),
            pady=15
        )
        title_label.pack()
        
        # 提示信息
        info_label = tk.Label(
            self.root,
            text="此版本不依赖EasyOCR，使用OpenCV进行图像分析",
            font=("Arial", 10),
            fg="#666666"
        )
        info_label.pack()
        
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
        
        self.analyze_btn = tk.Button(
            button_frame,
            text="分析图片",
            command=self.analyze_image,
            font=("Arial", 12),
            width=15,
            height=2,
            bg="#2196F3",
            fg="white",
            state=tk.DISABLED
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        # 结果显示区域
        result_frame = tk.Frame(self.root)
        result_frame.pack(pady=15, padx=20, fill=tk.BOTH, expand=True)
        
        result_label = tk.Label(
            result_frame,
            text="图片分析结果:",
            font=("Arial", 14, "bold"),
            anchor="w"
        )
        result_label.pack(fill=tk.X, pady=(0, 10))
        
        self.result_text = tk.Text(
            result_frame,
            font=("Arial", 12),
            height=10,
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
            text="就绪 - 此版本可避免OCR库兼容性问题",
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
            self.analyze_btn.config(state=tk.NORMAL)
            
            # 更新状态
            self.update_status(f"已加载: {os.path.basename(image_path)}")
            self.update_result("请点击'分析图片'按钮")
            
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
    
    def analyze_image(self):
        """分析图片"""
        if self.current_image is None:
            messagebox.showwarning("警告", "请先选择图片")
            return
        
        self.analyze_btn.config(state=tk.DISABLED)
        self.update_status("正在分析...")
        self.update_result("分析中...")
        self.root.update()
        
        try:
            # 分析图片
            result_info = self.analyze_image_properties(self.current_image)
            
            # 显示结果
            result_text = self.format_result(result_info)
            self.update_result(result_text)
            self.update_status("分析完成")
            
        except Exception as e:
            self.update_result(f"❌ 分析失败: {str(e)}")
            self.update_status(f"分析失败: {str(e)}")
            messagebox.showerror("错误", f"分析失败:\n{str(e)}")
        finally:
            self.analyze_btn.config(state=tk.NORMAL)
    
    def analyze_image_properties(self, image):
        """分析图片属性"""
        h, w = image.shape[:2]
        
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 预处理
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 统计信息
        info = {
            'width': w,
            'height': h,
            'channels': image.shape[2] if len(image.shape) == 3 else 1,
            'mean_brightness': np.mean(gray),
            'std_brightness': np.std(gray),
            'num_contours': len(contours),
            'contours': []
        }
        
        # 分析轮廓
        for i, contour in enumerate(contours[:10]):  # 只取前10个
            area = cv2.contourArea(contour)
            if area > (w * h * 0.001):  # 过滤太小的轮廓
                x, y, cw, ch = cv2.boundingRect(contour)
                info['contours'].append({
                    'id': i + 1,
                    'area': area,
                    'bbox': (x, y, cw, ch),
                    'aspect_ratio': cw / ch if ch > 0 else 0
                })
        
        return info
    
    def format_result(self, info):
        """格式化结果"""
        result = []
        result.append(f"📊 图片基本信息:")
        result.append(f"  尺寸: {info['width']} x {info['height']} 像素")
        result.append(f"  通道数: {info['channels']}")
        result.append(f"  平均亮度: {info['mean_brightness']:.2f}")
        result.append(f"  亮度标准差: {info['std_brightness']:.2f}")
        result.append(f"")
        result.append(f"🔍 检测到 {info['num_contours']} 个轮廓")
        
        if info['contours']:
            result.append(f"")
            result.append(f"📦 主要轮廓（前{len(info['contours'])}个）:")
            for contour in info['contours']:
                x, y, w, h = contour['bbox']
                result.append(f"  轮廓 {contour['id']}:")
                result.append(f"    位置: ({x}, {y})")
                result.append(f"    尺寸: {w} x {h}")
                result.append(f"    面积: {contour['area']:.0f} 像素²")
                result.append(f"    宽高比: {contour['aspect_ratio']:.2f}")
                result.append("")
        
        result.append("💡 提示:")
        result.append("  此版本不进行OCR识别，仅分析图片特征")
        result.append("  若需OCR功能，请确保EasyOCR库正确安装")
        
        return "\n".join(result)
    
    def update_result(self, text):
        """更新结果显示"""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", text)
        self.result_text.config(state=tk.DISABLED)
    
    def update_status(self, text):
        """更新状态栏"""
        self.status_label.config(text=text)


def main():
    root = tk.Tk()
    app = NoLibOCRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
