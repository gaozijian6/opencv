# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import cv2
import numpy as np
import easyocr
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import os

class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("数字 OCR 识别")
        self.root.geometry("600x700")
        self.root.resizable(False, False)
        
        # 初始化EasyOCR读取器
        print("正在初始化EasyOCR...")
        self.reader = easyocr.Reader(['en'], gpu=False)  # 使用CPU模式避免兼容性问题
        print("EasyOCR初始化完成")
        
        # 状态变量
        self.current_image = None
        self.current_image_path = None
        self.detected_rectangles = []
        self.is_processing = False
        
        self.setup_ui()
        
        # 设置拖拽支持
        self.setup_drag_drop()
    
    def setup_ui(self):
        # 标题
        title_label = tk.Label(
            self.root,
            text="数字 OCR 识别",
            font=("Arial", 24, "bold"),
            pady=20
        )
        title_label.pack()
        
        # 拖拽区域
        self.drag_frame = tk.Frame(
            self.root,
            bg="#f0f0f0",
            relief=tk.RAISED,
            borderwidth=2,
            width=500,
            height=200
        )
        self.drag_frame.pack(pady=20, padx=50, fill=tk.BOTH, expand=True)
        self.drag_frame.pack_propagate(False)
        
        drag_label = tk.Label(
            self.drag_frame,
            text="📷\n拖拽图片到这里",
            font=("Arial", 16),
            bg="#f0f0f0",
            fg="#666666"
        )
        drag_label.pack(expand=True)
        
        # 选择图片按钮
        select_btn = tk.Button(
            self.root,
            text="选择图片",
            command=self.select_image,
            font=("Arial", 12),
            width=15,
            height=2
        )
        select_btn.pack(pady=10)
        
        # 按钮组
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=10)
        
        self.recognize_btn = tk.Button(
            self.button_frame,
            text="识别整张图片",
            command=self.recognize_full_image,
            font=("Arial", 12),
            width=15,
            height=2,
            state=tk.DISABLED
        )
        self.recognize_btn.pack(side=tk.LEFT, padx=10)
        
        # 结果显示区域
        result_frame = tk.Frame(self.root)
        result_frame.pack(pady=20, padx=50, fill=tk.BOTH, expand=True)
        
        result_label = tk.Label(
            result_frame,
            text="识别结果:",
            font=("Arial", 16, "bold"),
            anchor="w"
        )
        result_label.pack(fill=tk.X, pady=(0, 10))
        
        self.result_text = tk.Text(
            result_frame,
            font=("Arial", 32, "bold"),
            height=4,
            wrap=tk.WORD,
            relief=tk.SUNKEN,
            borderwidth=2
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.insert("1.0", "请选择或拖拽图片")
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
    
    def setup_drag_drop(self):
        """设置拖拽支持"""
        def on_drag_enter(event):
            self.drag_frame.config(bg="#d0f0d0")
        
        def on_drag_leave(event):
            self.drag_frame.config(bg="#f0f0f0")
        
        def on_drop(event):
            self.drag_frame.config(bg="#f0f0f0")
            # 获取拖拽的文件路径
            files = self.root.tk.splitlist(event.data)
            if files:
                file_path = files[0]
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.load_image(file_path)
                else:
                    messagebox.showerror("错误", "请拖拽图片文件")
        
        # 绑定拖拽事件
        self.drag_frame.bind("<Button-1>", lambda e: self.select_image())
        self.drag_frame.bind("<Enter>", on_drag_enter)
        self.drag_frame.bind("<Leave>", on_drag_leave)
        
        # 注意: tkinter的拖拽支持有限,这里使用点击选择代替
        # 如果需要真正的拖拽,需要使用tkinterdnd2库
    
    def select_image(self):
        """选择图片文件"""
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
            
            # 更新拖拽区域显示缩略图
            self.update_drag_area(image)
            
            # 更新按钮状态
            self.recognize_btn.config(state=tk.NORMAL)
            
            # 更新状态
            self.update_status(f"已加载: {os.path.basename(image_path)}")
            self.update_result("识别中...", "orange")
            
            # 自动识别
            self.recognize_full_image()
            
        except Exception as e:
            messagebox.showerror("错误", f"加载图片失败: {str(e)}")
    
    def update_drag_area(self, image):
        """更新拖拽区域显示图片缩略图"""
        # 计算缩略图尺寸
        max_width = 480
        max_height = 180
        
        h, w = image.shape[:2]
        scale = min(max_width / w, max_height / h, 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 调整大小
        resized = cv2.resize(image, (new_w, new_h))
        
        # 转换为RGB
        if len(resized.shape) == 3:
            resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            resized_rgb = resized
        
        # 转换为PIL Image
        pil_image = Image.fromarray(resized_rgb)
        photo = ImageTk.PhotoImage(pil_image)
        
        # 清除原有内容
        for widget in self.drag_frame.winfo_children():
            widget.destroy()
        
        # 显示图片
        img_label = tk.Label(self.drag_frame, image=photo, bg="#f0f0f0")
        img_label.image = photo  # 保持引用
        img_label.pack(expand=True)
    
    def detect_rectangles(self, image):
        """检测图片中的正方形(使用OpenCV逻辑,参考Swift代码)"""
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 预处理
        processed = self.preprocess_image(gray)
        
        # 查找轮廓
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        rectangles = []
        h, w = image.shape[:2]
        valid_observations = []
        
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            if area < (w * h * 0.01):  # 过滤太小的轮廓
                continue
            
            # 近似轮廓为多边形
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 检查是否为四边形
            if len(approx) == 4:
                # 获取边界框
                x, y, w_rect, h_rect = cv2.boundingRect(approx)
                
                # 计算宽高比
                aspect_ratio = w_rect / h_rect if h_rect > 0 else 0
                rect_area = (w_rect / w) * (h_rect / h)  # 归一化面积
                
                # 检查是否为近似正方形(宽高比在0.8到1.2之间)
                if 0.8 <= aspect_ratio <= 1.2:
                    # 归一化坐标(0-1)
                    norm_rect = (
                        x / w,
                        y / h,
                        w_rect / w,
                        h_rect / h
                    )
                    rectangles.append(norm_rect)
                    valid_observations.append({
                        'rect': norm_rect,
                        'area': rect_area,
                        'aspect_ratio': aspect_ratio
                    })
        
        # 检查大矩形(外边框,面积大于80%)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < (w * h * 0.8):  # 只检查大矩形
                continue
            
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                x, y, w_rect, h_rect = cv2.boundingRect(approx)
                aspect_ratio = w_rect / h_rect if h_rect > 0 else 0
                rect_area = (w_rect / w) * (h_rect / h)
                
                # 允许稍微放宽宽高比限制(0.7到1.4)来检测外边框
                if rect_area > 0.8 and 0.7 <= aspect_ratio <= 1.4:
                    norm_rect = (x / w, y / h, w_rect / w, h_rect / h)
                    # 检查是否已存在(避免重复)
                    exists = any(
                        self._calculate_overlap(rect, norm_rect) > 0.9
                        for rect in rectangles
                    )
                    if not exists:
                        rectangles.append(norm_rect)
        
        # 过滤重叠的正方形
        rectangles = self._filter_overlapping_rectangles(rectangles)
        
        return rectangles
    
    def _calculate_overlap(self, rect1, rect2):
        """计算两个矩形的重叠度"""
        x1_1, y1_1, w1, h1 = rect1
        x1_2, y1_2 = x1_1 + w1, y1_1 + h1
        
        x2_1, y2_1, w2, h2 = rect2
        x2_2, y2_2 = x2_1 + w2, y2_1 + h2
        
        # 计算交集
        x_intersect = max(0, min(x1_2, x2_2) - max(x1_1, x2_1))
        y_intersect = max(0, min(y1_2, y2_2) - max(y1_1, y2_1))
        intersection_area = x_intersect * y_intersect
        
        # 计算并集
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - intersection_area
        
        if union_area == 0:
            return 0
        
        return intersection_area / union_area
    
    def _filter_overlapping_rectangles(self, rectangles):
        """过滤重叠的正方形"""
        if not rectangles:
            return []
        
        filtered = []
        used = set()
        
        for i, rect in enumerate(rectangles):
            if i in used:
                continue
            
            best_rect = rect
            best_index = i
            
            # 检查是否有重叠的正方形
            for j, other_rect in enumerate(rectangles):
                if i != j and j not in used:
                    overlap = self._calculate_overlap(rect, other_rect)
                    if overlap > 0.8:  # 重叠度超过80%
                        used.add(j)
            
            filtered.append(best_rect)
            used.add(best_index)
        
        return filtered
    
    def preprocess_image(self, gray_image):
        """预处理图像"""
        # 中值滤波
        median_filtered = cv2.medianBlur(gray_image, 3)
        
        # 自适应阈值化
        thresh = cv2.adaptiveThreshold(
            median_filtered, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 19, 2
        )
        
        return thresh
    
    def preprocess_cell(self, cell, blur_kernel_size=3):
        """对单个单元格进行预处理(使用现有OpenCV逻辑)"""
        # 高斯模糊
        blurred = cv2.GaussianBlur(cell, (blur_kernel_size, blur_kernel_size), 0)
        return blurred
    
    def recognize_numbers(self, image):
        """识别图片中的数字(使用EasyOCR和OpenCV预处理)"""
        # 将图片缩放到60x60(参考Swift代码)
        resized = cv2.resize(image, (60, 60))
        
        # 预处理: 对单元格进行高斯模糊
        processed = self.preprocess_cell(resized, blur_kernel_size=3)
        
        # 转换为RGB
        if len(processed.shape) == 2:
            processed_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        else:
            processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        # 使用EasyOCR识别
        results = self.reader.readtext(
            processed_rgb,
            allowlist='123456789',
            paragraph=False
        )
        
        recognized_numbers = []
        
        for (bbox, text, confidence) in results:
            if confidence > 0.3:
                # 处理文本
                text = text.strip()
                
                # 特殊处理: 8经常被误识别为00
                if text == "00" or text == "0O" or text == "O0":
                    text = "8"
                
                # 只保留1-9的数字
                for char in text:
                    if char.isdigit():
                        digit = int(char)
                        if 1 <= digit <= 9:
                            recognized_numbers.append(str(digit))
        
        return recognized_numbers
    
    def recognize_full_image(self):
        """识别整张图片"""
        if self.current_image is None:
            messagebox.showwarning("警告", "请先选择图片")
            return
        
        if self.is_processing:
            return
        
        self.is_processing = True
        self.recognize_btn.config(state=tk.DISABLED)
        self.update_status("正在识别...")
        self.update_result("识别中...", "orange")
        
        # 在后台线程中处理
        thread = threading.Thread(target=self._recognize_worker)
        thread.daemon = True
        thread.start()
    
    def _recognize_worker(self):
        """后台识别工作线程"""
        try:
            # 识别整张图片
            numbers = self.recognize_numbers(self.current_image)
            
            # 更新UI(需要在主线程中执行)
            self.root.after(0, self._update_recognition_result, numbers)
            
        except Exception as e:
            self.root.after(0, lambda: self._handle_error(str(e)))
    
    def _update_recognition_result(self, numbers):
        """更新识别结果"""
        self.is_processing = False
        self.recognize_btn.config(state=tk.NORMAL)
        
        if numbers:
            result_text = "  ".join(numbers)
            self.update_result(result_text, "green")
            self.update_status(f"识别完成: 找到 {len(numbers)} 个数字")
            
            # 显示识别结果窗口(参考Swift代码)
            self.show_result_window(result_text)
        else:
            self.update_result("⚠️ 未识别到数字", "red")
            self.update_status("识别完成: 未找到数字")
    
    def show_result_window(self, recognized_text):
        """显示识别结果窗口(参考Swift代码)"""
        # 创建新窗口
        result_window = tk.Toplevel(self.root)
        result_window.title(f"识别结果: {recognized_text}")
        
        # 在原图上绘制识别结果
        annotated_image = self.current_image.copy()
        h, w = annotated_image.shape[:2]
        
        # 在图像中心绘制文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(w, h) * 0.03
        thickness = max(2, int(font_scale / 10))
        
        # 获取文字大小
        (text_width, text_height), baseline = cv2.getTextSize(
            recognized_text, font, font_scale, thickness
        )
        
        # 计算文字位置(居中)
        text_x = (w - text_width) // 2
        text_y = (h + text_height) // 2
        
        # 绘制文字背景
        cv2.rectangle(
            annotated_image,
            (text_x - 10, text_y - text_height - 10),
            (text_x + text_width + 10, text_y + baseline + 10),
            (0, 0, 0),
            -1
        )
        
        # 绘制文字
        cv2.putText(
            annotated_image,
            recognized_text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 255, 0),
            thickness
        )
        
        # 转换为RGB
        annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(annotated_rgb)
        
        # 计算窗口大小
        img_width, img_height = pil_image.size
        window_width = min(img_width + 40, 1200)
        window_height = min(img_height + 100, 800)
        
        result_window.geometry(f"{window_width}x{window_height}")
        result_window.minsize(300, 200)
        
        # 创建滚动区域
        canvas = tk.Canvas(result_window)
        scrollbar_v = tk.Scrollbar(result_window, orient="vertical", command=canvas.yview)
        scrollbar_h = tk.Scrollbar(result_window, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)
        
        # 显示识别结果文字
        text_label = tk.Label(
            result_window,
            text=f"识别结果: {recognized_text}",
            font=("Arial", 16, "bold"),
            anchor="center"
        )
        text_label.pack(pady=10)
        
        # 显示图片
        photo = ImageTk.PhotoImage(pil_image)
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.image = photo  # 保持引用
        
        canvas.configure(scrollregion=canvas.bbox("all"))
        
        # 布局
        canvas.pack(side="left", fill="both", expand=True, padx=20, pady=(0, 20))
        scrollbar_v.pack(side="right", fill="y")
        scrollbar_h.pack(side="bottom", fill="x")
    
    def _handle_error(self, error_msg):
        """处理错误"""
        self.is_processing = False
        self.recognize_btn.config(state=tk.NORMAL)
        self.update_result(f"❌ 识别失败: {error_msg}", "red")
        self.update_status("识别失败")
        messagebox.showerror("错误", f"识别失败: {error_msg}")
    
    def update_result(self, text, color="black"):
        """更新结果显示"""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", text)
        self.result_text.config(state=tk.DISABLED)
        
        # 根据颜色设置文本颜色
        color_map = {
            "green": "#00aa00",
            "red": "#aa0000",
            "orange": "#ff8800",
            "blue": "#0000aa",
            "black": "#000000"
        }
        self.result_text.config(fg=color_map.get(color, "black"))
    
    def update_status(self, text):
        """更新状态栏"""
        self.status_label.config(text=text)


def main():
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
