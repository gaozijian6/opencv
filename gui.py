# -*- coding: utf-8 -*-
"""GUI界面 - 拖放图片自动裁切数独网格"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import os
import threading
import re
from index import extract_cells_only, test

# 尝试导入拖放库，如果不可用则禁用拖放功能
DND_AVAILABLE = False
TkClass = tk.Tk
DND_FILES = None

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    # 尝试创建窗口测试是否真的可用（tkinterdnd2 需要 tkdnd 库支持）
    test_root = TkinterDnD.Tk()
    test_root.destroy()
    DND_AVAILABLE = True
    TkClass = TkinterDnD.Tk
except (ImportError, SystemError, OSError, AttributeError, RuntimeError) as e:
    # tkinterdnd2 在 macOS 上可能需要额外的 tkdnd 库配置
    # 如果不可用，静默禁用拖放功能，使用标准 tk.Tk
    DND_AVAILABLE = False
    TkClass = tk.Tk
    DND_FILES = None


class SudokuCutterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("数独图片裁切工具")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        # 设置背景色（macOS兼容）
        self.root.configure(bg="white")
        
        # 设置输出目录
        self.output_dir = "cells0"
        
        # 创建主框架（使用tk.Frame而不是ttk.Frame，提高兼容性）
        main_frame = tk.Frame(root, bg="white", padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = tk.Label(
            main_frame, 
            text="数独图片裁切工具", 
            font=("Arial", 18, "bold"),
            bg="white",
            fg="black"
        )
        title_label.pack(pady=(0, 20))
        
        # 拖放区域（使用tk.LabelFrame提高兼容性）
        drop_frame = tk.LabelFrame(
            main_frame, 
            text="📁 图片拖放区域", 
            bg="white",
            fg="black",
            padx=20, 
            pady=20,
            relief=tk.RAISED,
            bd=2
        )
        drop_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # 创建拖放区域容器（保存为实例变量）
        self.drop_container = tk.Frame(drop_frame, bg="#e3f2fd", relief=tk.SUNKEN, bd=2)
        self.drop_container.pack(fill=tk.BOTH, expand=True)
        
        # 主提示文字
        if DND_AVAILABLE:
            drop_text = "📤 请将图片拖入此处"
            sub_text = "或点击下方按钮选择文件"
        else:
            drop_text = "📂 请点击下方按钮选择文件"
            sub_text = "(拖放功能需要安装tkinterdnd2库)"
        
        self.drop_label = tk.Label(
            self.drop_container,
            text=drop_text,
            font=("Arial", 16, "bold"),
            bg="#e3f2fd",
            fg="#1976d2",
            cursor="hand2"
        )
        self.drop_label.pack(pady=(30, 10))
        
        # 副提示文字
        self.drop_sub_label = tk.Label(
            self.drop_container,
            text=sub_text,
            font=("Arial", 11),
            bg="#e3f2fd",
            fg="#666666",
            cursor="hand2"
        )
        self.drop_sub_label.pack(pady=(0, 30))
        
        # 支持的格式提示（保存为实例变量）
        self.format_label = tk.Label(
            self.drop_container,
            text="支持格式: JPG, PNG, BMP, TIFF",
            font=("Arial", 9),
            bg="#e3f2fd",
            fg="#999999"
        )
        self.format_label.pack(pady=(0, 10))
        
        # 注册拖放事件（如果可用）
        if DND_AVAILABLE:
            self.drop_container.drop_target_register(DND_FILES)
            self.drop_container.dnd_bind('<<Drop>>', self.on_drop)
            self.drop_label.drop_target_register(DND_FILES)
            self.drop_label.dnd_bind('<<Drop>>', self.on_drop)
            self.drop_sub_label.drop_target_register(DND_FILES)
            self.drop_sub_label.dnd_bind('<<Drop>>', self.on_drop)
        
        # 点击整个拖放区域都可以选择文件
        self.drop_container.bind('<Button-1>', self.on_click_select)
        self.drop_label.bind('<Button-1>', self.on_click_select)
        self.drop_sub_label.bind('<Button-1>', self.on_click_select)
        
        # 鼠标悬停效果
        def on_enter(event):
            self.drop_container.config(bg="#bbdefb")
            self.drop_label.config(bg="#bbdefb")
            self.drop_sub_label.config(bg="#bbdefb")
            self.format_label.config(bg="#bbdefb")
        
        def on_leave(event):
            self.drop_container.config(bg="#e3f2fd")
            self.drop_label.config(bg="#e3f2fd")
            self.drop_sub_label.config(bg="#e3f2fd")
            self.format_label.config(bg="#e3f2fd")
        
        self.drop_container.bind('<Enter>', on_enter)
        self.drop_container.bind('<Leave>', on_leave)
        self.drop_label.bind('<Enter>', on_enter)
        self.drop_label.bind('<Leave>', on_leave)
        self.drop_sub_label.bind('<Enter>', on_enter)
        self.drop_sub_label.bind('<Leave>', on_leave)
        
        # 按钮区域
        button_frame = tk.Frame(main_frame, bg="white")
        button_frame.pack(pady=(0, 10))
        
        # 选择文件按钮
        select_btn = tk.Button(
            button_frame,
            text="选择图片文件",
            command=self.select_file,
            font=("Arial", 12),
            bg="#4CAF50",
            fg="white",
            relief=tk.RAISED,
            padx=20,
            pady=10,
            cursor="hand2"
        )
        select_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # 本地测试按钮
        test_btn = tk.Button(
            button_frame,
            text="本地测试",
            command=self.run_local_test,
            font=("Arial", 12),
            bg="#FF9800",
            fg="white",
            relief=tk.RAISED,
            padx=20,
            pady=10,
            cursor="hand2"
        )
        test_btn.pack(side=tk.LEFT)
        
        # 进度条
        self.progress = ttk.Progressbar(
            main_frame,
            mode='indeterminate',
            length=400
        )
        self.progress.pack(pady=(0, 10))
        
        # 状态标签
        self.status_label = tk.Label(
            main_frame,
            text="等待处理图片...",
            font=("Arial", 10),
            bg="white",
            fg="black"
        )
        self.status_label.pack()
        
        # 输出目录显示
        dir_frame = tk.Frame(main_frame, bg="white")
        dir_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Label(
            dir_frame, 
            text="输出目录:", 
            font=("Arial", 10),
            bg="white",
            fg="black"
        ).pack(side=tk.LEFT)
        
        self.dir_label = tk.Label(
            dir_frame,
            text=self.output_dir,
            font=("Arial", 10, "bold"),
            fg="blue",
            bg="white"
        )
        self.dir_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # 修改输出目录按钮
        change_dir_btn = tk.Button(
            dir_frame,
            text="修改",
            command=self.change_output_dir,
            font=("Arial", 10),
            bg="#2196F3",
            fg="white",
            relief=tk.RAISED,
            padx=10,
            pady=5,
            cursor="hand2"
        )
        change_dir_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        # Toast通知标签（初始隐藏）
        self.toast_label = tk.Label(
            root,
            text="",
            font=("Arial", 14, "bold"),
            bg="#323232",
            fg="white",
            relief=tk.RAISED,
            padx=30,
            pady=15,
            borderwidth=2
        )
        # 初始不显示
        self.toast_label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)
        self.toast_label.place_forget()
    
    def on_drop(self, event):
        """处理拖放事件"""
        if not DND_AVAILABLE:
            return
        files = self.root.tk.splitlist(event.data)
        if files:
            file_path = files[0]
            self.process_image(file_path)
    
    def on_click_select(self, event):
        """点击拖放区域时选择文件"""
        self.select_file()
    
    def select_file(self):
        """选择文件对话框"""
        file_path = filedialog.askopenfilename(
            title="选择数独图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("所有文件", "*.*")
            ]
        )
        if file_path:
            self.process_image(file_path)
    
    def change_output_dir(self):
        """修改输出目录"""
        new_dir = filedialog.askdirectory(
            title="选择输出目录",
            initialdir=self.output_dir if os.path.exists(self.output_dir) else "."
        )
        if new_dir:
            self.output_dir = new_dir
            self.dir_label.config(text=self.output_dir)
    
    def run_local_test(self):
        """运行本地测试，处理test字典中的所有图片"""
        # 获取test字典中存在的图片
        test_images = []
        for image_name in test.keys():
            if os.path.exists(image_name):
                test_images.append(image_name)
        
        if not test_images:
            messagebox.showwarning(
                "警告",
                "未找到测试图片！\n\n请确保以下图片文件存在于当前目录：\n" + 
                "\n".join(list(test.keys())[:10]) + 
                (f"\n... 共{len(test)}个" if len(test) > 10 else "")
            )
            return
        
        # 确认对话框
        result = messagebox.askyesno(
            "确认测试",
            f"将处理 {len(test_images)} 张测试图片\n\n"
            f"找到的图片：\n" + 
            "\n".join(test_images[:5]) + 
            (f"\n... 还有 {len(test_images) - 5} 张" if len(test_images) > 5 else "") +
            f"\n\n是否继续？"
        )
        
        if not result:
            return
        
        # 在新线程中运行测试
        thread = threading.Thread(
            target=self._run_local_test_thread,
            args=(test_images,),
            daemon=True
        )
        thread.start()
    
    def _run_local_test_thread(self, test_images):
        """在后台线程中运行本地测试"""
        total = len(test_images)
        success_count = 0
        fail_count = 0
        
        # 更新UI：开始测试
        self.root.after(0, self._update_test_status, f"开始测试: 0/{total}", 0, total)
        self.root.after(0, lambda: self.progress.config(mode='determinate', maximum=total))
        self.root.after(0, lambda: self.progress.start(0))
        
        for idx, image_path in enumerate(test_images, 1):
            filename = os.path.basename(image_path)
            
            # 从图片名提取数字，确定输出目录
            match = re.search(r'(\d+)', filename)
            if match:
                image_num = match.group(1)
                cells_dir = f"cells{image_num}"
            else:
                base_name = os.path.splitext(filename)[0]
                cells_dir = f"cells_{base_name}"
            
            # 更新进度
            self.root.after(0, self._update_test_status, f"处理中: {filename} ({idx}/{total})", idx, total)
            
            # 处理图片
            success = extract_cells_only(image_path, cells_dir)
            
            if success:
                success_count += 1
            else:
                fail_count += 1
        
        # 更新UI：测试完成
        self.root.after(0, self._update_test_complete, success_count, fail_count, total)
    
    def _update_test_status(self, status_text, current, total):
        """更新测试状态"""
        self.status_label.config(text=status_text, bg="white", fg="black")
        self.progress.config(value=current)
    
    def _update_test_complete(self, success_count, fail_count, total):
        """测试完成后的UI更新"""
        self.progress.stop()
        self.progress.config(mode='indeterminate')
        
        result_text = f"测试完成: 成功 {success_count}/{total}, 失败 {fail_count}/{total}"
        self.status_label.config(
            text=result_text,
            bg="white",
            fg="green" if fail_count == 0 else "orange"
        )
        
        # 显示toast通知
        if fail_count == 0:
            self.show_toast(f"测试完成: 全部成功 ({success_count}/{total})", is_success=True)
        else:
            self.show_toast(f"测试完成: 成功 {success_count}, 失败 {fail_count}", is_success=False)
        
        # 显示详细结果对话框
        messagebox.showinfo(
            "测试完成",
            f"本地测试完成！\n\n"
            f"总计: {total} 张图片\n"
            f"成功: {success_count} 张\n"
            f"失败: {fail_count} 张\n"
            f"成功率: {success_count/total*100:.1f}%"
        )
    
    def show_toast(self, message, is_success=True):
        """显示toast通知
        
        参数:
            message: 要显示的消息
            is_success: 是否为成功消息（True=成功/绿色，False=失败/红色）
        """
        # 设置toast样式
        if is_success:
            bg_color = "#4CAF50"  # 绿色
            icon = "✅"
        else:
            bg_color = "#f44336"  # 红色
            icon = "❌"
        
        self.toast_label.config(
            text=f"{icon} {message}",
            bg=bg_color,
            fg="white"
        )
        
        # 显示toast（居中显示在窗口上方）
        self.toast_label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)
        self.toast_label.lift()  # 确保在最上层
        
        # 1秒后自动隐藏
        self.root.after(1000, self._hide_toast)
    
    def _hide_toast(self):
        """隐藏toast通知"""
        self.toast_label.place_forget()
    
    def process_image(self, image_path):
        """处理图片"""
        # 验证文件是否存在
        if not os.path.exists(image_path):
            messagebox.showerror("错误", f"文件不存在: {image_path}")
            return
        
        # 验证是否为图片文件
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext not in valid_extensions:
            messagebox.showerror("错误", f"不支持的图片格式: {file_ext}")
            return
        
        # 验证图片是否可以读取
        test_image = cv2.imread(image_path)
        if test_image is None:
            messagebox.showerror("错误", f"无法读取图片: {image_path}")
            return
        
        # 更新状态
        filename = os.path.basename(image_path)
        self.status_label.config(text=f"正在处理: {filename}", bg="white", fg="black")
        self.progress.start(10)
        self.drop_container.config(bg="#fff3e0")
        self.drop_label.config(text=f"⏳ 处理中: {filename}", bg="#fff3e0", fg="#ff9800")
        self.drop_sub_label.config(text="请稍候...", bg="#fff3e0", fg="#ff9800")
        self.format_label.config(bg="#fff3e0")
        
        # 在新线程中处理，避免界面卡顿
        thread = threading.Thread(
            target=self._process_image_thread,
            args=(image_path,),
            daemon=True
        )
        thread.start()
    
    def _process_image_thread(self, image_path):
        """在后台线程中处理图片"""
        filename = os.path.basename(image_path)
        
        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 调用裁切函数
        success = extract_cells_only(image_path, self.output_dir)
        
        # 在主线程中更新UI
        self.root.after(0, self._process_complete, success, filename)
    
    def _process_complete(self, success, filename):
        """处理完成后的UI更新"""
        self.progress.stop()
        
        if success:
            self.status_label.config(
                text=f"✓ 处理完成: {filename}",
                foreground="green",
                bg="white"
            )
            self.drop_container.config(bg="#c8e6c9")
            self.drop_label.config(
                text=f"✅ 处理完成: {filename}",
                bg="#c8e6c9",
                fg="#2e7d32"
            )
            self.drop_sub_label.config(
                text="📤 拖放新图片继续处理\n或点击下方按钮选择文件",
                bg="#c8e6c9",
                fg="#2e7d32"
            )
            self.format_label.config(bg="#c8e6c9")
            # 显示toast通知
            self.show_toast(f"处理完成: {filename}", is_success=True)
        else:
            self.status_label.config(
                text=f"✗ 处理失败: {filename}",
                foreground="red",
                bg="white"
            )
            self.drop_container.config(bg="#ffcdd2")
            self.drop_label.config(
                text=f"❌ 处理失败: {filename}",
                bg="#ffcdd2",
                fg="#c62828"
            )
            self.drop_sub_label.config(
                text="请检查图片是否包含数独网格\n或尝试其他图片\n\n📤 拖放新图片继续处理",
                bg="#ffcdd2",
                fg="#c62828"
            )
            self.format_label.config(bg="#ffcdd2")
            # 显示toast通知
            self.show_toast(f"处理失败: {filename}", is_success=False)
        
        # 3秒后恢复初始状态
        self.root.after(3000, self._reset_status)
    
    def _reset_status(self):
        """重置状态显示"""
        self.status_label.config(
            text="等待处理图片...",
            foreground="black",
            bg="white"
        )
        # 重置拖放容器背景
        self.drop_container.config(bg="#e3f2fd")
        self.format_label.config(bg="#e3f2fd", fg="#999999")
        
        if DND_AVAILABLE:
            self.drop_label.config(
                text="📤 请将图片拖入此处",
                bg="#e3f2fd",
                fg="#1976d2"
            )
            self.drop_sub_label.config(
                text="或点击下方按钮选择文件",
                bg="#e3f2fd",
                fg="#666666"
            )
        else:
            self.drop_label.config(
                text="📂 请点击下方按钮选择文件",
                bg="#e3f2fd",
                fg="#1976d2"
            )
            self.drop_sub_label.config(
                text="(拖放功能需要安装tkinterdnd2库)",
                bg="#e3f2fd",
                fg="#666666"
            )


def main():
    """主函数"""
    # 尝试创建窗口，如果失败则提示用户
    root = None
    try:
        root = TkClass()
    except Exception as e:
        print(f"无法创建 GUI 窗口: {e}")
        print("\n建议解决方案：")
        print("1. 升级 Python 版本: brew install python@3.11")
        print("2. 或使用命令行版本: python index.py")
        return
    
    app = SudokuCutterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

