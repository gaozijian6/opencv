# -*- coding: utf-8 -*-
"""主程序 - 使用photo检测模块进行数独网格检测和裁切"""
import cv2
import numpy as np
import os
import time
import re
from photo_detection import extract_cells_photo, save_image_with_fallback

test={
    'image1.jpg': '',
    'image2.jpg': '',
    'image3.jpg': '',
    'image4.jpg': '',
    'image5.jpg': '',
    'image6.jpg': '',
    'image7.jpg': '',
    'image8.jpg': '',
    'image9.jpg': '',
    'image10.jpg': '',
    'image11.jpg': '',
    'image12.jpg': '',
    'image13.jpg': '',
    'image14.jpg': '',
    'image15.jpg': '',
    'image16.jpg': '',
    'image17.jpg': '',
    'image18.jpg': '',
    'image19.jpg': '',
    'image20.jpg': '',
    'image21.jpg': '',
    'image22.jpg': '',
    'image23.jpg': '',
    'image24.jpg': '',
}

def extract_cells_only(image_path, cells_dir):
    """只裁切单元格，不做识别
    
    使用photo_detection模块进行检测
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return False
    
    # 创建cells目录（提前创建，用于保存增强图片）
    if not os.path.exists(cells_dir):
        os.makedirs(cells_dir)
    
    # 使用photo_detection来检测
    success, grid_roi, w, h = extract_cells_photo(image, cells_dir)
    if not success:
        print(f"未找到网格: {image_path}")
        return False
    
    # 计算单元格尺寸
    cell_width = w / 9
    cell_height = h / 9
    
    # 创建标注了9x9网格分割线的图片
    grid_with_lines = grid_roi.copy()
    if len(grid_with_lines.shape) == 2:
        grid_with_lines = cv2.cvtColor(grid_with_lines, cv2.COLOR_GRAY2BGR)
    
    # 绘制外边框
    border_color = (255, 0, 255)  # 洋红色
    border_thickness = 3
    cv2.rectangle(grid_with_lines, (0, 0), (w - 1, h - 1), border_color, border_thickness)
    
    # 绘制9x9分割线
    line_color = (0, 255, 0)  # 绿色
    line_thickness = 2
    
    # 绘制10条竖线
    for i in range(10):
        x_pos = int(i * cell_width)
        cv2.line(grid_with_lines, (x_pos, 0), (x_pos, h), line_color, line_thickness)
    
    # 绘制10条横线
    for i in range(10):
        y_pos = int(i * cell_height)
        cv2.line(grid_with_lines, (0, y_pos), (w, y_pos), line_color, line_thickness)
    
    # 在每个单元格中心标注坐标 (row, col)
    for row in range(9):
        for col in range(9):
            center_x = int((col + 0.5) * cell_width)
            center_y = int((row + 0.5) * cell_height)
            label = f"({row},{col})"
            font_scale = 0.4
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            text_x = center_x - text_size[0] // 2
            text_y = center_y + text_size[1] // 2
            # 绘制文字背景（半透明）
            cv2.rectangle(grid_with_lines, 
                         (text_x - 2, text_y - text_size[1] - 2),
                         (text_x + text_size[0] + 2, text_y + 2),
                         (0, 0, 0), -1)
            # 绘制文字
            cv2.putText(grid_with_lines, label, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 1)
    
    # 保存标注了网格分割线的图片
    grid_lines_filename = os.path.join(cells_dir, 'grid_with_lines.jpg')
    save_image_with_fallback(grid_with_lines, grid_lines_filename)
    print(f"✓ 网格标注图片已保存: {grid_lines_filename}")
    
    # 裁切81个单元格
    for row in range(9):
        for col in range(9):
            margin_x = int(cell_width * 0.08)
            margin_y = int(cell_height * 0.08)
            cell_x = int(col * cell_width + margin_x)
            cell_y = int(row * cell_height + margin_y)
            cell_w = int(cell_width - 2 * margin_x)
            cell_h = int(cell_height - 2 * margin_y)
            
            cell_x = max(0, cell_x)
            cell_y = max(0, cell_y)
            cell_w = min(cell_w, w - cell_x)
            cell_h = min(cell_h, h - cell_y)
            
            cell = grid_roi[cell_y:cell_y + cell_h, cell_x:cell_x + cell_w]
            if cell.size > 0 and cell_w > 0 and cell_h > 0:
                cell_filename = os.path.join(cells_dir, f'cell_{row}_{col}.jpg')
                save_image_with_fallback(cell, cell_filename)
    
    return True

def main():
    """主程序 - 只裁切，不做识别"""
    # 记录程序开始时间
    start_time = time.time()
    print("=" * 60)
    print("数独图片裁切程序开始运行...")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print("=" * 60)
    
    # 遍历test字典中的所有图片
    for image_name in test.keys():
        print(f"\n处理图片: {image_name}")
        
        # 从图片名提取数字（如 image1.jpg -> 1, image.jpg -> 空）
        match = re.search(r'(\d+)', image_name)
        if match:
            image_num = match.group(1)
            cells_dir = f"cells{image_num}"
        else:
            # 如果没有数字，使用图片名（去掉扩展名）
            base_name = os.path.splitext(image_name)[0]
            cells_dir = f"cells_{base_name}"
        
        # 裁切单元格
        if extract_cells_only(image_name, cells_dir):
            print(f"✓ {image_name} 裁切完成，保存到 {cells_dir}/")
        else:
            print(f"✗ {image_name} 裁切失败")
    
    # 记录程序结束时间并计算总耗时
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 60)
    print("程序运行完成！")
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"总耗时: {total_time:.2f} 秒")
    print("=" * 60)

if __name__ == "__main__":
    main()
