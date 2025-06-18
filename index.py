# -*- coding: utf-8 -*-
import cv2
import numpy as np
import easyocr
import os
import time

res='500169000004000070600000800000802050003000200080401000001000003060000400000546009'

# 初始化EasyOCR读取器（只识别英文数字）
reader = easyocr.Reader(['en', 'ch_sim'], gpu=True)  # 支持英文和简体中文识别

# 创建输出目录
output_dir = "process_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def convert_res_to_array(res_string):
    """将标准答案字符串转换为9x9数组"""
    standard_array = []
    for i in range(9):
        row = []
        for j in range(9):
            digit = int(res_string[i * 9 + j])
            row.append(digit)
        standard_array.append(row)
    return standard_array

def calculate_accuracy(sudoku_array, standard_array):
    """计算识别准确率"""
    total_cells = 81
    correct_cells = 0
    
    # 分类统计
    correct_filled = 0  # 正确识别的非空数字
    total_filled = 0    # 标准答案中的非空数字总数
    correct_empty = 0   # 正确识别的空格
    total_empty = 0     # 标准答案中的空格总数
    
    for i in range(9):
        for j in range(9):
            standard_digit = standard_array[i][j]
            recognized_digit = sudoku_array[i][j]
            
            if standard_digit == recognized_digit:
                correct_cells += 1
                
                if standard_digit == 0:
                    correct_empty += 1
                else:
                    correct_filled += 1
            
            if standard_digit == 0:
                total_empty += 1
            else:
                total_filled += 1
    
    overall_accuracy = correct_cells / total_cells * 100
    filled_accuracy = correct_filled / total_filled * 100 if total_filled > 0 else 0
    empty_accuracy = correct_empty / total_empty * 100 if total_empty > 0 else 0
    
    return {
        'overall_accuracy': overall_accuracy,
        'filled_accuracy': filled_accuracy,
        'empty_accuracy': empty_accuracy,
        'correct_cells': correct_cells,
        'total_cells': total_cells,
        'correct_filled': correct_filled,
        'total_filled': total_filled,
        'correct_empty': correct_empty,
        'total_empty': total_empty
    }

def preprocess_image(image, threshold_param, output_dir):
    """预处理图像以提高OCR识别准确率"""
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 中值滤波，去除椒盐噪声
    median_filtered = cv2.medianBlur(blurred, 3)

    # 自适应阈值化 - 使用传入的参数
    thresh = cv2.adaptiveThreshold(
        median_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, threshold_param, 2
    )

    # 多步形态学操作去除噪声
    # 第一步：开运算去除小的白色噪点
    kernel_open = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # 第二步：闭运算填充数字内部的小孔
    kernel_close = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    # 第三步：再次开运算进一步清理噪点
    kernel_clean = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_clean, iterations=1)

    # 保存预处理结果到指定文件夹
    cv2.imwrite(os.path.join(output_dir, '1_original_gray.jpg'), gray)
    cv2.imwrite(os.path.join(output_dir, '2_gaussian_blur.jpg'), blurred)
    cv2.imwrite(os.path.join(output_dir, '2.5_median_filtered.jpg'), median_filtered)
    cv2.imwrite(os.path.join(output_dir, '3_adaptive_threshold.jpg'), thresh)
    cv2.imwrite(os.path.join(output_dir, '3.5_after_opening.jpg'), opened)
    cv2.imwrite(os.path.join(output_dir, '3.7_after_closing.jpg'), closed)
    cv2.imwrite(os.path.join(output_dir, '4_final_processed.jpg'), processed)
    print(f"预处理图像已保存到 {output_dir} 文件夹 (threshold_param={threshold_param})")
    
    return processed


def find_sudoku_grid(image, output_dir):
    """找到数独网格的轮廓"""
    # 查找轮廓
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大的矩形轮廓（应该是数独网格）
    largest_contour = max(contours, key=cv2.contourArea)

    # 近似轮廓为矩形
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # 保存轮廓检测结果
    contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, '5_largest_contour.jpg'), contour_image)

    # 如果找到了四边形，返回其坐标
    if len(approx) == 4:
        # 保存四边形轮廓
        quad_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(quad_image, [approx], -1, (0, 0, 255), 3)
        cv2.imwrite(os.path.join(output_dir, '6_sudoku_grid_quad.jpg'), quad_image)
        print("找到四边形数独网格")
        return approx.reshape(4, 2)
    else:
        # 如果没有找到完美的四边形，使用边界矩形
        x, y, w, h = cv2.boundingRect(largest_contour)
        rect_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(rect_image, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv2.imwrite(os.path.join(output_dir, '6_sudoku_grid_rectangle.jpg'), rect_image)
        print("使用矩形边界作为数独网格")
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])


def order_points(pts):
    """将四个点按照左上、右上、右下、左下的顺序排序"""
    # 初始化坐标数组
    rect = np.zeros((4, 2), dtype="float32")
    
    # 计算每个点的坐标和
    s = pts.sum(axis=1)
    # 左上角点的坐标和最小，右下角点的坐标和最大
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下
    
    # 计算每个点的坐标差
    diff = np.diff(pts, axis=1)
    # 右上角点的差值最小（x大y小），左下角点的差值最大（x小y大）
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下
    
    return rect


def recognize_digit_with_position_info(cell_image):
    """识别数字并返回位置信息，如果识别到多个数字则过滤掉"""
    # EasyOCR识别
    if len(cell_image.shape) == 2:
        cell_image_color = cv2.cvtColor(cell_image, cv2.COLOR_GRAY2BGR)
    else:
        cell_image_color = cell_image
    
    results = reader.readtext(cell_image_color, allowlist='123456789', width_ths=0.1, height_ths=0.1)
    
    all_digits = []  # 存储所有找到的数字信息
    
    if results:
        for (bbox, text, confidence) in results:
            if confidence > 0.3 and text.isdigit():
                # bbox包含四个点的坐标
                bbox_array = np.array(bbox)
                x_min = int(bbox_array[:, 0].min())
                y_min = int(bbox_array[:, 1].min())
                x_max = int(bbox_array[:, 0].max())
                y_max = int(bbox_array[:, 1].max())
                
                # 如果是单个数字
                if len(text) == 1:
                    all_digits.append({
                        'digit': int(text),
                        'position': (x_min, y_min, x_max, y_max),
                        'confidence': confidence,
                        'area': (x_max - x_min) * (y_max - y_min)
                    })
                else:
                    # 如果OCR识别出连续数字字符串，拆分为单个数字
                    for digit_char in text:
                        if digit_char.isdigit():
                            all_digits.append({
                                'digit': int(digit_char),
                                'position': (x_min, y_min, x_max, y_max),  # 共享同一区域
                                'confidence': confidence,
                                'area': (x_max - x_min) * (y_max - y_min),
                                'from_multi_char': True  # 标记来自多字符识别
                            })
    
    # 判断策略
    if len(all_digits) == 0:
        # 没有识别到数字，尝试轮廓分析
        contours, _ = cv2.findContours(cell_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) >= 50:
                x, y, w, h = cv2.boundingRect(largest_contour)
                return {
                    'digit': 0,  # 轮廓分析不返回具体数字
                    'position': (x, y, x+w, y+h),
                    'confidence': 0.0,
                    'found': True,
                    'filtered_reason': None
                }
        
        return {
            'digit': 0,
            'position': None,
            'confidence': 0.0,
            'found': False,
            'filtered_reason': None
        }
    
    elif len(all_digits) == 1:
        # 只有一个数字，正常返回
        digit_info = all_digits[0]
        return {
            'digit': digit_info['digit'],
            'position': digit_info['position'],
            'confidence': digit_info['confidence'],
            'found': True,
            'filtered_reason': None
        }
    
    else:
        # 识别到多个数字，进行过滤
        # 选择面积最大且置信度最高的数字用于位置信息显示
        best_digit = max(all_digits, key=lambda x: (x['area'], x['confidence']))
        digit_list = [str(d['digit']) for d in all_digits]
        
        return {
            'digit': 0,  # 过滤掉，设置为0
            'position': best_digit['position'],
            'confidence': best_digit['confidence'],
            'found': True,
            'filtered_reason': f"识别到多个数字: {', '.join(digit_list)}"
        }


def extract_digits_from_grid(image, grid_corners, output_dir):
    """从数独网格中提取每个单元格的数字（不使用透视变换）"""
    # 获取网格的边界矩形
    x, y, w, h = cv2.boundingRect(grid_corners)
    
    # 直接从原始图像中裁剪网格区域
    grid_roi = image[y:y+h, x:x+w]
    
    # 保存裁剪的网格区域
    cv2.imwrite(os.path.join(output_dir, '7_grid_roi.jpg'), grid_roi)
    print("网格区域裁剪完成")

    # 在裁剪后的图像上绘制网格线，便于观察
    grid_with_lines = grid_roi.copy()
    if len(grid_with_lines.shape) == 2:  # 如果是灰度图，转换为彩色
        grid_with_lines = cv2.cvtColor(grid_with_lines, cv2.COLOR_GRAY2BGR)
    
    # 创建一个用于绘制数字边界框的图像
    grid_with_digit_boxes = grid_with_lines.copy()
    
    cell_width = w // 9
    cell_height = h // 9
    for i in range(10):  # 绘制10条线（0到9）
        # 绘制垂直线
        cv2.line(grid_with_lines, (i * cell_width, 0), (i * cell_width, h), (0, 255, 0), 1)
        # 绘制水平线
        cv2.line(grid_with_lines, (0, i * cell_height), (w, i * cell_height), (0, 255, 0), 1)
    
    cv2.imwrite(os.path.join(output_dir, '8_grid_with_lines.jpg'), grid_with_lines)
    print("网格线图像已保存")

    # 初始化9x9数组
    sudoku_array = []
    digit_positions = []  # 存储数字位置信息
    filtered_count = 0    # 记录被尺寸条件过滤掉的数字数量

    # 计算单元格尺寸
    cell_width = w // 9
    cell_height = h // 9
    
    # 创建单元格处理步骤的保存目录
    cells_dir = os.path.join(output_dir, 'cells_processing_steps')
    if not os.path.exists(cells_dir):
        os.makedirs(cells_dir)

    print("开始识别每个单元格的数字...")
    for row in range(9):
        sudoku_row = []
        for col in range(9):
            # 计算更大的边距来避免截取到网格边框线
            # 使用单元格尺寸的15%作为边距，最少12像素
            margin_x = max(int(cell_width * 0.15), 12)
            margin_y = max(int(cell_height * 0.15), 12)
            
            # 计算单元格在裁剪图像中的坐标
            cell_x = col * cell_width + margin_x
            cell_y = row * cell_height + margin_y
            cell_w = cell_width - 2 * margin_x  # 左右各减去边距
            cell_h = cell_height - 2 * margin_y  # 上下各减去边距

            # 确保不会越界
            cell_x = max(0, cell_x)
            cell_y = max(0, cell_y)
            cell_w = min(cell_w, w - cell_x)
            cell_h = min(cell_h, h - cell_y)

            # 提取单元格
            cell = grid_roi[cell_y:cell_y + cell_h, cell_x:cell_x + cell_w]

            # 进一步处理单元格
            if cell.size > 0 and cell_w > 0 and cell_h > 0:
                # 保存原始截取的单元格
                cv2.imwrite(os.path.join(cells_dir, f'cell_{row}_{col}_01_original.jpg'), cell)
                
                # 对单元格进行额外的预处理
                cell_cleaned = preprocess_cell(cell)
                
                # 保存预处理后的单元格
                cv2.imwrite(os.path.join(cells_dir, f'cell_{row}_{col}_02_cleaned.jpg'), cell_cleaned)
                
                # 调整大小以提高OCR准确性
                cell_resized = cv2.resize(cell_cleaned, (84, 84))
                
                # 保存最终喂给OCR的单元格（这是实际识别的图像）
                cv2.imwrite(os.path.join(cells_dir, f'cell_{row}_{col}_03_final_for_ocr.jpg'), cell_resized)

                # 使用改进的识别函数获取数字和位置信息
                result_info = recognize_digit_with_position_info(cell_resized)
                
                digit = result_info['digit']
                
                # 新增：宽度和高度条件检查
                size_check_passed = True
                filter_reason = ""
                if result_info['found'] and result_info['position'] and digit > 0:
                    x_min, y_min, x_max, y_max = result_info['position']
                    digit_width = x_max - x_min
                    digit_height = y_max - y_min
                    
                    # 在调整后的84x84图像中，方格的有效宽度和高度都是84
                    resized_cell_width = 84
                    resized_cell_height = 84
                    min_required_width = resized_cell_width * 0.5   # 方格宽度的40%
                    min_required_height = resized_cell_height * 0.5  # 方格高度的40%
                    
                    # 检查宽度和高度，只有两个都小于最小要求才过滤
                    if digit_width < min_required_width and digit_height < min_required_height:
                        filter_reason = f"宽度{digit_width}px和高度{digit_height}px都小于最小要求{min_required_width:.1f}px"
                        
                        print(f"方格({row},{col})的数字{digit} {filter_reason}，过滤掉")
                        digit = 0  # 将数字设为0（无数字）
                        size_check_passed = False
                        filtered_count += 1
                
                # 新增：显示多数字过滤信息
                if result_info.get('filtered_reason'):
                    print(f"方格({row},{col}) {result_info['filtered_reason']}，过滤掉")
                    filtered_count += 1
                
                # 如果识别到了数字，在最终OCR图像上标注识别结果
                if digit > 0:
                    # 创建一个带标注的版本
                    cell_annotated = cv2.cvtColor(cell_resized, cv2.COLOR_GRAY2BGR) if len(cell_resized.shape) == 2 else cell_resized.copy()
                    
                    # 如果有位置信息，画出识别区域
                    if result_info['position']:
                        x_min, y_min, x_max, y_max = result_info['position']
                        cv2.rectangle(cell_annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(cell_annotated, f"Digit: {digit}", (5, 15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(cell_annotated, f"Conf: {result_info['confidence']:.2f}", (5, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    cv2.imwrite(os.path.join(cells_dir, f'cell_{row}_{col}_04_annotated_result.jpg'), cell_annotated)
                elif result_info.get('filtered_reason'):
                    # 为被多数字过滤的方格也创建标注图像
                    cell_annotated = cv2.cvtColor(cell_resized, cv2.COLOR_GRAY2BGR) if len(cell_resized.shape) == 2 else cell_resized.copy()
                    
                    if result_info['position']:
                        x_min, y_min, x_max, y_max = result_info['position']
                        cv2.rectangle(cell_annotated, (x_min, y_min), (x_max, y_max), (0, 165, 255), 2)  # 橙色框
                        cv2.putText(cell_annotated, "Multi-digits", (5, 15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                        cv2.putText(cell_annotated, "FILTERED", (5, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                    
                    cv2.imwrite(os.path.join(cells_dir, f'cell_{row}_{col}_04_annotated_result.jpg'), cell_annotated)
                
                # 同时保存一个对比图，显示边距效果
                if row < 3 and col < 3:  # 只为前几个单元格创建对比图
                    # 创建一个显示裁剪区域的图像
                    crop_demo = grid_roi.copy()
                    if len(crop_demo.shape) == 2:
                        crop_demo = cv2.cvtColor(crop_demo, cv2.COLOR_GRAY2BGR)
                    
                    # 绘制原始网格线
                    orig_x = col * cell_width
                    orig_y = row * cell_height
                    orig_w = cell_width
                    orig_h = cell_height
                    cv2.rectangle(crop_demo, (orig_x, orig_y), (orig_x + orig_w, orig_y + orig_h), (255, 0, 0), 2)  # 蓝色：原始网格
                    
                    # 绘制实际裁剪区域
                    cv2.rectangle(crop_demo, (cell_x, cell_y), (cell_x + cell_w, cell_y + cell_h), (0, 255, 0), 2)  # 绿色：实际裁剪区域
                    
                    # 添加文字说明
                    cv2.putText(crop_demo, "Blue: Grid", (orig_x, orig_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                    cv2.putText(crop_demo, "Green: Crop", (cell_x, cell_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    cv2.imwrite(os.path.join(cells_dir, f'cell_{row}_{col}_00_crop_area_demo.jpg'), crop_demo)
                
                sudoku_row.append(digit)
                
                # 如果找到了数字且通过尺寸检查，记录位置信息
                if result_info['found'] and result_info['position']:
                    # 将相对于调整后单元格的坐标转换为相对于原始网格的坐标
                    x_min, y_min, x_max, y_max = result_info['position']
                    
                    # 缩放回原始单元格大小
                    scale_x = cell_w / 84
                    scale_y = cell_h / 84
                    
                    actual_x_min = int(cell_x + x_min * scale_x)
                    actual_y_min = int(cell_y + y_min * scale_y)
                    actual_x_max = int(cell_x + x_max * scale_x)
                    actual_y_max = int(cell_y + y_max * scale_y)
                    
                    digit_positions.append({
                        'row': row,
                        'col': col,
                        'digit': digit,  # 这里的digit已经是经过尺寸检查后的结果
                        'bbox': (actual_x_min, actual_y_min, actual_x_max, actual_y_max),
                        'confidence': result_info['confidence'],
                        'size_filtered': not size_check_passed,  # 标记是否被尺寸条件过滤
                        'filter_reason': filter_reason if not size_check_passed else ""
                    })
            else:
                sudoku_row.append(0)

        sudoku_array.append(sudoku_row)
        print(f"第 {row + 1} 行识别完成")

    # 在网格图像上绘制数字边界框
    for pos_info in digit_positions:
        x_min, y_min, x_max, y_max = pos_info['bbox']
        digit = pos_info['digit']
        confidence = pos_info['confidence']
        size_filtered = pos_info.get('size_filtered', False)
        
        # 根据识别结果选择颜色
        if digit > 0:
            # 识别到数字用红色框
            color = (0, 0, 255)  # 红色
            thickness = 2
        elif size_filtered:
            # 被尺寸条件过滤的用橙色框
            color = (0, 165, 255)  # 橙色
            thickness = 1
        else:
            # 检测到内容但未识别出数字用黄色框
            color = (0, 255, 255)  # 黄色
            thickness = 1
        
        # 绘制边界框
        cv2.rectangle(grid_with_digit_boxes, (x_min, y_min), (x_max, y_max), color, thickness)
        
        # 在框旁边添加数字标签（如果识别到了数字）
        if digit > 0:
            label = f"{digit}"
        elif size_filtered:
            label = "S"  # S表示被尺寸条件过滤（Size filtered）
        else:
            label = "?"  # ?表示检测到内容但未识别
            
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        
        # 在边界框上方显示标签
        text_x = x_min
        text_y = y_min - 5 if y_min - 5 > 0 else y_max + 15
        
        cv2.putText(grid_with_digit_boxes, label, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

    # 保存带有数字边界框的图像
    cv2.imwrite(os.path.join(output_dir, '9_grid_with_digit_boxes.jpg'), grid_with_digit_boxes)
    print(f"数字边界框图像已保存到 {os.path.join(output_dir, '9_grid_with_digit_boxes.jpg')}")
    
    # 打印识别到的数字位置统计
    recognized_digits = [pos for pos in digit_positions if pos['digit'] > 0]
    detected_content = [pos for pos in digit_positions if pos['digit'] == 0 and not pos.get('size_filtered', False)]
    
    print(f"识别到 {len(recognized_digits)} 个数字，检测到 {len(detected_content)} 个未识别内容")
    print(f"因尺寸条件过滤掉 {filtered_count} 个可能的误识别")
    print(f"所有单元格的处理步骤图像已保存到: {cells_dir}")
    
    return sudoku_array


def preprocess_cell(cell):
    """对单个单元格进行预处理"""
    # 高斯模糊
    blurred = cv2.GaussianBlur(cell, (3, 3), 0)
    
    # 形态学操作去除噪点
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    
    return opened


def recognize_sudoku_digits(image_path, threshold_param, output_dir):
    """主函数：识别数独图片中的数字"""
    # 读取图像
    image = cv2.imread(image_path)

    if image is None:
        print(f"无法读取图像: {image_path}")
        return None

    # 保存原始图像
    cv2.imwrite(os.path.join(output_dir, '0_original_sudoku_image.jpg'), image)
    print(f"原始图像已保存到 {output_dir}")

    # 预处理图像
    processed = preprocess_image(image, threshold_param, output_dir)

    # 找到数独网格
    grid_corners = find_sudoku_grid(processed, output_dir)

    # 从网格中提取数字
    sudoku_digits = extract_digits_from_grid(processed, grid_corners, output_dir)

    return sudoku_digits


def save_sudoku_result_to_txt(sudoku_array, output_dir, threshold_param):
    """将数独识别结果保存到txt文件"""
    txt_filename = os.path.join(output_dir, f'sudoku_result_threshold_{threshold_param}.txt')
    
    # 获取标准答案
    standard_array = convert_res_to_array(res)
    
    # 计算准确率
    accuracy_stats = calculate_accuracy(sudoku_array, standard_array)
    
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write(f"数独识别结果 (阈值参数: {threshold_param})\n")
        f.write("=" * 60 + "\n\n")
        
        # 标准答案显示
        f.write("标准答案:\n")
        for i, row in enumerate(standard_array):
            if i % 3 == 0 and i != 0:
                f.write("------+-------+------\n")

            row_str = ""
            for j, digit in enumerate(row):
                if j % 3 == 0 and j != 0:
                    row_str += "| "

                if digit == 0:
                    row_str += ". "
                else:
                    row_str += f"{digit} "

            f.write(row_str + "\n")
        
        f.write("\n" + "=" * 60 + "\n")
        
        # 识别结果显示
        f.write("识别结果:\n")
        for i, row in enumerate(sudoku_array):
            if i % 3 == 0 and i != 0:
                f.write("------+-------+------\n")

            row_str = ""
            for j, digit in enumerate(row):
                if j % 3 == 0 and j != 0:
                    row_str += "| "

                if digit == 0:
                    row_str += ". "
                else:
                    row_str += f"{digit} "

            f.write(row_str + "\n")
        
        f.write("\n" + "=" * 60 + "\n")
        
        # 对比显示（标记错误）
        f.write("对比结果 (✓=正确, ✗=错误):\n")
        for i, row in enumerate(sudoku_array):
            if i % 3 == 0 and i != 0:
                f.write("------+-------+------\n")

            row_str = ""
            for j, digit in enumerate(row):
                if j % 3 == 0 and j != 0:
                    row_str += "| "

                if digit == standard_array[i][j]:
                    if digit == 0:
                        row_str += ".✓"
                    else:
                        row_str += f"{digit}✓"
                else:
                    if digit == 0:
                        row_str += ".✗"
                    else:
                        row_str += f"{digit}✗"

            f.write(row_str + "\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("二维数组格式:\n")
        f.write("标准答案: \n")
        for row in standard_array:
            f.write(str(row) + "\n")
        f.write("\n识别结果: \n")
        for row in sudoku_array:
            f.write(str(row) + "\n")
        
        f.write("\n" + "=" * 60 + "\n")
        # 准确率统计信息
        f.write("准确率统计:\n")
        f.write(f"总体准确率: {accuracy_stats['correct_cells']}/{accuracy_stats['total_cells']} = {accuracy_stats['overall_accuracy']:.1f}%\n")
        f.write(f"数字识别准确率: {accuracy_stats['correct_filled']}/{accuracy_stats['total_filled']} = {accuracy_stats['filled_accuracy']:.1f}%\n")
        f.write(f"空格识别准确率: {accuracy_stats['correct_empty']}/{accuracy_stats['total_empty']} = {accuracy_stats['empty_accuracy']:.1f}%\n")
        
        # 识别统计
        total_recognized = sum(sum(1 for digit in row if digit != 0) for row in sudoku_array)
        f.write(f"\n识别统计:\n")
        f.write(f"识别到的数字总数: {total_recognized}\n")
        f.write(f"应有数字总数: {accuracy_stats['total_filled']}\n")
        f.write(f"识别完成度: {total_recognized/accuracy_stats['total_filled']*100:.1f}%\n")
    
    print(f"识别结果已保存到: {txt_filename}")
    return accuracy_stats

def main():
    """主程序"""
    # 记录程序开始时间
    start_time = time.time()
    print("=" * 60)
    print("数独识别程序开始运行...")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print("=" * 60)
    
    image_path = "image.png"
    
    # 要测试的阈值化参数数组（去掉参数9，加入参数13）
    threshold_params = [11, 13, 15,17, 19]
    
    print("开始使用不同阈值化参数识别数独图片...")
    print(f"将测试参数: {threshold_params}")
    print(f"标准答案: {res}")
    
    # 用于保存所有参数的准确率统计
    all_accuracy_stats = {}
    
    for param in threshold_params:
        print(f"\n正在处理参数 {param}...")
        
        # 为每个参数创建单独的输出目录
        output_dir = f"process_images_threshold_{param}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 识别数独数字
        result = recognize_sudoku_digits(image_path, param, output_dir)
        
        if result is not None:
            # 将识别结果保存到txt文件并获取准确率统计
            accuracy_stats = save_sudoku_result_to_txt(result, output_dir, param)
            all_accuracy_stats[param] = accuracy_stats
            
            # 在控制台显示详细结果
            total_recognized = sum(sum(1 for digit in row if digit != 0) for row in result)
            print(f"参数 {param}:")
            print(f"  总体准确率: {accuracy_stats['overall_accuracy']:.1f}%")
            print(f"  数字识别准确率: {accuracy_stats['filled_accuracy']:.1f}%")
            print(f"  空格识别准确率: {accuracy_stats['empty_accuracy']:.1f}%")
            print(f"  识别数字总数: {total_recognized}")
        else:
            print(f"参数 {param}: 识别失败")
        
        print(f"参数 {param} 处理完成，结果保存在 {output_dir} 文件夹中")
    
    print(f"\n所有参数处理完成！")
    print("\n准确率汇总:")
    print("参数\t总体准确率\t数字准确率\t空格准确率")
    print("-" * 50)
    for param in threshold_params:
        if param in all_accuracy_stats:
            stats = all_accuracy_stats[param]
            print(f"{param}\t{stats['overall_accuracy']:.1f}%\t\t{stats['filled_accuracy']:.1f}%\t\t{stats['empty_accuracy']:.1f}%")
    
    print("\n你可以对比以下文件夹中的结果：")
    for param in threshold_params:
        print(f"- process_images_threshold_{param}")
        print(f"  └── sudoku_result_threshold_{param}.txt")
    
    # 记录程序结束时间并计算总耗时
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 60)
    print("程序运行完成！")
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"总耗时: {total_time:.2f} 秒")
    
    # 将耗时转换为更易读的格式
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    
    if hours > 0:
        print(f"总耗时(详细): {hours}小时 {minutes}分钟 {seconds:.2f}秒")
    elif minutes > 0:
        print(f"总耗时(详细): {minutes}分钟 {seconds:.2f}秒")
    else:
        print(f"总耗时(详细): {seconds:.2f}秒")
    print("=" * 60)

if __name__ == "__main__":
    main()
