# -*- coding: utf-8 -*-
"""照片检测模块 - 用于处理现实照片类型的数独图片"""
import cv2
import numpy as np
import os


def save_image_with_fallback(image, filepath):
    """使用多种方法尝试保存图像，处理中文路径问题"""
    # 方法1: 直接使用cv2.imwrite
    try:
        success = cv2.imwrite(filepath, image)
        if success and os.path.exists(filepath):
            return True
    except Exception as e:
        print(f"方法1保存失败: {e}")
    
    # 方法2: 使用cv2.imencode然后写入文件 (处理中文路径)
    try:
        # 获取文件扩展名
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            ext = '.jpg'
        
        # 编码图像
        success, encoded_image = cv2.imencode(ext, image)
        if success:
            # 写入文件
            with open(filepath, 'wb') as f:
                f.write(encoded_image.tobytes())
            
            # 验证文件是否存在且不为空
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                return True
    except Exception as e:
        print(f"方法2保存失败: {e}")
    
    # 方法3: 尝试使用临时文件名，然后重命名
    try:
        import tempfile
        temp_dir = os.path.dirname(filepath)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', dir=temp_dir)
        temp_path = temp_file.name
        temp_file.close()
        
        # 保存到临时文件
        success = cv2.imwrite(temp_path, image)
        if success and os.path.exists(temp_path):
            # 移动到目标位置
            import shutil
            shutil.move(temp_path, filepath)
            
            if os.path.exists(filepath):
                return True
        
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    except Exception as e:
        print(f"方法3保存失败: {e}")
    
    return False


def is_dark_mode(image, threshold=128):
    """判断图片是否是暗色模式
    
    原理：计算灰度图的平均亮度
    如果平均亮度低于阈值，认为是暗色模式
    
    参数:
        image: 输入图像（BGR格式或灰度图）
        threshold: 亮度阈值，默认128
    
    返回:
        (is_dark, mean_brightness): (是否是暗色模式, 平均亮度)
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 计算平均亮度
    mean_brightness = np.mean(gray)
    
    # 判断是否是暗色模式
    is_dark = mean_brightness < threshold
    
    return is_dark, mean_brightness


def order_points(pts):
    """对4个角点排序：左上、右上、右下、左下"""
    rect = np.zeros((4, 2), dtype="float32")
    
    # 计算和：左上点的和最小，右下点的和最大
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下
    
    # 计算差：右上点的差最小，左下点的差最大
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下
    
    return rect


def extract_corners_from_contour_hough(contour, image_shape):
    """通过霍夫直线检测从轮廓中提取四个角点
    
    参数:
        contour: 轮廓点集
        image_shape: 图像尺寸 (height, width)
    
    返回:
        corners: 四个角点的数组，形状为 (4, 1, 2)，如果失败返回 None
    """
    # 1. 绘制轮廓到空白图
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, 2)
    
    # 2. 霍夫直线检测
    lines = cv2.HoughLinesP(mask, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
    
    if lines is None or len(lines) < 4:
        return None
    
    # 3. 计算每条直线的长度和角度，按角度分组
    horizontal_lines = []  # 接近水平的线（角度在 -45° 到 45° 之间）
    vertical_lines = []    # 接近垂直的线（角度在 45° 到 135° 之间）
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        # 计算角度（转换为度）
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angle = abs(angle)
        
        # 归一化角度到 0-90 度
        if angle > 90:
            angle = 180 - angle
        
        if angle < 45:  # 接近水平
            horizontal_lines.append((length, line[0], (x1 + x2) / 2, (y1 + y2) / 2))
        else:  # 接近垂直
            vertical_lines.append((length, line[0], (x1 + x2) / 2, (y1 + y2) / 2))
    
    # 需要至少2条水平线和2条垂直线
    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        return None
    
    # 4. 找到最长的2条水平线和2条垂直线，并根据位置确定上下左右
    horizontal_lines.sort(reverse=True, key=lambda x: x[0])
    vertical_lines.sort(reverse=True, key=lambda x: x[0])
    
    # 根据y坐标确定上边和下边
    top_horizontal = None
    bottom_horizontal = None
    for length, line, cx, cy in horizontal_lines[:min(5, len(horizontal_lines))]:
        if top_horizontal is None or cy < top_horizontal[3]:
            top_horizontal = (length, line, cx, cy)
        if bottom_horizontal is None or cy > bottom_horizontal[3]:
            bottom_horizontal = (length, line, cx, cy)
    
    # 根据x坐标确定左边和右边
    left_vertical = None
    right_vertical = None
    for length, line, cx, cy in vertical_lines[:min(5, len(vertical_lines))]:
        if left_vertical is None or cx < left_vertical[2]:
            left_vertical = (length, line, cx, cy)
        if right_vertical is None or cx > right_vertical[2]:
            right_vertical = (length, line, cx, cy)
    
    if top_horizontal is None or bottom_horizontal is None or left_vertical is None or right_vertical is None:
        return None
    
    # 5. 计算直线交点
    def line_intersection(line1, line2):
        """计算两条线段的交点"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # 转换为直线方程：ax + by + c = 0
        a1 = y2 - y1
        b1 = x1 - x2
        c1 = x2 * y1 - x1 * y2
        
        a2 = y4 - y3
        b2 = x3 - x4
        c2 = x4 * y3 - x3 * y4
        
        # 计算行列式
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-10:  # 平行线
            return None
        
        # 计算交点
        x = (b1 * c2 - b2 * c1) / det
        y = (a2 * c1 - a1 * c2) / det
        
        return (int(x), int(y))
    
    # 计算四个交点
    top_line = top_horizontal[1]
    bottom_line = bottom_horizontal[1]
    left_line = left_vertical[1]
    right_line = right_vertical[1]
    
    # 左上角：上水平线与左垂直线的交点
    tl = line_intersection(top_line, left_line)
    if tl is None:
        tl = line_intersection(top_line, right_line)
    if tl is None:
        return None
    
    # 右上角：上水平线与右垂直线的交点
    tr = line_intersection(top_line, right_line)
    if tr is None:
        tr = line_intersection(top_line, left_line)
    if tr is None:
        return None
    
    # 右下角：下水平线与右垂直线的交点
    br = line_intersection(bottom_line, right_line)
    if br is None:
        br = line_intersection(bottom_line, left_line)
    if br is None:
        return None
    
    # 左下角：下水平线与左垂直线的交点
    bl = line_intersection(bottom_line, left_line)
    if bl is None:
        bl = line_intersection(bottom_line, right_line)
    if bl is None:
        return None
    
    corners = [tl, tr, br, bl]
    
    # 转换为 OpenCV 轮廓格式
    corners_array = np.array([[list(c)] for c in corners], dtype=np.int32)
    
    return corners_array


def extract_cells_photo(image, cells_dir):
    """照片检测主函数 - 检测现实照片中的数独网格并进行透视变换矫正
    
    流程：
    1. 预处理去噪
    2. 检测外边框
    3. 透视变换矫正
    
    参数:
        image: 输入图像（BGR格式）
        cells_dir: 输出目录
    
    返回:
        (success, warped_image, w, h) - (是否成功, 矫正后的图像, 宽度, 高度)
    """
    # 从cells_dir中提取图片信息用于调试
    cells_dir_name = os.path.basename(os.path.normpath(cells_dir))
    print(f"========================================")
    print(f"正在检测图片: {cells_dir_name}")
    print(f"输出目录: {cells_dir}")
    print(f"========================================")
    print("   照片类型，使用透视变换矫正流程")
    print("   开始现实照片检测流程...")
    
    # 0. 判断是否是暗色模式
    is_dark, mean_brightness = is_dark_mode(image)
    print(f"   暗色模式检测: {'是' if is_dark else '否'} (平均亮度: {mean_brightness:.2f})")
    
    # 1. 预处理（GitHub通用方案）
    print("   步骤1: 预处理")
    # 转灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    save_image_with_fallback(gray, os.path.join(cells_dir, 'photo_1_gray.jpg'))
    
    # 根据暗色模式调整预处理策略（参考screenshot_detection.py的实现）
    if is_dark:
        # 暗色模式：使用CLAHE增强对比度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        save_image_with_fallback(enhanced, os.path.join(cells_dir, 'photo_1_5_clahe_enhanced.jpg'))
        
        # 高斯模糊（使用较小的核，保护细节）
        blurred = cv2.GaussianBlur(enhanced, (9, 9), 0)
        save_image_with_fallback(blurred, os.path.join(cells_dir, 'photo_2_blurred.jpg'))
        
        # 自适应阈值（暗色模式：使用THRESH_BINARY_INV，但使用更小的blockSize和C值提高敏感度）
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,  # 反转：数独网格为白色
            11,  # 较小的blockSize，对暗色图片更敏感
            3    # 较小的C值，提高阈值敏感度
        )
    else:
        # 亮色模式：标准处理流程
        # 高斯模糊（GitHub通用参数：9x9）
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        save_image_with_fallback(blurred, os.path.join(cells_dir, 'photo_2_blurred.jpg'))
        
        # 自适应阈值（亮色模式：使用THRESH_BINARY_INV）
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,  # 反转：数独网格为白色
            11, 2
        )
    save_image_with_fallback(thresh, os.path.join(cells_dir, 'photo_3_thresh.jpg'))
    
    # 形态学操作（GitHub通用方案：膨胀，连接断开的线条）
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    save_image_with_fallback(thresh, os.path.join(cells_dir, 'photo_3_5_morphology.jpg'))
    
    print("   ✓ 预处理完成")
    
    # 2. 检测内边框（避免图标干扰）
    print("   步骤2: 检测内边框")
    
    # 找轮廓（使用RETR_CCOMP获取所有轮廓，包括内部轮廓）
    contours, hierarchy = cv2.findContours(
        thresh,
        cv2.RETR_CCOMP,  # 获取所有轮廓，包括内部轮廓
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        print("   ❌ 未找到任何轮廓")
        return False, None, 0, 0
    
    print(f"   ✓ 找到 {len(contours)} 个轮廓")
    
    # 获取图片尺寸，用于宽度过滤
    img_height, img_width = image.shape[:2]
    min_width = img_width * 0.5  # 最小宽度为图片宽度的一半
    max_aspect_ratio = 1.15  # 最大宽高比：数独网格应该接近正方形
    print(f"   图片尺寸: {img_width}x{img_height}, 最小宽度要求: {min_width:.0f}, 最大宽高比: {max_aspect_ratio}")
    
    # 保存所有轮廓的调试图片
    all_contours_image = image.copy()
    cv2.drawContours(all_contours_image, contours, -1, (0, 255, 0), 2)
    save_image_with_fallback(all_contours_image, os.path.join(cells_dir, 'photo_3_5_all_contours.jpg'))
    
    # 按面积排序所有轮廓
    contour_areas = [(i, cv2.contourArea(contour)) for i, contour in enumerate(contours)]
    contour_areas = sorted(contour_areas, key=lambda x: x[1], reverse=True)
    
    sudoku_contour = None
    original_contour_for_hough = None  # 保存原始轮廓，用于霍夫检测
    
    # 策略：找到最大的外轮廓，然后在其内部找最大的内轮廓（子轮廓）
    # hierarchy结构：[Next, Previous, First_Child, Parent]
    # 外轮廓的Parent为-1，内轮廓的Parent为外轮廓的索引
    
    # 先找到最大的外轮廓（Parent为-1）
    largest_outer_contour_idx = None
    for idx, area in contour_areas[:20]:  # 只看前20大
        if hierarchy[0][idx][3] == -1:  # 是外轮廓（Parent为-1）
            largest_outer_contour_idx = idx
            print(f"   ✓ 找到最大外轮廓（索引{idx}，面积{area:.0f}）")
            break
    
    if largest_outer_contour_idx is not None:
        # 在这个外轮廓内部，找到最大的内轮廓（子轮廓）
        # 查找所有Parent为largest_outer_contour_idx的轮廓
        inner_contours = []
        for idx, area in contour_areas:
            if hierarchy[0][idx][3] == largest_outer_contour_idx:  # 是内轮廓
                # 检查宽度和宽高比：边界矩形宽度必须至少是图片宽度的一半，且宽高比接近1
                x, y, w, h = cv2.boundingRect(contours[idx])
                if w >= min_width:
                    ratio = max(w, h) / min(w, h)
                    if ratio <= max_aspect_ratio:
                        inner_contours.append((idx, area, w))
        
        if inner_contours:
            # 按面积排序内轮廓
            inner_contours = sorted(inner_contours, key=lambda x: x[1], reverse=True)
            print(f"   ✓ 在外轮廓内找到 {len(inner_contours)} 个符合宽度要求的内轮廓")
            
            # 尝试找到最大的内轮廓，且是四边形，且宽高比合理
            for inner_idx, inner_area, inner_width in inner_contours[:10]:  # 只看前10大内轮廓
                inner_contour = contours[inner_idx]
                # 检查宽高比
                x, y, w, h = cv2.boundingRect(inner_contour)
                ratio = max(w, h) / min(w, h)
                if ratio > max_aspect_ratio:
                    continue
                
                peri = cv2.arcLength(inner_contour, True)
                approx = cv2.approxPolyDP(inner_contour, 0.02 * peri, True)
                
                if len(approx) == 4:
                    sudoku_contour = approx
                    original_contour_for_hough = inner_contour  # 保存原始轮廓
                    print(f"   ✓ 找到内边框四边形轮廓（索引{inner_idx}，面积{inner_area:.0f}，宽度{inner_width}，宽高比{ratio:.3f}）")
                    break
            
            # 如果没找到四边形内轮廓，使用最大的内轮廓（也要检查宽高比）
            if sudoku_contour is None and inner_contours:
                largest_inner_idx = inner_contours[0][0]
                inner_contour = contours[largest_inner_idx]
                x, y, w, h = cv2.boundingRect(inner_contour)
                ratio = max(w, h) / min(w, h)
                if ratio <= max_aspect_ratio:
                    peri = cv2.arcLength(inner_contour, True)
                    approx = cv2.approxPolyDP(inner_contour, 0.02 * peri, True)
                    
                    if len(approx) == 4:
                        sudoku_contour = approx
                        original_contour_for_hough = inner_contour  # 保存原始轮廓
                        print(f"   ✓ 使用最大内轮廓作为四边形（索引{largest_inner_idx}，宽度{inner_contours[0][2]}，宽高比{ratio:.3f}）")
    
    # 如果没找到内轮廓，回退到找外轮廓中的四边形（也要检查宽度和宽高比）
    if sudoku_contour is None:
        print("   ⚠️ 未找到内边框，尝试在外轮廓中查找四边形")
        for idx, area in contour_areas[:20]:
            contour = contours[idx]
            # 检查宽度和宽高比
            x, y, w, h = cv2.boundingRect(contour)
            if w < min_width:
                continue
            
            ratio = max(w, h) / min(w, h)
            if ratio > max_aspect_ratio:
                continue
            
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4:
                sudoku_contour = approx
                original_contour_for_hough = contour  # 保存原始轮廓
                print(f"   ✓ 找到外轮廓四边形（索引{idx}，面积{area:.0f}，宽度{w}，宽高比{ratio:.3f}）")
                break
    
    # 保存检测到的轮廓
    contour_image = image.copy()
    if sudoku_contour is not None:
        cv2.drawContours(contour_image, [sudoku_contour], -1, (0, 255, 0), 3)
        # 如果找到了外轮廓，也绘制出来（用不同颜色）
        if largest_outer_contour_idx is not None:
            cv2.drawContours(contour_image, [contours[largest_outer_contour_idx]], -1, (255, 0, 0), 2)
    else:
        # 如果没有找到四边形，绘制最大轮廓
        print(f"   [else分支] sudoku_contour 是 None，绘制最大轮廓作为绿色曲线")
        if contour_areas:
            max_contour_idx = contour_areas[0][0]
            max_contour = contours[max_contour_idx]
            cv2.drawContours(contour_image, [max_contour], -1, (0, 255, 0), 2)
    save_image_with_fallback(contour_image, os.path.join(cells_dir, 'photo_4_contour.jpg'))
    
    # 如果没找到四边形，尝试使用霍夫直线检测从最大轮廓提取角点
    if sudoku_contour is None:
        print("   ⚠️ 未找到四边形轮廓，尝试使用霍夫直线检测从最大轮廓提取角点")
        if not contour_areas:
            print("   ❌ 没有可用轮廓")
            return False, None, 0, 0
        
        # 先尝试从最大轮廓使用霍夫直线检测提取角点
        max_contour_idx = contour_areas[0][0]
        max_contour = contours[max_contour_idx]
        
        # 检查最大轮廓的边界矩形是否符合要求
        x, y, w, h = cv2.boundingRect(max_contour)
        if w >= min_width:
            ratio = max(w, h) / min(w, h)
            if ratio <= max_aspect_ratio:
                corners = extract_corners_from_contour_hough(max_contour, image.shape)
                if corners is not None and len(corners) == 4:
                    sudoku_contour = corners
                    original_contour_for_hough = max_contour  # 保存原始轮廓
                    print(f"   ✓ 使用霍夫直线检测提取的角点（索引{max_contour_idx}）")
                    print(f"   [霍夫直线检测] sudoku_contour 形状: {sudoku_contour.shape}")
                    print(f"   [霍夫直线检测] sudoku_contour 内容:\n{sudoku_contour}")
        
        # 如果霍夫直线检测失败，回退到边界矩形方法
        if sudoku_contour is None:
            print("   ⚠️ 霍夫直线检测失败，尝试使用边界矩形")
            found_valid_rect = False
            for idx, area in contour_areas[:20]:
                x, y, w, h = cv2.boundingRect(contours[idx])
                
                # 检查宽度：必须至少是图片宽度的一半
                if w < min_width:
                    continue
                
                # 检查边界矩形的宽高比：数独网格应该接近正方形
                ratio = max(w, h) / min(w, h)
                if ratio > max_aspect_ratio:
                    continue
                
                # 创建边界矩形的四个角点
                sudoku_contour = np.array([
                    [[x, y]],           # 左上
                    [[x + w, y]],       # 右上
                    [[x + w, y + h]],   # 右下
                    [[x, y + h]]        # 左下
                ], dtype=np.int32)
                original_contour_for_hough = contours[idx]  # 保存原始轮廓用于霍夫检测
                print(f"   ✓ 使用边界矩形作为网格（索引{idx}）: ({x}, {y}), 尺寸: {w}x{h}，宽高比: {ratio:.3f}")
                print(f"   [边界矩形赋值] sudoku_contour 形状: {sudoku_contour.shape}")
                print(f"   [边界矩形赋值] sudoku_contour 内容:\n{sudoku_contour}")
                found_valid_rect = True
                break
            
            if not found_valid_rect:
                print(f"   ❌ 未找到符合宽度要求的边界矩形（最小宽度: {min_width:.0f}）")
                return False, None, 0, 0
    
    # 计算并打印最终确定的四边形的宽高比，如果超过3%则使用霍夫检测
    if sudoku_contour is not None:
        x, y, w, h = cv2.boundingRect(sudoku_contour)
        final_ratio = max(w, h) / min(w, h)
        print(f"   ✓ 最终确定的四边形尺寸: {w}x{h}，最大宽高比: {final_ratio:.3f}")
        
        # 如果宽高比超过3%（1.03），使用霍夫检测重新提取角点
        if final_ratio > 1.03 and original_contour_for_hough is not None:
            print(f"   ⚠️ 宽高比超过3%，使用霍夫直线检测重新提取角点")
            original_contour = sudoku_contour.copy()  # 保存原始四边形
            original_ratio = final_ratio  # 保存原始宽高比
            
            corners = extract_corners_from_contour_hough(original_contour_for_hough, image.shape)
            if corners is not None and len(corners) == 4:
                hough_contour = corners
                x_hough, y_hough, w_hough, h_hough = cv2.boundingRect(hough_contour)
                hough_ratio = max(w_hough, h_hough) / min(w_hough, h_hough)
                print(f"   [霍夫检测结果] 四边形尺寸: {w_hough}x{h_hough}，最大宽高比: {hough_ratio:.3f}")
                
                # 比较宽高比，选择较小的
                if hough_ratio < original_ratio:
                    sudoku_contour = hough_contour
                    final_ratio = hough_ratio
                    print(f"   ✓ 使用霍夫检测结果（宽高比更优: {hough_ratio:.3f} < {original_ratio:.3f}）")
                else:
                    sudoku_contour = original_contour
                    final_ratio = original_ratio
                    print(f"   ✓ 保留原四边形（宽高比更优: {original_ratio:.3f} < {hough_ratio:.3f}）")
            else:
                print(f"   ⚠️ 霍夫检测失败，继续使用原四边形")
    
    # 3. 透视变换矫正
    print("   步骤3: 透视变换矫正")
    # 获取4个角点
    pts = sudoku_contour.reshape(4, 2).astype("float32")
    
    # 对角点排序（左上、右上、右下、左下）
    rect = order_points(pts)
    
    # 计算目标正方形大小（取较大的边长）
    (tl, tr, br, bl) = rect
    
    # 计算宽度
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # 计算高度
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # 使用正方形（取较大的边）
    side = max(maxWidth, maxHeight)
    
    # 目标点（正方形的四个角）
    dst = np.array([
        [0, 0],
        [side - 1, 0],
        [side - 1, side - 1],
        [0, side - 1]
    ], dtype="float32")
    
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    
    # 对彩色图像进行透视变换
    warped = cv2.warpPerspective(image, M, (side, side))
    save_image_with_fallback(warped, os.path.join(cells_dir, 'photo_5_warped.jpg'))
    
    # 对二值化图也进行透视变换
    warped_thresh = cv2.warpPerspective(thresh, M, (side, side))
    save_image_with_fallback(warped_thresh, os.path.join(cells_dir, 'photo_5_warped_thresh.jpg'))
    
    print(f"   ✓ 透视变换完成，矫正后尺寸: {side}x{side}")
    print(f"   ✓ 现实照片检测流程完成")
    
    # 保存原图标注图片（对于照片，保存矫正后的图像）
    original_with_rect = warped.copy()
    if len(original_with_rect.shape) == 2:
        original_with_rect = cv2.cvtColor(original_with_rect, cv2.COLOR_GRAY2BGR)
    
    # 绘制矫正后的边界
    rect_color = (0, 255, 0)  # 绿色
    rect_thickness = 3
    cv2.rectangle(original_with_rect, (0, 0), (side - 1, side - 1), rect_color, rect_thickness)
    label_text = f"Warped Grid: {side}x{side}"
    cv2.putText(original_with_rect, label_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, rect_color, 2)
    
    # 保存原图标注图片
    original_rect_filename = os.path.join(cells_dir, 'original_with_rect.jpg')
    save_image_with_fallback(original_with_rect, original_rect_filename)
    print(f"✓ 原图标注图片已保存: {original_rect_filename}")
    
    # 返回原图（透视变换后的彩色图）而不是二值化图
    return True, warped, side, side

