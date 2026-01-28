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
    
    # 2. 检测外边框（GitHub通用方案）
    print("   步骤2: 检测外边框")
    
    # 找轮廓（只要外轮廓）
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,  # 只要外轮廓
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        print("   ❌ 未找到任何轮廓")
        return False, None, 0, 0
    
    print(f"   ✓ 找到 {len(contours)} 个轮廓")
    
    # 保存所有轮廓的调试图片
    all_contours_image = image.copy()
    cv2.drawContours(all_contours_image, contours, -1, (0, 255, 0), 2)
    save_image_with_fallback(all_contours_image, os.path.join(cells_dir, 'photo_3_5_all_contours.jpg'))
    
    # 按面积排序，只看前10大的轮廓（GitHub通用方案）
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    sudoku_contour = None
    
    for i, contour in enumerate(contours[:10]):  # 只看前10大
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if len(approx) == 4:
            sudoku_contour = approx
            print(f"   ✓ 找到四边形轮廓（第{i+1}大轮廓）")
            break
    
    # 保存检测到的轮廓
    contour_image = image.copy()
    if sudoku_contour is not None:
        cv2.drawContours(contour_image, [sudoku_contour], -1, (0, 255, 0), 3)
    else:
        # 如果没有找到四边形，绘制最大轮廓
        cv2.drawContours(contour_image, [contours[0]], -1, (0, 255, 0), 2)
    save_image_with_fallback(contour_image, os.path.join(cells_dir, 'photo_4_contour.jpg'))
    
    # 如果没找到四边形，使用最大轮廓的边界矩形
    if sudoku_contour is None:
        print("   ⚠️ 未找到四边形轮廓，使用边界矩形")
        x, y, w, h = cv2.boundingRect(contours[0])
        
        # 检查边界矩形的宽高比
        ratio = max(w, h) / min(w, h)
        if ratio > 1.3:
            print(f"   ❌ 边界矩形宽高比不合理 ({ratio:.3f} > 1.3)")
            return False, None, 0, 0
        
        # 创建边界矩形的四个角点
        sudoku_contour = np.array([
            [[x, y]],           # 左上
            [[x + w, y]],       # 右上
            [[x + w, y + h]],   # 右下
            [[x, y + h]]        # 左下
        ], dtype=np.int32)
        print(f"   ✓ 使用边界矩形作为网格: ({x}, {y}), 尺寸: {w}x{h}, 宽高比: {ratio:.3f}")
    
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

