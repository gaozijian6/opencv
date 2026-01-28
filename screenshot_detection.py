# -*- coding: utf-8 -*-
"""截图检测模块 - 用于处理截图类型的数独图片"""
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


def preprocess_image(image, threshold_param, kernel_size, output_dir, use_sharpen=True, blur_kernel_size=3):
    """预处理图像以提高OCR识别准确率 - 不对整体图片进行高斯模糊"""
    # 首先确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 移除整体高斯模糊，直接使用灰度图进行后续处理
    processed_image = gray
    
    # 根据参数决定是否添加锐化效果（如果启用的话）
    if use_sharpen:
        sharpen_kernel = np.array([[ 0, -1,  0],
                                  [-1,  5, -1],
                                  [ 0, -1,  0]], dtype=np.float32)
        sharpened = cv2.filter2D(processed_image, -1, sharpen_kernel)
        processed_image = sharpened
        # 检查锐化图像保存是否成功 - 修复文件名
        sharpen_path = os.path.join(output_dir, '2_3_sharpened.jpg')
        success = save_image_with_fallback(sharpened, sharpen_path)
        if success:
            print(f"✓ 锐化图像保存成功: {sharpen_path}")
        else:
            print(f"✗ 锐化图像保存失败: {sharpen_path}")
    
    # 中值滤波，去除椒盐噪声
    median_filtered = cv2.medianBlur(processed_image, 3)

    # 自适应阈值化 - 使用传入的参数
    thresh = cv2.adaptiveThreshold(
        median_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, threshold_param, 2
    )

    # 直接使用阈值化后的图像作为最终处理结果
    processed = thresh

    # 保存预处理结果到指定文件夹，并检查每个保存操作是否成功
    save_results = []
    
    # 更新保存的图像列表，移除高斯模糊图像，改为保存直接处理的图像
    images_to_save = [
        ('1_original_gray.jpg', gray),
        ('2_no_blur_processed.jpg', processed_image),  # 无模糊的处理图像
        ('2_5_median_filtered.jpg', median_filtered),
        ('3_adaptive_threshold.jpg', thresh),
        ('4_final_processed.jpg', processed)
    ]
    
    for filename, img in images_to_save:
        filepath = os.path.join(output_dir, filename)
        
        # 确保图像数据不为空
        if img is None:
            print(f"✗ {filename} 图像数据为空")
            save_results.append((filename, False))
            continue
            
        # 确保图像数据类型正确
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        # 使用带有回退机制的保存函数
        success = save_image_with_fallback(img, filepath)
        save_results.append((filename, success))
        
        if success:
            print(f"✓ {filename} 保存成功")
        else:
            print(f"✗ {filename} 保存失败: {filepath}")
    
    # 汇总保存结果
    successful_saves = sum(1 for _, success in save_results if success)
    total_saves = len(save_results)
    
    sharpen_status = "已添加锐化效果" if use_sharpen else "未使用锐化"
    print(f"预处理图像保存结果: {successful_saves}/{total_saves} 成功")
    print(f"保存位置: {output_dir}")
    print(f"参数: threshold_param={threshold_param}, kernel_size={kernel_size}x{kernel_size}, 整体模糊: 已移除, 方格模糊={blur_kernel_size}x{blur_kernel_size}, {sharpen_status}")
    
    return processed


def detect_image_brightness(gray):
    """检测图像的平均亮度
    
    返回:
        brightness_level: 'dark' (< 80), 'medium' (80-160), 'bright' (> 160)
        mean_value: 平均灰度值
    """
    mean_value = np.mean(gray)
    if mean_value < 80:
        return 'dark', mean_value
    elif mean_value < 160:
        return 'medium', mean_value
    else:
        return 'bright', mean_value


def preprocess_for_grid_detection(image, output_dir=None):
    """专门用于网格检测的预处理，保护细边框，根据图片亮度自适应调整参数
    
    参数:
        image: 输入图像
        output_dir: 输出目录（可选），如果提供则保存增强后的图片
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 检测图片亮度
    brightness_level, mean_value = detect_image_brightness(gray)
    print(f"   图片亮度检测: {brightness_level} (平均灰度值: {mean_value:.1f})")
    
    # 根据亮度调整预处理策略
    if brightness_level == 'dark':
        # 暗色图片：增强对比度和亮度
        # 1. 使用CLAHE (对比度受限的自适应直方图均衡化) 增强对比度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 保存CLAHE增强后的图片
        if output_dir:
            enhanced_path = os.path.join(output_dir, '4_1_clahe_enhanced.jpg')
            save_image_with_fallback(enhanced, enhanced_path)
            print(f"   ✓ CLAHE增强图片已保存: {enhanced_path}")
        
        # 2. 轻微的高斯模糊去噪
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # 保存模糊后的图片
        if output_dir:
            blurred_path = os.path.join(output_dir, '4_2_blurred_after_clahe.jpg')
            save_image_with_fallback(blurred, blurred_path)
            print(f"   ✓ 模糊后图片已保存: {blurred_path}")
        
        # 3. 对暗色图片使用更小的blockSize和更小的C值，提高敏感度
        thresh = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11,  # 较小的blockSize，对暗色图片更敏感
            3    # 较小的C值，提高阈值敏感度
        )
        
    elif brightness_level == 'medium':
        # 中等亮度：使用标准参数
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            15,  # 标准blockSize
            5    # 标准C值
        )
        
    else:  # bright
        # 亮色图片：使用原有参数（效果已经很好）
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            15,  # 减小 blockSize，对细边框更敏感
            5    # 增大 C 值，使阈值更保守
        )
    
    return thresh


def validate_and_complete_rectangle(pts, angle_tolerance=5, max_length_ratio=1.03):
    """从4个点中选取3个点，验证是否有水平和垂直边，并计算第4个点
    
    参数:
        pts: 4个点的坐标，形状为 (4, 2)
        angle_tolerance: 角度容差（度），默认5度
        max_length_ratio: 长边/短边的最大比值，默认1.03（长边不能比短边长超过3%）
    
    返回:
        (is_valid, corrected_pts, max_ratio) - (是否有效, 修正后的4个点, 最大比例)
        如果无效，返回 (False, None, max_ratio)
    """
    if len(pts) != 4:
        return False, None, 0
    
    pts = np.array(pts, dtype=np.float32)
    
    # 计算向量与水平/垂直线的角度
    def angle_to_horizontal(v):
        """计算向量与水平线的角度（度）"""
        angle = np.arctan2(v[1], v[0]) * 180 / np.pi
        # 归一化到 -90 到 90 度
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        return abs(angle)
    
    def angle_to_vertical(v):
        """计算向量与垂直线的角度（度）"""
        angle = np.arctan2(v[0], v[1]) * 180 / np.pi
        # 归一化到 -90 到 90 度
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        return abs(angle)
    
    def distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    # 尝试不同的3个点组合（3个点可以形成2条边）
    # 组合：0-1-2, 0-1-3, 0-2-3, 1-2-3
    combinations = [
        ([0, 1, 2], 3),  # 使用点0,1,2，计算点3
        ([0, 1, 3], 2),  # 使用点0,1,3，计算点2
        ([0, 2, 3], 1),  # 使用点0,2,3，计算点1
        ([1, 2, 3], 0),  # 使用点1,2,3，计算点0
    ]
    
    max_ratio = 0  # 记录所有尝试中的最大比例
    valid_results = []  # 存储所有符合条件的结果
    
    for indices, missing_idx in combinations:
        p0, p1, p2 = pts[indices]
        
        # 计算两条边
        v1 = p1 - p0  # 第一条边
        v2 = p2 - p1  # 第二条边
        
        # 计算边的长度
        len1 = distance(p0, p1)
        len2 = distance(p1, p2)
        
        # 检查是否有水平和垂直边
        # 情况1: v1是水平，v2是垂直
        angle1_h = angle_to_horizontal(v1)
        angle2_v = angle_to_vertical(v2)
        if angle1_h <= angle_tolerance and angle2_v <= angle_tolerance:
            # 检查长度比例：长边/短边 <= max_length_ratio
            ratio = max(len1, len2) / min(len1, len2)
            max_ratio = max(max_ratio, ratio)
            if ratio <= max_length_ratio:
                # 计算第4个点
                p3_calc = p0 + v1 + v2
                corrected_pts = pts.copy()
                corrected_pts[missing_idx] = p3_calc
                # 计算质量分数：角度偏差总和 + 比例偏差（越小越好）
                quality_score = angle1_h + angle2_v + (ratio - 1.0) * 10
                valid_results.append((quality_score, corrected_pts, ratio))
        
        # 情况2: v1是垂直，v2是水平
        angle1_v = angle_to_vertical(v1)
        angle2_h = angle_to_horizontal(v2)
        if angle1_v <= angle_tolerance and angle2_h <= angle_tolerance:
            # 检查长度比例：长边/短边 <= max_length_ratio
            ratio = max(len1, len2) / min(len1, len2)
            max_ratio = max(max_ratio, ratio)
            if ratio <= max_length_ratio:
                # 计算第4个点
                p3_calc = p0 + v1 + v2
                corrected_pts = pts.copy()
                corrected_pts[missing_idx] = p3_calc
                # 计算质量分数：角度偏差总和 + 比例偏差（越小越好）
                quality_score = angle1_v + angle2_h + (ratio - 1.0) * 10
                valid_results.append((quality_score, corrected_pts, ratio))
        
        # 情况3: 尝试其他边的组合（p0-p2 和 p1-p2）
        v3 = p2 - p0  # 对角线
        # 检查 v1水平，v3垂直
        angle1_h = angle_to_horizontal(v1)
        angle3_v = angle_to_vertical(v3)
        if angle1_h <= angle_tolerance and angle3_v <= angle_tolerance:
            len3 = distance(p0, p2)
            ratio = max(len1, len3) / min(len1, len3)
            max_ratio = max(max_ratio, ratio)
            if ratio <= max_length_ratio:
                p3_calc = p1 + v3 - v1
                corrected_pts = pts.copy()
                corrected_pts[missing_idx] = p3_calc
                # 计算质量分数：角度偏差总和 + 比例偏差（越小越好）
                quality_score = angle1_h + angle3_v + (ratio - 1.0) * 10
                valid_results.append((quality_score, corrected_pts, ratio))
        
        # 情况4: v1垂直，v3水平
        angle1_v = angle_to_vertical(v1)
        angle3_h = angle_to_horizontal(v3)
        if angle1_v <= angle_tolerance and angle3_h <= angle_tolerance:
            len3 = distance(p0, p2)
            ratio = max(len1, len3) / min(len1, len3)
            max_ratio = max(max_ratio, ratio)
            if ratio <= max_length_ratio:
                p3_calc = p1 + v3 - v1
                corrected_pts = pts.copy()
                corrected_pts[missing_idx] = p3_calc
                # 计算质量分数：角度偏差总和 + 比例偏差（越小越好）
                quality_score = angle1_v + angle3_h + (ratio - 1.0) * 10
                valid_results.append((quality_score, corrected_pts, ratio))
    
    # 如果有符合条件的结果，选择质量分数最好的（最小的）
    if valid_results:
        valid_results.sort(key=lambda x: x[0])  # 按质量分数排序
        best_score, best_pts, best_ratio = valid_results[0]
        print(f"   找到 {len(valid_results)} 个符合条件的结果，选择质量分数最好的 (分数: {best_score:.2f}, 比例: {best_ratio:.4f})")
        return True, best_pts, best_ratio
    
    return False, None, max_ratio


def find_sudoku_grid(image, output_dir, require_rectangle=True, angle_tolerance=5, max_length_ratio=1.03):
    """找到数独网格的轮廓
    
    参数:
        image: 输入图像
        output_dir: 输出目录
        require_rectangle: 是否要求检测到的四边形必须是矩形，默认True
        angle_tolerance: 角度容差（度），默认5度（竖边偏离垂直线，横边偏离水平线）
        max_length_ratio: 长边/短边的最大比值，默认1.03（长边不能比短边长超过3%）
    """
    # 使用专门优化的预处理来保护细边框
    processed_for_detection = preprocess_for_grid_detection(image, output_dir)
    
    # 保存用于检测的预处理图像
    detection_image = cv2.cvtColor(processed_for_detection, cv2.COLOR_GRAY2BGR)
    save_image_with_fallback(detection_image, os.path.join(output_dir, '4_5_grid_detection_preprocessed.jpg'))
    
    # 查找轮廓
    contours, _ = cv2.findContours(processed_for_detection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("⚠️ 未找到任何轮廓，尝试使用原始预处理图像")
        # 如果没找到轮廓，回退到原始预处理方法
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("❌ 仍然未找到轮廓")
            return None

    # 找到最大的矩形轮廓（应该是数独网格）
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 计算轮廓面积，过滤太小的轮廓
    image_area = image.shape[0] * image.shape[1]
    contour_area = cv2.contourArea(largest_contour)
    area_ratio = contour_area / image_area
    
    if area_ratio < 0.1:  # 如果最大轮廓面积小于图片的10%，可能不是网格
        print(f"⚠️ 最大轮廓面积过小 ({area_ratio*100:.1f}%)，尝试其他方法")
        # 尝试找面积第二大的轮廓
        if len(contours) > 1:
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for contour in sorted_contours[1:]:
                area_ratio = cv2.contourArea(contour) / image_area
                if area_ratio > 0.1:
                    largest_contour = contour
                    print(f"✓ 使用面积第二大的轮廓 ({area_ratio*100:.1f}%)")
                    break

    # 近似轮廓为矩形
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # 保存轮廓检测结果
    contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    if len(contour_image.shape) == 2:
        contour_image = cv2.cvtColor(contour_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 2)
    save_image_with_fallback(contour_image, os.path.join(output_dir, '5_largest_contour.jpg'))

    # 如果找到了四边形，检查是否满足矩形要求
    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype(np.float32)
        
        # 如果要求必须是矩形，进行验证
        if require_rectangle:
            # 使用新的验证方法：检查3个点是否有水平和垂直边
            is_valid, corrected_pts, max_ratio = validate_and_complete_rectangle(pts, angle_tolerance, max_length_ratio)
            if is_valid:
                # 保存四边形轮廓（使用修正后的点）
                quad_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
                if len(quad_image.shape) == 2:
                    quad_image = cv2.cvtColor(quad_image, cv2.COLOR_GRAY2BGR)
                # 绘制原始检测的点
                cv2.drawContours(quad_image, [approx], -1, (0, 0, 255), 2)  # 红色，原始检测
                # 绘制修正后的点
                corrected_approx = corrected_pts.astype(np.int32).reshape(-1, 1, 2)
                cv2.drawContours(quad_image, [corrected_approx], -1, (255, 0, 255), 3)  # 洋红色，修正后的
                save_image_with_fallback(quad_image, os.path.join(output_dir, '6_sudoku_grid_quad.jpg'))
                print("✓ 找到矩形数独网格（通过矩形验证，已修正第4个点）")
                return corrected_pts
            else:
                # 检查比例是否大于1.03
                if max_ratio > max_length_ratio:
                    print(f"❌ 错误：检测到的四边形长边/短边比例 ({max_ratio:.4f}) 大于允许的最大值 ({max_length_ratio:.4f})，停止执行后续操作")
                    return None
                print("⚠️ 检测到四边形但不满足矩形要求（无水平和垂直边），使用边界矩形")
                # 不满足矩形要求，继续执行下面的boundingRect逻辑
        else:
            # 不要求矩形，但仍需要检查比例
            is_valid, corrected_pts, max_ratio = validate_and_complete_rectangle(pts, angle_tolerance, max_length_ratio)
            if max_ratio > max_length_ratio:
                print(f"❌ 错误：检测到的四边形长边/短边比例 ({max_ratio:.4f}) 大于允许的最大值 ({max_length_ratio:.4f})，停止执行后续操作")
                return None
            
            quad_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
            if len(quad_image.shape) == 2:
                quad_image = cv2.cvtColor(quad_image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(quad_image, [approx], -1, (0, 0, 255), 3)
            save_image_with_fallback(quad_image, os.path.join(output_dir, '6_sudoku_grid_quad.jpg'))
            print("✓ 找到四边形数独网格（未验证矩形）")
            return approx.reshape(4, 2)
    else:
        # 如果没有找到完美的四边形，使用边界矩形
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 检查边界矩形的宽高比
        ratio = max(w, h) / min(w, h)
        if ratio > max_length_ratio:
            print(f"❌ 错误：边界矩形的长边/短边比例 ({ratio:.4f}) 大于允许的最大值 ({max_length_ratio:.4f})，停止执行后续操作")
            return None
        
        rect_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
        if len(rect_image.shape) == 2:
            rect_image = cv2.cvtColor(rect_image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(rect_image, (x, y), (x + w, y + h), (255, 0, 0), 3)
        save_image_with_fallback(rect_image, os.path.join(output_dir, '6_sudoku_grid_rectangle.jpg'))
        print("⚠️ 使用矩形边界作为数独网格（未找到完美四边形）")
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])


def extract_cells_screenshot(image, cells_dir):
    """截图检测主函数 - 只裁切单元格，不做识别
    
    参数:
        image: 输入图像（BGR格式）
        cells_dir: 输出目录
    
    返回:
        (success, grid_roi, w, h) - (是否成功, 网格区域, 宽度, 高度)
    """
    print(f"   截图类型，使用轮廓检测方法")
    
    # 固定阈值化参数
    threshold_param = 19
    
    # 创建临时输出目录（用于网格检测的中间文件）
    temp_output = "temp_grid_detection"
    if not os.path.exists(temp_output):
        os.makedirs(temp_output)
    
    # 预处理图像（用于网格检测）
    processed = preprocess_image(image, threshold_param, 3, temp_output, False, 3)
    
    # 找到数独网格
    grid_corners = find_sudoku_grid(processed, cells_dir)
    if grid_corners is None:
        print(f"未找到网格")
        return False, None, 0, 0
    
    # 使用原始彩色图像进行裁剪
    source_image = image
    if len(source_image.shape) == 2:
        source_image = cv2.cvtColor(source_image, cv2.COLOR_GRAY2BGR)
    
    # 直接使用边界矩形裁剪，保持原始形状，不做透视变换
    x, y, w, h = cv2.boundingRect(grid_corners)
    print(f"   使用边界矩形裁剪（保持原始形状，无透视变换）")
    print(f"   位置: ({x}, {y})")
    print(f"   尺寸: {w} x {h} 像素")
    
    # 在原图上标注检测到的矩形网格
    original_with_rect = source_image.copy()
    if len(original_with_rect.shape) == 2:
        original_with_rect = cv2.cvtColor(original_with_rect, cv2.COLOR_GRAY2BGR)
    
    # 绘制检测到的矩形边界
    rect_color = (0, 255, 0)  # 绿色
    rect_thickness = 3
    cv2.rectangle(original_with_rect, (x, y), (x + w, y + h), rect_color, rect_thickness)
    
    # 如果检测到的是四边形，也绘制四边形轮廓
    if len(grid_corners) == 4:
        pts = grid_corners.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(original_with_rect, [pts], True, (255, 0, 0), 2)  # 蓝色，四边形轮廓
    
    # 添加标注文字
    label_text = f"Grid: {w}x{h} at ({x},{y})"
    cv2.putText(original_with_rect, label_text, (x, y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, rect_color, 2)
    
    # 保存原图标注图片
    original_rect_filename = os.path.join(cells_dir, 'original_with_rect.jpg')
    save_image_with_fallback(original_with_rect, original_rect_filename)
    print(f"✓ 原图标注图片已保存: {original_rect_filename}")
    
    # 裁切网格区域
    grid_roi = source_image[y:y+h, x:x+w]
    
    return True, grid_roi, w, h

