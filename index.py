# -*- coding: utf-8 -*-
import cv2
import numpy as np
import easyocr
import os
import time

test={
    'image13.jpg':'964000000802954360530600294429817536706005029350269040190000603203596400640103902',
    'image12.jpg':'214087005307002008806000007589726431721493856463518972130070084940801703678234519',
    'image11.jpg':'005000062063009105000000004000006703006705008100800006801200640600000530040000820',
    'image9.jpg':'410090362020134090903600401090010040084070910031940280300401629040069130169003004',
    'image7.jpg':'546001200000600004009240610060000027070000060390000040020079406607402000904300002',
    'image5.jpg':'410090362020134000903600401090010040084070910031940280300401629040069130169003004',
    'image4.jpg':'083600000000070048074080000067108000000000027000307081402850000700000800000700100',
    'image3.jpg':'000030508203000064085040273000027080307080602820400000564010820932000051008050006',
    'image1.jpg':'000004059059210000003509600092605003804900765065000000000000500500403002000050080',
    'image.jpg':'000030508203000064085040273000027080307080602820400000564010820932000051008050006',
    'image14.jpg':'250600070800010000000000000070000001000032000000000000000007240010900000300000800'
}

image_path = "image15.jpg"
res='560782103318500072270003805832075019941028507657001208728006984496857321183200756'

isForce=0
blur_kernel_size=5


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
    save_image_with_fallback(contour_image, os.path.join(output_dir, '5_largest_contour.jpg'))

    # 如果找到了四边形，返回其坐标
    if len(approx) == 4:
        # 保存四边形轮廓
        quad_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(quad_image, [approx], -1, (0, 0, 255), 3)
        save_image_with_fallback(quad_image, os.path.join(output_dir, '6_sudoku_grid_quad.jpg'))
        print("找到四边形数独网格")
        return approx.reshape(4, 2)
    else:
        # 如果没有找到完美的四边形，使用边界矩形
        x, y, w, h = cv2.boundingRect(largest_contour)
        rect_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(rect_image, (x, y), (x + w, y + h), (255, 0, 0), 3)
        save_image_with_fallback(rect_image, os.path.join(output_dir, '6_sudoku_grid_rectangle.jpg'))
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
    
    results = reader.readtext(cell_image_color, allowlist='123456789',paragraph=False)
    
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


def get_blur_kernel_size_by_width(grid_width):
    """根据网格宽度动态选择模糊核大小"""
    if isForce:
        return blur_kernel_size
    if grid_width < 500:
        return 7
    elif grid_width >= 500 and grid_width < 750:
        return 11
    elif grid_width >= 750 and grid_width < 1000:
        return 15
    else:  # >= 1000
        return 19

def extract_digits_from_grid(image, grid_corners, kernel_size, output_dir, use_sharpen=True, blur_kernel_size=3):
    """从数独网格中提取每个单元格的数字（不使用透视变换）"""
    # 获取网格的边界矩形
    x, y, w, h = cv2.boundingRect(grid_corners)
    
    # 根据网格宽度动态设置模糊核大小
    dynamic_blur_kernel_size = get_blur_kernel_size_by_width(w)
    print(f"检测到网格宽度: {w}px，自动选择模糊核大小: {dynamic_blur_kernel_size}x{dynamic_blur_kernel_size}")
    
    # 直接从原始图像中裁剪网格区域
    grid_roi = image[y:y+h, x:x+w]
    
    # 保存裁剪的网格区域
    save_image_with_fallback(grid_roi, os.path.join(output_dir, '7_grid_roi.jpg'))
    print("网格区域裁剪完成")

    # 在裁剪后的图像上绘制网格线，便于观察
    grid_with_lines = grid_roi.copy()
    if len(grid_with_lines.shape) == 2:  # 如果是灰度图，转换为彩色
        grid_with_lines = cv2.cvtColor(grid_with_lines, cv2.COLOR_GRAY2BGR)
    
    # 创建一个用于绘制数字边界框的图像
    grid_with_digit_boxes = grid_with_lines.copy()
    
    # 创建在二值图上绘制单元格边界的图像
    # 只取网格区域，不要方框外的内容
    processed_with_cells = image[y:y+h, x:x+w].copy()
    if len(processed_with_cells.shape) == 2:  # 如果是灰度图，转换为彩色以便绘制彩色线条
        processed_with_cells = cv2.cvtColor(processed_with_cells, cv2.COLOR_GRAY2BGR)
    
    cell_width = w / 9
    cell_height = h / 9
    for i in range(10):  # 绘制10条线（0到9）
        # 绘制垂直线
        cv2.line(grid_with_lines, (int(i * cell_width), 0), (int(i * cell_width), h), (0, 255, 0), 1)
        # 绘制水平线
        cv2.line(grid_with_lines, (0, int(i * cell_height)), (w, int(i * cell_height)), (0, 255, 0), 1)
    
    save_image_with_fallback(grid_with_lines, os.path.join(output_dir, '8_grid_with_lines.jpg'))
    print("网格线图像已保存")

    # 初始化9x9数组
    sudoku_array = []
    digit_positions = []  # 存储数字位置信息
    filtered_count = 0    # 记录被尺寸条件过滤掉的数字数量
    
    # 定义过滤原因映射
    filter_reason_map = {}  # key: 过滤原因类型, value: 字母
    filter_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
    next_letter_index = 0

    # 计算单元格尺寸
    cell_width = w / 9
    cell_height = h / 9

    print("开始识别每个单元格的数字...")
    for row in range(9):
        sudoku_row = []
        for col in range(9):
            # 只使用百分比边距
            margin_x = int(cell_width * 0.08)
            margin_y = int(cell_height * 0.08)
            
            # 计算单元格在裁剪图像中的坐标，转换为整数
            cell_x = int(col * cell_width + margin_x)
            cell_y = int(row * cell_height + margin_y)
            cell_w = int(cell_width - 2 * margin_x)  # 左右各减去边距
            cell_h = int(cell_height - 2 * margin_y)  # 上下各减去边距

            # 确保不会越界
            cell_x = max(0, cell_x)
            cell_y = max(0, cell_y)
            cell_w = min(cell_w, w - cell_x)
            cell_h = min(cell_h, h - cell_y)

            # 在裁剪后的网格图像上绘制蓝色细线的单元格边界
            # 由于processed_with_cells现在是裁剪后的网格区域，直接使用相对坐标
            cv2.rectangle(processed_with_cells, 
                        (cell_x, cell_y), 
                        (cell_x + cell_w, cell_y + cell_h), 
                        (255, 101, 0), 1)  # 蓝色 (BGR格式)，线宽1像素

            # 提取单元格
            cell = grid_roi[cell_y:cell_y + cell_h, cell_x:cell_x + cell_w]

            # 进一步处理单元格
            if cell.size > 0 and cell_w > 0 and cell_h > 0:
                # 对单元格进行额外的预处理 - 使用动态模糊核大小
                cell_cleaned = preprocess_cell(cell, kernel_size, use_sharpen, dynamic_blur_kernel_size)
                
                # 调整大小以提高OCR准确性
                cell_resized = cv2.resize(cell_cleaned, (84, 84))

                # 使用改进的识别函数获取数字和位置信息
                result_info = recognize_digit_with_position_info(cell_resized)
                
                original_digit = result_info['digit']  # 保存原始识别结果
                digit = result_info['digit']
                filter_type = None  # 记录过滤类型
                filter_reason = ""
                
                # 新增：打印每个识别出的数字的位置和边距比例信息（在过滤之前）
                if result_info['found'] and result_info['position'] and original_digit > 0:
                    x_min, y_min, x_max, y_max = result_info['position']
                    
                    # 计算边距
                    distance1 = x_min  # 左边框到方格左边的距离
                    distance2 = 84 - x_max  # 右边框到方格右边框的距离
                    
                    # 计算边距比例，处理除零情况
                    min_distance = min(distance1, distance2)
                    max_distance = max(distance1, distance2)
                    if min_distance == 0:
                        margin_ratio_text = "min=0"
                    else:
                        margin_ratio = max_distance / min_distance
                        margin_ratio_text = f"{margin_ratio:.2f}"
                    
                    # 计算中心距离比例
                    center_x = 42
                    center_distance1 = abs(x_min - center_x)
                    center_distance2 = abs(x_max - center_x)
                    min_center_distance = min(center_distance1, center_distance2)
                    max_center_distance = max(center_distance1, center_distance2)
                    if min_center_distance == 0:
                        center_ratio_text = "min=0"
                    else:
                        center_ratio = max_center_distance / min_center_distance
                        center_ratio_text = f"{center_ratio:.2f}"
                    
                    # 计算数字尺寸
                    digit_width = x_max - x_min
                    digit_height = y_max - y_min
                    
                    print(f"方格({row},{col}) 识别到数字{original_digit}: 位置({x_min},{y_min})-({x_max},{y_max}) "
                          f"尺寸{digit_width}x{digit_height} 左边距{distance1}px 右边距{distance2}px "
                          f"边距比例{margin_ratio_text} 中心比例{center_ratio_text} 置信度{result_info['confidence']:.2f}")
                
                # 新增：宽度和高度条件检查
                size_check_passed = True
                if result_info['found'] and result_info['position'] and digit > 0:
                    x_min, y_min, x_max, y_max = result_info['position']
                    digit_width = x_max - x_min
                    digit_height = y_max - y_min
                    
                    # 在调整后的84x84图像中，方格的有效宽度和高度都是84
                    resized_cell_width = 84
                    resized_cell_height = 84
                    min_required_width = resized_cell_width * 0.4   # 方格宽度的40%
                    min_required_height = resized_cell_height * 0.5  # 方格高度的50%
                    
                    # 修改条件：数字2-9且宽度或高度其中之一小于方格一半就过滤
                    if digit >= 2 and digit <= 9 and (digit_width < min_required_width or digit_height < min_required_height):
                        if digit_width < min_required_width and digit_height < min_required_height:
                            filter_type = "size_both"
                            filter_reason = f"数字{digit}的宽度{digit_width}px和高度{digit_height}px都小于最小要求{min_required_width:.1f}px"
                        elif digit_width < min_required_width:
                            filter_type = "size_width"
                            filter_reason = f"数字{digit}的宽度{digit_width}px小于最小要求{min_required_width:.1f}px"
                        else:
                            filter_type = "size_height"
                            filter_reason = f"数字{digit}的高度{digit_height}px小于最小要求{min_required_height:.1f}px"
                        
                        print(f"  → 过滤原因: {filter_reason}")
                        digit = 0  # 将数字设为0（无数字）
                        size_check_passed = False
                        filtered_count += 1
                
                # 新增：位置检查
                if result_info['found'] and result_info['position'] and digit > 0 and size_check_passed:
                    x_min, y_min, x_max, y_max = result_info['position']
                    
                    # 在84x84的图像中，中间位置是42
                    center_x = 42
                    
                    # 检查左上角是否在中间位置右边
                    if x_min > center_x:
                        filter_type = "position_left"
                        filter_reason = f"左上角位置({x_min})在方格中间({center_x})右边"
                        print(f"  → 过滤原因: {filter_reason}")
                        digit = 0
                        filtered_count += 1
                    # 检查右上角是否在中间位置左边
                    elif x_max < center_x:
                        filter_type = "position_right"
                        filter_reason = f"右上角位置({x_max})在方格中间({center_x})左边"
                        print(f"  → 过滤原因: {filter_reason}")
                        digit = 0
                        filtered_count += 1
                
                # 新增：高宽比检查
                if result_info['found'] and result_info['position'] and digit > 0:
                    x_min, y_min, x_max, y_max = result_info['position']
                    digit_width = x_max - x_min
                    digit_height = y_max - y_min
                    
                    # 计算高宽比
                    if digit_width > 0:  # 避免除零错误
                        aspect_ratio = digit_height / digit_width
                        
                        # 如果数字是2-9且高宽比不在1-2之间，则过滤掉
                        if digit >= 2 and digit <= 9 and (aspect_ratio < 0.89 or aspect_ratio > 2.15):
                            filter_type = "aspect_ratio"
                            filter_reason = f"数字{digit}的高宽比{aspect_ratio:.2f}不在0.9-2.15范围内(高{digit_height}px,宽{digit_width}px)"
                            print(f"  → 过滤原因: {filter_reason}")
                            digit = 0
                            filtered_count += 1
                
                # 新增：中心距离比例检查
                if result_info['found'] and result_info['position'] and digit > 0:
                    x_min, y_min, x_max, y_max = result_info['position']
                    
                    # 在84x84的图像中，中心位置是42
                    center_x = 42
                    
                    # 计算左边界到中心的距离和右边界到中心的距离
                    distance1 = abs(x_min - center_x)  # 左边界到中心的距离
                    distance2 = abs(x_max - center_x)  # 右边界到中心的距离
                    
                    # 找出最小和最大距离
                    min_distance = min(distance1, distance2)
                    max_distance = max(distance1, distance2)
                    
                    # 如果最小距离为0，直接过滤
                    if min_distance == 0:
                        filter_type = "center_distance"
                        filter_reason = f"数字{digit}最小中心距离为0(左距离{distance1}px,右距离{distance2}px)"
                        print(f"  → 过滤原因: {filter_reason}")
                        digit = 0
                        filtered_count += 1
                    else:
                        distance_ratio = max_distance / min_distance
                        
                        # 如果比例超过2.5就过滤掉
                        if distance_ratio > 2.5:
                            filter_type = "center_distance"
                            filter_reason = f"数字{digit}的中心距离比例{distance_ratio:.2f}超过2.5(左距离{distance1}px,右距离{distance2}px)"
                            print(f"  → 过滤原因: {filter_reason}")
                            digit = 0
                            filtered_count += 1
                
                # 新增：边距分布比例检查
                if result_info['found'] and result_info['position'] and digit > 0:
                    x_min, y_min, x_max, y_max = result_info['position']
                    
                    # 在84x84的图像中，方格左边界是0，右边界是84
                    distance1 = x_min  # 左边框到方格左边的距离
                    distance2 = 84 - x_max  # 右边框到方格右边框的距离
                    
                    # 找出最小和最大距离
                    min_distance = min(distance1, distance2)
                    max_distance = max(distance1, distance2)
                    
                    # 如果最小距离为0，直接过滤
                    if min_distance == 0:
                        filter_type = "margin_ratio"
                        filter_reason = f"数字{digit}最小边距为0(左边距{distance1}px,右边距{distance2}px)"
                        print(f"  → 过滤原因: {filter_reason}")
                        digit = 0
                        filtered_count += 1
                    else:
                        margin_ratio = max_distance / min_distance
                        
                        # 如果比例超过2.5就过滤掉
                        if margin_ratio > 10:
                            filter_type = "margin_ratio"
                            filter_reason = f"数字{digit}的边距比例{margin_ratio:.2f}超过2.5(左边距{distance1}px,右边距{distance2}px)"
                            print(f"  → 过滤原因: {filter_reason}")
                            digit = 0
                            filtered_count += 1
                
                # 新增：显示多数字过滤信息
                if result_info.get('filtered_reason'):
                    filter_type = "multi_digit"
                    filter_reason = result_info.get('filtered_reason')
                    print(f"  → 过滤原因: {filter_reason}")
                    filtered_count += 1
                
                # 为过滤类型分配字母
                if filter_type and filter_type not in filter_reason_map:
                    if next_letter_index < len(filter_letters):
                        filter_reason_map[filter_type] = filter_letters[next_letter_index]
                        next_letter_index += 1
                    else:
                        filter_reason_map[filter_type] = '?'  # 如果字母用完了，用?代替
                
                sudoku_row.append(digit)
                
                # 如果找到了数字，记录位置信息
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
                        'digit': digit,  # 这里的digit已经是经过过滤后的结果
                        'bbox': (actual_x_min, actual_y_min, actual_x_max, actual_y_max),
                        'confidence': result_info['confidence'],
                        'found': True,  # 添加found键
                        'position': result_info['position'],  # 添加position键
                        'filtered': filter_type is not None,  # 标记是否被过滤
                        'filter_type': filter_type,
                        'filter_reason': filter_reason
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
        filtered = pos_info.get('filtered', False)
        filter_type = pos_info.get('filter_type', None)
        
        # 根据识别结果选择颜色
        if digit > 0:
            # 识别到数字用红色框
            color = (0, 0, 255)  # 红色
            thickness = 2
            label = f"{digit}"
        elif filtered:
            # 被过滤的用橙色框
            color = (0, 165, 255)  # 橙色
            thickness = 1
            # 使用对应的字母标记
            label = filter_reason_map.get(filter_type, "?")
        else:
            # 检测到内容但未识别出数字用黄色框
            color = (0, 255, 255)  # 黄色
            thickness = 1
            label = "?"  # ?表示检测到内容但未识别
        
        # 绘制边界框
        cv2.rectangle(grid_with_digit_boxes, (x_min, y_min), (x_max, y_max), color, thickness)
        
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        
        # 在边界框左上角显示标签
        text_x = x_min
        text_y = y_min - 5 if y_min - 5 > 0 else y_max + 15
        
        cv2.putText(grid_with_digit_boxes, label, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

    # 保存带有数字边界框的图像
    save_image_with_fallback(grid_with_digit_boxes, os.path.join(output_dir, '9_grid_with_digit_boxes.jpg'))
    print(f"数字边界框图像已保存到 {os.path.join(output_dir, '9_grid_with_digit_boxes.jpg')}")
    
    # 打印过滤原因映射表
    if filter_reason_map:
        print("\n过滤原因映射表:")
        for filter_type, letter in filter_reason_map.items():
            if filter_type == "size_both":
                description = "宽度和高度都小于要求"
            elif filter_type == "size_width":
                description = "宽度小于要求"
            elif filter_type == "size_height":
                description = "高度小于要求"
            elif filter_type == "position_left":
                description = "位置偏右(左上角在中间右边)"
            elif filter_type == "position_right":
                description = "位置偏左(右上角在中间左边)"
            elif filter_type == "multi_digit":
                description = "识别到多个数字"
            elif filter_type == "aspect_ratio":
                description = "高宽比不在1-2范围内"
            elif filter_type == "center_distance":
                description = "中心距离比例超过2"
            elif filter_type == "margin_ratio":
                description = "边距分布比例超过2.5"
            else:
                description = filter_type
            print(f"  {letter}: {description}")
    
    # 创建高斯模糊后的整体图像
    print("正在创建高斯模糊后的整体图像...")
    blurred_grid = grid_roi.copy()  # 复制原始网格区域
    
    # 遍历每个方格，应用高斯模糊并放回原位置
    for row in range(9):
        for col in range(9):
            # 计算方格坐标（与上面识别时使用的相同逻辑）
            margin_x = int(cell_width * 0.08)
            margin_y = int(cell_height * 0.08)
            
            cell_x = int(col * cell_width + margin_x)
            cell_y = int(row * cell_height + margin_y)
            cell_w = int(cell_width - 2 * margin_x)
            cell_h = int(cell_height - 2 * margin_y)

            # 确保不会越界
            cell_x = max(0, cell_x)
            cell_y = max(0, cell_y)
            cell_w = min(cell_w, w - cell_x)
            cell_h = min(cell_h, h - cell_y)

            # 提取单元格
            cell = grid_roi[cell_y:cell_y + cell_h, cell_x:cell_x + cell_w]

            # 对单元格应用高斯模糊处理
            if cell.size > 0 and cell_w > 0 and cell_h > 0:
                # 使用与OCR识别相同的预处理方法，使用动态模糊核大小
                cell_blurred = preprocess_cell(cell, kernel_size, use_sharpen, dynamic_blur_kernel_size)
                
                # 将模糊后的方格放回原位置
                blurred_grid[cell_y:cell_y + cell_h, cell_x:cell_x + cell_w] = cell_blurred

    # 保存高斯模糊后的整体图像
    blurred_image_path = os.path.join(output_dir, '11_blurred_grid_composite.jpg')
    success = save_image_with_fallback(blurred_grid, blurred_image_path)
    if success:
        print(f"✓ 高斯模糊整体图像已保存到: {blurred_image_path}")
    else:
        print(f"✗ 高斯模糊整体图像保存失败: {blurred_image_path}")
    
    # 可选：同时创建一个彩色版本的高斯模糊图像（用于更好的可视化）
    if len(blurred_grid.shape) == 2:  # 如果是灰度图
        blurred_grid_color = cv2.cvtColor(blurred_grid, cv2.COLOR_GRAY2BGR)
        
        # 在彩色版本上绘制网格线以便查看方格边界
        for i in range(10):  # 绘制10条线（0到9）
            # 绘制垂直线
            cv2.line(blurred_grid_color, (int(i * cell_width), 0), (int(i * cell_width), h), (0, 255, 0), 1)
            # 绘制水平线
            cv2.line(blurred_grid_color, (0, int(i * cell_height)), (w, int(i * cell_height)), (0, 255, 0), 1)
        
        blurred_color_path = os.path.join(output_dir, '12_blurred_grid_with_lines.jpg')
        success_color = save_image_with_fallback(blurred_grid_color, blurred_color_path)
        if success_color:
            print(f"✓ 带网格线的高斯模糊图像已保存到: {blurred_color_path}")
        else:
            print(f"✗ 带网格线的高斯模糊图像保存失败: {blurred_color_path}")

    # 保存带有蓝色单元格边界的二值图像
    save_image_with_fallback(processed_with_cells, os.path.join(output_dir, '10_processed_with_cell_boundaries.jpg'))
    print(f"带单元格边界的二值图像已保存到 {os.path.join(output_dir, '10_processed_with_cell_boundaries.jpg')}")
    
    # 打印识别到的数字位置统计
    recognized_digits = [pos for pos in digit_positions if pos['digit'] > 0]
    detected_content = [pos for pos in digit_positions if pos['digit'] == 0 and not pos.get('size_filtered', False)]
    
    print(f"识别到 {len(recognized_digits)} 个数字，检测到 {len(detected_content)} 个未识别内容")
    print(f"因尺寸条件过滤掉 {filtered_count} 个可能的误识别")
    
    return sudoku_array, dynamic_blur_kernel_size


def preprocess_cell(cell, kernel_size, use_sharpen=True, blur_kernel_size=3):
    """对单个单元格进行预处理"""
    # 高斯模糊 - 使用可调节的模糊核大小
    blurred = cv2.GaussianBlur(cell, (blur_kernel_size, blur_kernel_size), 0)
    
    # 根据参数决定是否添加锐化效果
    if use_sharpen:
        sharpen_kernel = np.array([[ 0, -1,  0],
                                  [-1,  5, -1],
                                  [ 0, -1,  0]], dtype=np.float32)
        sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)
        processed_blur = sharpened
    else:
        processed_blur = blurred
    
    # 直接返回处理后的图像，不进行形态学操作
    return processed_blur


def recognize_sudoku_digits(image_path, threshold_param, kernel_size, output_dir, use_sharpen=True, blur_kernel_size=3):
    """主函数：识别数独图片中的数字"""
    # 读取图像
    image = cv2.imread(image_path)

    if image is None:
        print(f"无法读取图像: {image_path}")
        return None

    # 保存原始图像
    save_image_with_fallback(image, os.path.join(output_dir, '0_original_sudoku_image.jpg'))
    print(f"原始图像已保存到 {output_dir}")

    # 预处理图像
    processed = preprocess_image(image, threshold_param, kernel_size, output_dir, use_sharpen, blur_kernel_size)

    # 找到数独网格
    grid_corners = find_sudoku_grid(processed, output_dir)

    # 从网格中提取数字
    sudoku_digits, actual_blur_kernel_size = extract_digits_from_grid(processed, grid_corners, kernel_size, output_dir, use_sharpen, blur_kernel_size)

    return sudoku_digits, actual_blur_kernel_size


def save_sudoku_result_to_txt(sudoku_array, output_dir, threshold_param, kernel_size, use_sharpen, blur_kernel_size, actual_blur_kernel_size=None):
    """将数独识别结果保存到txt文件"""
    actual_blur = actual_blur_kernel_size if actual_blur_kernel_size else blur_kernel_size
    sharpen_suffix = "锐化" if use_sharpen else "无锐化"
    txt_filename = os.path.join(output_dir, f'数独识别结果_卷积核{kernel_size}_阈值{threshold_param}_模糊{actual_blur}_{sharpen_suffix}.txt')
    
    # 获取标准答案
    standard_array = convert_res_to_array(res)
    
    # 计算准确率
    accuracy_stats = calculate_accuracy(sudoku_array, standard_array)
    
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write(f"数独识别结果 (阈值参数: {threshold_param}, 卷积核大小: {kernel_size}x{kernel_size}, 模糊核大小: {actual_blur}x{actual_blur}, 锐化: {'是' if use_sharpen else '否'})\n")
        f.write("=" * 70 + "\n\n")
        
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
        
        f.write("\n" + "=" * 70 + "\n")
        
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
        
        f.write("\n" + "=" * 70 + "\n")
        
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
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("二维数组格式:\n")
        f.write("标准答案: \n")
        for row in standard_array:
            f.write(str(row) + "\n")
        f.write("\n识别结果: \n")
        for row in sudoku_array:
            f.write(str(row) + "\n")
        
        f.write("\n" + "=" * 70 + "\n")
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
    
    
    # 固定阈值化参数
    threshold_param = 19
    
    # 固定卷积核大小为3
    kernel_size = 3
    
    # 只使用无锐化版本
    use_sharpen = False
    
    # 使用默认模糊核大小（实际会根据网格宽度动态选择）
    blur_kernel_size = 11  # 这个值不重要，会被动态覆盖
    
    print("开始数独识别...")
    print(f"阈值参数: {threshold_param}")
    print(f"卷积核大小: {kernel_size}x{kernel_size}")
    print(f"标准答案: {res}")
    print(f"锐化处理: 否")
    print("模糊核大小: 根据网格宽度自动选择")
    
    # 创建输出目录
    output_dir = "sudoku_recognition_result"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 识别数独数字
    result, actual_blur_kernel_size = recognize_sudoku_digits(image_path, threshold_param, kernel_size, output_dir, use_sharpen, blur_kernel_size)
    
    if result is not None:
        # 将识别结果保存到txt文件并获取准确率统计
        accuracy_stats = save_sudoku_result_to_txt(result, output_dir, threshold_param, kernel_size, use_sharpen, blur_kernel_size, actual_blur_kernel_size)
        
        # 在控制台显示详细结果
        total_recognized = sum(sum(1 for digit in row if digit != 0) for row in result)
        print(f"\n识别完成!")
        print(f"实际使用的模糊核大小: {actual_blur_kernel_size}x{actual_blur_kernel_size}")
        print(f"总体准确率: {accuracy_stats['overall_accuracy']:.1f}%")
        print(f"数字识别准确率: {accuracy_stats['filled_accuracy']:.1f}%")
        print(f"空格识别准确率: {accuracy_stats['empty_accuracy']:.1f}%")
        print(f"识别数字总数: {total_recognized}")
        
        print(f"\n结果保存在 {output_dir} 文件夹中")
        txt_name = f"数独识别结果_卷积核{kernel_size}_阈值{threshold_param}_模糊{actual_blur_kernel_size}_无锐化.txt"
        print(f"详细结果文件: {txt_name}")
    else:
        print("识别失败")
    
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
