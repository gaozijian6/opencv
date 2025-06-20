import cv2
import numpy as np

# 创建一个300x300的彩色图像
height, width = 300, 300
image = np.zeros((height, width, 3), dtype=np.uint8)

# 设置背景为蓝色
image[:, :] = (255, 100, 50)  # BGR格式：蓝色为主

# 在图像上绘制一些图形
# 绘制一个红色圆形
cv2.circle(image, (150, 150), 80, (0, 0, 255), -1)

# 绘制一个绿色矩形
cv2.rectangle(image, (50, 50), (120, 120), (0, 255, 0), 3)

# 绘制一些文字
cv2.putText(image, 'Hello OpenCV', (80, 250), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# 保存图像到当前文件夹
filename = 'created_image.jpg'
success = cv2.imwrite(filename, image)

if success:
    print(f"图像已成功保存为 {filename}")
else:
    print("保存图像失败")

# 显示图像信息
print(f"图像尺寸: {image.shape}")
print(f"图像类型: {image.dtype}")
