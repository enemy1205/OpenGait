import cv2
import numpy as np

# 加载图像
image = cv2.imread(
    "/home/sp/datasets/CASIA_B/Yolov8Seg/GaitDatasetB-silh_occ_random_r40/003/cl-01/036/003-cl-01-036-054.png",
    cv2.IMREAD_GRAYSCALE,
)

# 确保图像是二值的
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# 查找轮廓
contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
output_img = np.zeros_like(binary_image)
# 绘制轮廓
cv2.drawContours(output_img, contours, -1, 255, 1)  # 绘制所有轮廓，线宽为2

# 如果想要保存结果图像
cv2.imwrite("output_contours.png", output_img)
