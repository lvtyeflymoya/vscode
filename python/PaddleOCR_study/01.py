import matplotlib.pyplot as plt
import cv2

# # 创建一个10x10英寸的图形窗口
# plt.figure(figsize=(10, 10))

# # 绘制第一个子图
# plt.subplot(1, 2, 1)  # 1行2列，当前是第1个
# plt.plot([1, 2, 3], [1, 4, 9])

# # 绘制第二个子图
# plt.subplot(1, 2, 2)  # 1行2列，当前是第2个
# plt.plot([1, 2, 3], [1, 0.5, 0.1667])

# # 显示图形
# plt.show()
image = cv2.imread('C:/Users/lelezhang/Desktop/01.png')
cv2.imshow('Loaded Image', image)
cv2.waitKey(0)  # 等待按键