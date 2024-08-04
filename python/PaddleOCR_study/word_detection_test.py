from paddleocr import PaddleOCR
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ocr = PaddleOCR(show_log=False)
img_path = './PaddleOCR_study/test_image/all.png'
result = ocr.ocr(img_path, rec = False)
#print(f"the predicted text box of {img_path} are follows.")
print(result)

import numpy as np
import cv2
import matplotlib.pyplot as plt
image = cv2.imread(img_path)
boxes = [line[0] for line in result[0]]
for box in result[0]:
    box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
    image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
# 画出读取的图片
# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# plt.pause(30)
cv2.imshow('Loaded Image', image)
cv2.waitKey(0)  # 等待按键

