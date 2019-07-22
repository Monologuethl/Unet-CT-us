import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r'D:\TONG\PycharmProjects\Unet-US\data\membrane\train\label\0.png')
# R = img[:, :, 2]
# cv2.imshow("img", img)
# cv2.waitKey(0)

plt.imshow(img)
plt.show()

plt.hist(img.ravel(), 256, [0, 256])
plt.show()
