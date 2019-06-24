import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

path = r"C:\Users\Tong\Desktop\Unet-US\data\membrane\train\label"

for i in os.listdir(path):
    label_path = os.path.join(os.path.abspath(path), i)
    print(label_path)
    img = cv2.imread(label_path)
    B = img[:, :, 0] / 128 * 4
    G = img[:, :, 1] / 128 * 2
    R = img[:, :, 2] / 128
    src_new = np.zeros(R.shape).astype("uint8")
    src_new = B + G + R
    cv2.imwrite(label_path, src_new)
