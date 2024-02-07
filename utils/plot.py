# test: draw polygons over an image.
import sys
import math
import cv2 as cv
import numpy as np
import random
import sys
sys.path.append("/home/huasi/AllNeedCopy_datasets/dianli/YOLOv5_DOTA_OBB-master/")
from utils.datasets import rotate_augment

# font
font = cv.FONT_HERSHEY_SIMPLEX
# fontScale
fontScale = 1  
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2

# X_CENTER_NORM = X_CENTER_ABS/IMAGE_WIDTH
# Y_CENTER_NORM = Y_CENTER_ABS/IMAGE_HEIGHT
# WIDTH_NORM = WIDTH_OF_LABEL_ABS/IMAGE_WIDTH
# HEIGHT_NORM = HEIGHT_OF_LABEL_ABS/IMAGE_HEIGHT
# 注意：roLabelImg不使用long side长边表示法；
def RoVOCToPoly(xc_n, yc_n, w_n, h_n, W, H, theta1):
    xc = xc_n * W
    yc = yc_n * H
    w = w_n * W
    h = h_n * H
    # 1. calculate the angle theta2
    L = math.sqrt(w**2 + h**2)
    # print("L:    "+str(L))
    # print("w:    "+str(w))
    # print("h:    "+str(h))
    theta2 = math.degrees(math.atan(h/w))
    theta = theta1 + theta2
    # print(str(theta1))
    # print(str(theta))
    # 2. calculate (x1, y1)
    # print(str(L/2))
    dx1 = (L/2) * math.cos(math.radians(theta))
    dy1 = (L/2) * math.sin(math.radians(theta))
    # print("dx1an:"+str(math.cos(math.radians(theta))))
    # print("x1xc :"+str(dx1))
    x1 = xc + dx1
    y1 = yc + dy1
    # 3. calculate (x3, y3)
    x3 = xc - dx1
    y3 = yc - dy1
    # 4. calculate (x2, y2)
    dx2 = h * math.cos(math.radians(90 - theta1))
    dy2 = h * math.sin(math.radians(90 - theta1))
    x2 = x1 + dx2
    y2 = y1 - dy2
    # 5. calculate (x4, y4)
    dx4 = w * math.cos(math.radians(theta1))
    dy4 = w * math.sin(math.radians(theta1))
    x4 = x1 - dx4
    y4 = y1 - dy4

    return x1, y1, x2, y2, x3, y3, x4, y4

# long side to poly corrected;
# 长边表示法到poly格式转换，可视化需要；
def LongSideToPoly(xc_n, yc_n, w_n, h_n, W, H, theta1):
    xc = xc_n * W
    yc = yc_n * H
    w = w_n * W # length of long side 
    h = h_n * H # length of short side
    # 1. calculate the angle theta2
    L = math.sqrt(w**2 + h**2)
    # print("L:    "+str(L))
    # print("w:    "+str(w))
    # print("h:    "+str(h))
    theta2 = math.degrees(math.atan(h/w))
    theta = theta1 + theta2
    print(f"theta1:{theta1}, theta:{theta}")
    # 2. calculate (x1, y1)
    # print(str(L/2))
    dx1 = (L/2) * math.cos(math.radians(theta))
    dy1 = (L/2) * math.sin(math.radians(theta))
    # print("dx1an:"+str(math.cos(math.radians(theta))))
    # print("x1xc :"+str(dx1))
    x1 = xc - dx1
    y1 = yc + dy1
    # 3. calculate (x3, y3)
    x3 = xc + dx1
    y3 = yc - dy1
    # 4. calculate (x2, y2)
    dx2 = w * math.cos(math.radians(theta1))
    dy2 = w * math.sin(math.radians(theta1))
    x2 = x1 + dx2
    y2 = y1 - dy2
    # 5. calculate (x4, y4)
    dx4 = h * math.cos(math.radians(90 - theta1))
    dy4 = h * math.sin(math.radians(90 - theta1))
    x4 = x1 - dx4
    y4 = y1 - dy4

    return x1, y1, x2, y2, x3, y3, x4, y4


# 5 0.4931640625 0.4999999403953552 0.1621093451976776 0.1328124701976776 90  
# 3 0.4453125 0.54736328125 0.0634765625 0.03515625 90
# 0 0.49267578125 0.45361328125 0.2607421875 0.1337890625 90
# 3 0.52978515625 0.46435546875 0.1474609375 0.0498046875 90
# 1 0.49609375 0.37158203125 0.0986328125 0.048828125 90
# 7 0.5333402752876282 0.38969066739082336 0.4465892016887665 0.023188814520835876 107
def DrawPoly(img, xc_n, yc_n, x1, y1, x2, y2, x3, y3, x4, y4):
    H, W, C = img.shape 
    xc = xc_n * W
    yc = yc_n * H
    # print("center:"+str(xc)+" "+str(yc))
    # print("x1y1:  "+str(x1)+" "+str(y1))
    # print("x2y2:  "+str(x2)+" "+str(y2))
    pts = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv.polylines(img,[pts],True,(0,255,0))
    img = cv.circle(img, (int(xc),int(yc)), radius=4, color=(0, 255, 255))
    # Using cv2.putText() method
    # img = cv.putText(img, '1', (int(x1), int(y1)), font, 
    #            fontScale, color, thickness, cv.LINE_AA)
    # img = cv.putText(img, '2', (int(x2), int(y2)), font, 
    #            fontScale, color, thickness, cv.LINE_AA)
    # img = cv.putText(img, '3', (int(x3), int(y3)), font, 
    #            fontScale, color, thickness, cv.LINE_AA) 
    # img = cv.putText(img, '4', (int(x4), int(y4)), font, 
    #                fontScale, color, thickness, cv.LINE_AA) 
# 1. read lines -> string list
# 2. for each string of line, split by spaces;
# 3. convert to required formats;
def VisualizeLabels(img, label):
    H, W, C = img.shape
    with open(label) as file_in:
        lines = []
        for line in file_in:
            class_id, xc_n, yc_n, w_n, h_n, angle = line.split()
            xc, yc, w, h, an = float(xc_n), float(yc_n), float(w_n), float(h_n), int(angle)
            # convert to 4 points
            x1, y1, x2, y2, x3, y3, x4, y4 = LongSideToPoly(xc, yc, w, h, W, H, an)
            # print(str(x1)+" "+str(y1))
            DrawPoly(img, xc, yc, x1, y1, x2, y2, x3, y3, x4, y4)

    cv.imwrite("./out.jpg", img)

def VisualizeLabelsNP(img, labels, classes, name, output_path):
    H, W, C = img.shape
    for label in labels:
        class_id, xc, yc, w, h, an = label[0], label[1], label[2], label[3], label[4], label[5]
        #xc, yc, w, h, an = float(xc_n), float(yc_n), float(w_n), float(h_n), int(angle)
        # convert to 4 points
        x1, y1, x2, y2, x3, y3, x4, y4 = LongSideToPoly(xc, yc, w, h, W, H, an)
        # print(str(x1)+" "+str(y1))
        DrawPoly(img, xc, yc, x1, y1, x2, y2, x3, y3, x4, y4)
        print(f"class id:{int(class_id)}, class:{classes[int(class_id)]}")
        print(f"coord:{int(x1),int(y1)}")
        # cv.putText(img, classes[int(class_id)], (int(x1),int(y1)),  cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv.LINE_AA)

    cv.putText(img, name, (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA)
    cv.imwrite(output_path+name+".jpg", img)


def CalculateLabelMaxWHRatio(label):
    max = 0
    with open(label) as file_in:
        for line in file_in:
            class_id, xc_n, yc_n, w_n, h_n, angle = line.split()
            wh_ratio = float(w_n) / float(h_n)
            if (wh_ratio > max):
                max = wh_ratio
    return max

def GetLabels(label):
    with open(label) as file_in:
        lines = []
        for line in file_in:
            class_id, xc_n, yc_n, w_n, h_n, angle = line.split()
            label_lst = int(class_id),float(xc_n), float(yc_n), float(w_n), float(h_n), float(angle)
            lines.append(np.array(label_lst))
    return lines
# get label directory
import os

def CalculateMaxWHRatio(label_dir):
    directory = os.fsencode(label_dir)
    max_wh_ratio = 0
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        label_path = label_dir+filename
        wh_ratio = CalculateLabelMaxWHRatio(label_path)
        if (wh_ratio > max_wh_ratio):
            max_wh_ratio = wh_ratio

    print(max_wh_ratio)


# img = cv.imread(sys.argv[1])
# label = sys.argv[2]
# VisualizeLabels(img, label)

# # read images in directory argv[1]
# import os
# from pathlib import Path

# directory = os.fsencode(sys.argv[1])

# out_id = 0
# for file in os.listdir(directory):    
#     filename = os.fsdecode(file)
#     if filename.endswith(".jpg"):
#         img = cv.imread(sys.argv[1]+filename)
#         label_path = sys.argv[2]+filename[:-4]+".txt"
#         label = Path(label_path)
#         #H, W, C = img.shape
#         if (not label.is_file()):
#             print(label)
#         else:
#             VisualizeLabels(img, label, "vis_"+str(out_id))
#             out_id = out_id+1
#             # print(str(out_id))

# test rotate augmentation
# rotate_augment(angle, scale, image, labels):
