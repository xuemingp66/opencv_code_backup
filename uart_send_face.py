import numpy as np
import cv2
import dlib
import math
import serial
import time

pos_list = []
nose_list = []

serial_port = serial.Serial(
    port="/dev/ttyUSB0",
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
)

time.sleep(1)

detector = dlib.get_frontal_face_detector()  # 获取人脸分类器
predictor = dlib.shape_predictor(
    "/home/dlinano/Desktop/opencv_dlib/shape_predictor_68_face_landmarks.dat"
)  # 获取人脸检测器

cap = cv2.VideoCapture(0)
while True:
    # cv2读取图像
    ok, img = cap.read()
    img = cv2.resize(img, (480, 360))
    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 人脸数rects
    rects = detector(img_gray, 0)
    for i in range(len(rects)):  # 遍历每个人脸
        landmarks = np.matrix(
            [[p.x, p.y] for p in predictor(img, rects[i]).parts()]
        )  # 获取人脸坐标点
        for idx, point in enumerate(landmarks):  # 遍历每个坐标点
            pos = (point[0, 0], point[0, 1])  # 读出人脸坐标
            pos_list.append([point[0, 0], point[0, 1]])  # 储存每个人脸到列表 用我们习惯的方式
            # cv2.circle(img, pos, 2, color=(0, 0, 255), thickness=1)  # 在每个点画圆

    # print(len(pos_list))
    if len(pos_list) == 68:
        print(pos_list[31])
        nose_list = pos_list[31]
        serial_port.write(str(nose_list).encode("UTF-8"))
        serial_port.write("\r\n".encode("UTF-8"))
        
        #print(' '.join(nose_list))
        # cv2.circle(img, pos_list[31], 2, color=(0, 0, 255), thickness=1)
    pos_list.clear()

    cv2.namedWindow("img", 1)
    cv2.imshow("img", img)
    k = cv2.waitKey(1)
    if k == 27:  # press 'ESC' to quit
        break

# 释放摄像头 release camera
cap.release()
# do a bit of cleanup
cv2.destroyAllWindows()
