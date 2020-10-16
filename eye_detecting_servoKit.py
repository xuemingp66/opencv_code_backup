# -*- coding: utf-8 -*-
# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils

# from adafruit_servokit import ServoKit
import numpy as np  # 数据处理的库 numpy
import argparse
import imutils
import time
import dlib
import cv2
import math
import tkinter as tk  # 导入GUI相关的库
from PIL import Image, ImageTk
import threading

"""
# 定义舵机
myKit = ServoKit(channels=16)
pan_out = 90
titl_out = 70
myKit.servo[1].angle = pan_out
myKit.servo[0].angle = titl_out
"""


def eye_aspect_ratio(eye):
    # 垂直眼标志（X，Y）坐标
    A = dist.euclidean(eye[1], eye[5])  # 计算两个集合之间的欧式距离
    B = dist.euclidean(eye[2], eye[4])
    # 计算水平之间的欧几里得距离
    # 水平眼标志（X，Y）坐标
    C = dist.euclidean(eye[0], eye[3])
    # 眼睛长宽比的计算
    ear = (A + B) / (2.0 * C)
    # 返回眼睛的长宽比s
    return ear


def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
    B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
    C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar


# 定义两个常数
# 眼睛长宽比
# 闪烁阈值
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 1
# 初始化帧计数器和眨眼总数5

# 打哈欠长宽比
# 闪烁阈值
MAR_THRESH = 0.5
MOUTH_AR_CONSEC_FRAMES = 3

counter_blinks = 0
total_blinks = 0  #

counter_mouth = 0
total_mouth = 0

# 定义总的窗口标题，大小
window_head = tk.Tk()
window_head.title("疲劳检测")
window_head.geometry("362x500")


# 定义摄像头画面大小，位置
Frame_head = tk.Frame(window_head, width=360, height=270)
Frame_head.grid(row=0, column=0, padx=0, pady=0)  # 按照元素排列和左上角坐标

# 对画面的label进一步定义
label_head = tk.Label(Frame_head)
label_head.grid(row=0, column=0)

# 对文字label定义
label_fps = tk.Label(text="FPS:", font=("Arial", 18))
label_fps.place(x=0, y=450)

label_ear = tk.Label(text="ear:", font=("Arial", 18))
label_ear.place(x=0, y=300)

#
label_blinks = tk.Label(text="blinks:", font=("Arial", 18))
label_blinks.place(x=0, y=400)


label_mar = tk.Label(text="mar:", font=("Arial", 18))
label_mar.place(x=180, y=300)

#
label_mouth = tk.Label(text="Yowning:", font=("Arial", 18))
label_mouth.place(x=180, y=400)


start_time = time.time()
time_counter = 0

# 初始化DLIB的人脸检测器（HOG），然后创建面部标志物预测
print("[INFO] loading facial landmark predictor...")
# 第一步：使用dlib.get_frontal_face_detector() 获得脸部位置检测器
detector = dlib.get_frontal_face_detector()
# 第二步：使用dlib.shape_predictor获得脸部特征位置检测器
predictor = dlib.shape_predictor("/home/pxm/shape_predictor_68_face_landmarks.dat")

# 第三步：分别获取左右眼面部标志的索引
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# 第四步：打开cv2 本地摄像头
cap = cv2.VideoCapture(0)

# 从视频流循环帧
def cam_loop():
    global counter_blinks
    global total_blinks

    global counter_mouth
    global total_mouth

    global time_counter
    global start_time

    while True:
        # 第五步：进行循环，读取图片，并对图片做维度扩大，并进灰度化
        ret, frame = cap.read()
        frame = cv2.resize(frame, (360, 270))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 第六步：使用detector(gray, 0) 进行脸部位置检测
        rects = detector(gray, 0)

        # 第七步：循环脸部位置信息，使用predictor(gray, rect)获得脸部特征位置的信息
        for rect in rects:
            shape = predictor(gray, rect)

            # 第八步：将脸部特征信息转换为数组array的格式
            shape = face_utils.shape_to_np(shape)

            # 第九步：提取左眼和右眼坐标
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            mouth = shape[mStart:mEnd]

            # 第十步：构造函数计算左右眼的EAR值，使用平均值作为最终的EAR
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # 打哈欠
            mar = mouth_aspect_ratio(mouth)

            # 第十一步：使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

            # 第十二步：进行画图操作，用矩形框标注人脸
            left = rect.left()
            top = rect.top()
            right = rect.right()
            bottom = rect.bottom()
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
            """
            (pan, tilt) = shape[28]
            pan_error = 180 - pan
            tilt_error = 135 - tilt
            # 增量式PID，只用了P
            if abs(pan_error) > 10:
                pan_out = pan_out + pan_error // 30
            if abs(tilt_error) > 10:
                titl_out = titl_out - tilt_error // 23
            if pan_out >= 170:
                pan_out = 170
            if pan_out <= 10:
                pan_out = 10
            if titl_out >= 170:
                titl_out = 170
            if titl_out <= 10:
                titl_out = 10
            
            # 控制舵机输出
            myKit.servo[1].angle = pan_out
            myKit.servo[0].angle = titl_out
            """
            """
                分别计算左眼和右眼的评分求平均作为最终的评分，如果小于阈值，则加1，如果连续3次都小于阈值，则表示进行了一次眨眼活动
            """
            # 第十三步：循环，满足条件的，眨眼次数+1
            if ear < EYE_AR_THRESH:  # 眼睛长宽比：0.2
                counter_blinks += 1
            else:
                # 如果连续3次都小于阈值，则表示进行了一次眨眼活动
                if counter_blinks >= EYE_AR_CONSEC_FRAMES:  # 阈值：3
                    total_blinks += 1
                # 重置眼帧计数器
                counter_blinks = 0
            ear = float("%.3f" % ear)

            if mar > MAR_THRESH:  # 张嘴阈值0.5
                counter_mouth += 1
            else:
                # 如果连续3次都小于阈值，则表示打了一次哈欠
                if counter_mouth >= MOUTH_AR_CONSEC_FRAMES:  # 阈值：3
                    total_mouth += 1
                # 重置嘴帧计数器
                counter_mouth = 0
            mar = float("%.3f" % mar)


            # 第十四步：进行画图操作，68个特征点标识
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            label_blinks.config(text="blinks: " + str(total_blinks))
            label_ear.config(text="ear: " + str(ear))

            label_mouth.config(text="Yowning: " + str(total_mouth))
            label_mar.config(text="mar: " + str(mar))

            break

        time_counter += 1
        # if (time.time() - start_time) != 0:
        fps = float("%.1f" % (time_counter / (time.time() - start_time)))

        # print("FPS: ", counter / (time.time() - start_time))
        time_counter = 0
        start_time = time.time()

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        label_head.imgtk = imgtk
        label_head.configure(image=imgtk)
        label_head.image = imgtk
        label_fps.config(text="FPS: " + str(fps))


# cam_loop()
videoThread = threading.Thread(target=cam_loop, args=())
videoThread.start()
window_head.mainloop()