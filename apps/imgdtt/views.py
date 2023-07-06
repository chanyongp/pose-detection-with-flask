import math
import ssl
from time import time

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
from flask import Blueprint, Flask, Response, render_template

ssl._create_default_https_context = ssl._create_unverified_context

# Initializing mediapipe pose class.
# mediapipe pose class를 초기화 한다.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
# pose detect function에 image detect=True, 최소감지신뢰도 = 0.3, 모델 복잡도 =2를 준다.
pose = mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.3, model_complexity=2
)

# Initializing mediapipe drawing class, useful for annotation.
# mediapipe의 drawing class를 초기화한다.
mp_drawing = mp.solutions.drawing_utils

# 이미지 읽어오기
# 샘플 이미지를 cv2.imread()로 읽어온다
# Read an image from the specified path.
sample_img = cv2.imread(
    "/Users/parkchan-yong/flaskbook1/apps/images/20230704_115049_005_saved.jpg"
)


def detectPose(image, pose, display=True):
    # 예시이미지 copy하기
    output_image = image.copy()

    # 컬러 이미지 BGR TO RGB 변환
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # pose detection 수행
    results = pose.process(imageRGB)

    # input image의 너비&높이 탐색
    height, width, _ = image.shape

    # detection landmarks를 저장할 빈 list 초기화
    landmarks = []

    # landmark가 감지 되었는지 확인
    if results.pose_landmarks:
        # landmark 그리기
        mp_drawing.draw_landmarks(
            image=output_image,
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
        )

        # 감지된 landmark 반복
        for landmark in results.pose_landmarks.landmark:
            # landmark를 list에 추가하기
            landmarks.append(
                (
                    int(landmark.x * width),
                    int(landmark.y * height),
                    (landmark.z * width),
                )
            )

    # 오리지널 image와 pose detect된 image 비교
    if display:
        # 오리지널 & 아웃풋 이미지 그리기
        plt.figure(figsize=[22, 22])
        plt.subplot(121)
        plt.imshow(image[:, :, ::-1])
        plt.title("Original Image")
        plt.axis("off")
        plt.subplot(122)
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis("off")

        # 3D 랜드마크 나타내기
        mp_drawing.plot_landmarks(
            results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS
        )

    # 그렇지 않다면, output_image 와 landmark return한다
    else:
        return output_image, landmarks


# 앵글 계산 함수
def calculateAngle(landmark1, landmark2, landmark3):
    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    # Check if the angle is less than zero.
    if angle < 0:
        # Add 360 to the found angle.
        angle += 360

    # Return the calculated angle.
    return angle


# 분류 함수


def classifyPose(landmarks, output_image, display=False):
    # Initialize the label of the pose. It is not known at this stage.
    label = "Unknown Pose"

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)

    anglel = calculateAngle(
        landmarks[mp_pose.PoseLandmark.NOSE.value],
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
    )
    angler = calculateAngle(
        landmarks[mp_pose.PoseLandmark.NOSE.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
    )
    if anglel < 177 or angler < 177:
        label = "forward head posture"
    if anglel > 177 or angler > 177:
        label = "normal"
    if label != "Unknown Pose":
        color = (0, 255, 0)

    # 분류되지 않은 자세라면 Unkwown Pose로 왼쪽 상단에 연두색으로 text 입력
    cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    # 결과 이미지 보여주기 Check if the resultant image is specified to be displayed.
    if display:
        # 결과 이미지를 BGR TO RGB로 matplotlib을 이용해 꺼내준다.
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis("off")

    else:
        # 결과 이미지랑 표시될 label을 return 한다
        return output_image, label


# -------------------------
# Setup Pose function for video.
pose_video = mp_pose.Pose(
    static_image_mode=False, min_detection_confidence=0.5, model_complexity=1
)

imgdtt = Blueprint(
    "imgdtt",
    __name__,
    template_folder="templates",
    static_folder="static",
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

df = pd.DataFrame()
cap = cv2.VideoCapture(0)


def generate_frames():
    global df
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.flip(frame, 1)

        # Get the width and height of the frame
        frame_height, frame_width, _ = frame.shape

        # Resize the frame while keeping the aspect ratio.
        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

        # Perform Pose landmark detection.
        frame, landmarks = detectPose(frame, pose_video, display=False)

        # Check if the landmarks are detected.
        if landmarks:
            # Perform the Pose Classification.
            frame, _ = classifyPose(landmarks, frame, display=False)
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


"""
        # Wait until a key is pressed.
        # Retreive the ASCII code of the key pressed
        k = cv2.waitKey(1) & 0xFF

        # Check if 'ESC' is pressed.
        if k == 27:
            # Break the loop.
            break
"""


@imgdtt.route("/")
def index():
    return render_template("imgdtt/index.html")


@imgdtt.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


"""
def create_app():
    app = Flask(__name__)
    app.register_blueprint(imgdtt)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)

"""
