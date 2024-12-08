import os
import numpy as np
import cv2
import mediapipe as mp
from itertools import product
from my_functions import *
import keyboard
from PIL import ImageFont, ImageDraw, Image

# 폰트 경로 설정
fontpath = "C:\\Users\\brant\\Desktop\\3-2\\creative1\\Sign-Language-Translator\\gulim.ttc"  # 시스템에 설치된 한글 폰트 경로
font = ImageFont.truetype(fontpath, 24)  # 폰트 크기 설정

# 한글 텍스트 추가 함수
def draw_text_korean(img, text, pos):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=(255, 255, 255))
    return np.array(img_pil)

# Define the actions (signs) that will be recorded and stored in the dataset
actions = np.array(['잘가'])

# Define the number of sequences and frames to be recorded for each action
sequences = 30
frames = 10

# Set the path where the dataset will be stored
PATH = os.path.join('data')

# Create directories for each action, sequence, and frame in the dataset
for action, sequence in product(actions, range(sequences)):
    try:
        os.makedirs(os.path.join(PATH, action, str(sequence)))
    except:
        pass

# Access the camera and check if the camera is opened successfully
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

# Create a MediaPipe Holistic object for hand tracking and landmark extraction
with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    # Loop through each action, sequence, and frame to record data
    for action, sequence, frame in product(actions, range(sequences), range(frames)):
        # If it is the first frame of a sequence, wait for the spacebar key press to start recording
        if frame == 0:
            while True:
                if keyboard.is_pressed(' '):
                    break
                _, image = cap.read()

                results = image_process(image, holistic)
                draw_landmarks(image, results)

                image = draw_text_korean(image, f'"{action}" 데이터를 녹화 중입니다. 시퀀스 번호: {sequence}', (20, 20))
                image = draw_text_korean(image, '대기 중.', (20, 400))
                image = draw_text_korean(image, '"Space"를 눌러 시작하세요.', (20, 450))
                cv2.imshow('Camera', image)
                cv2.waitKey(1)

                # Check if the 'Camera' window was closed and break the loop
                if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
                    break
        else:
            # For subsequent frames, directly read the image from the camera
            _, image = cap.read()
            # Process the image and extract hand landmarks using the MediaPipe Holistic pipeline
            results = image_process(image, holistic)
            # Draw the hand landmarks on the image
            draw_landmarks(image, results)

            # Display text on the image indicating the action and sequence number being recorded
            image = draw_text_korean(image, f'"{action}" 데이터를 녹화 중입니다. 시퀀스 번호: {sequence}', (20, 20))
            cv2.imshow('Camera', image)
            cv2.waitKey(1)

        # Check if the 'Camera' window was closed and break the loop
        if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
            break

        # Extract the landmarks from both hands and save them in arrays
        keypoints = keypoint_extraction(results)
        frame_path = os.path.join(PATH, action, str(sequence), str(frame))
        np.save(frame_path, keypoints)

    # Release the camera and close any remaining windows
    cap.release()
    cv2.destroyAllWindows()
