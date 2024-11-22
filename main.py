# 필요한 라이브러리 임포트
import numpy as np
import os
import string
import mediapipe as mp
import cv2
from my_functions import *
import keyboard
from tensorflow.keras.models import load_model
import language_tool_python
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

# 경로와 모델, 도구 설정
PATH = 'C:\\Users\\brant\\Desktop\\3-2\\creative1\\Sign-Language-Translator\\data'
actions = np.array(os.listdir(PATH))
model = load_model('my_model.h5')
tool = language_tool_python.LanguageToolPublicAPI('en-UK')

sentence, keypoints, last_prediction, grammar_result = [], [], [], []

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

print(actions)

with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    while cap.isOpened():
        _, image = cap.read()
        results = image_process(image, holistic)
        draw_landmarks(image, results)
        keypoints.append(keypoint_extraction(results))

        if len(keypoints) == 30:
            keypoints = np.array(keypoints)
            prediction = model.predict(keypoints[np.newaxis, :, :])
            keypoints = []
            if np.amax(prediction) > 0.5 and last_prediction != actions[np.argmax(prediction)]: #and last_prediction != actions[np.argmax(prediction)]:
                sentence.append(actions[np.argmax(prediction)])
                last_prediction = actions[np.argmax(prediction)]
                print(sentence)

        if len(sentence) > 7:
            sentence = sentence[-7:]

        if keyboard.is_pressed(' '):
            sentence, keypoints, last_prediction, grammar_result = [], [], [], []

        if sentence:
            sentence[0] = sentence[0].capitalize()

        if len(sentence) >= 2:
            if sentence[-1] in string.ascii_letters and sentence[-2] in string.ascii_letters:
                sentence[-1] = sentence[-2] + sentence[-1]
                sentence.pop(len(sentence) - 2)
                sentence[-1] = sentence[-1].capitalize()

        # if keyboard.is_pressed('enter'):
        #     text = ' '.join(sentence)
        #     grammar_result = tool.correct(text)

        # 텍스트를 표시할 Y 좌표를 조정
        text_Y_coord = 430  # Y 좌표 조정 값 (기존 470에서 더 위로 올려 설정)

        if grammar_result:
            text_bbox = font.getbbox(grammar_result)
            text_width = text_bbox[2] - text_bbox[0]
            text_X_coord = (image.shape[1] - text_width) // 2
            image = draw_text_korean(image, grammar_result, (text_X_coord, text_Y_coord))
        else:
            text_bbox = font.getbbox(' '.join(sentence))
            text_width = text_bbox[2] - text_bbox[0]
            text_X_coord = (image.shape[1] - text_width) // 2
            image = draw_text_korean(image, ' '.join(sentence), (text_X_coord, text_Y_coord))

        cv2.imshow('Camera', image)
        cv2.waitKey(1)

        if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    tool.close()
