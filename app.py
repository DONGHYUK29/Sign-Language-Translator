# app.py
import os
from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import language_tool_python
import string
from my_functions import *

app = Flask(__name__)
#경로변경
PATH = r'C:\Users\samsung\SLTHTML\data'
# 전역 변수 설정
model = load_model('my_model.h5', compile=False)
actions = np.array(os.listdir(PATH))
tool = language_tool_python.LanguageToolPublicAPI('en-UK')
sentence, keypoints, last_prediction = [], [], []
grammar_result = ""


def generate_frames():
    cap = cv2.VideoCapture(0)

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
        while True:
            success, image = cap.read()
            if not success:
                break

            # 기존 이미지 처리 로직
            results = image_process(image, holistic)
            draw_landmarks(image, results)

            global sentence, keypoints, last_prediction, grammar_result
            keypoints.append(keypoint_extraction(results))

            if len(keypoints) == 10:
                keypoints_array = np.array(keypoints)
                prediction = model.predict(keypoints_array[np.newaxis, :, :])
                keypoints = []

                if np.amax(prediction) > 0.1:
                    if last_prediction != actions[np.argmax(prediction)]:
                        sentence.append(actions[np.argmax(prediction)])
                        last_prediction = actions[np.argmax(prediction)]

            if len(sentence) > 7:
                sentence = sentence[-7:]

            if sentence:
                sentence[0] = sentence[0].capitalize()

            if len(sentence) >= 2:
                if sentence[-1] in string.ascii_letters and sentence[-2] in string.ascii_letters:
                    sentence[-1] = sentence[-2] + sentence[-1]
                    sentence.pop(len(sentence) - 2)
                    sentence[-1] = sentence[-1].capitalize()

            # 텍스트 표시
            text = ' '.join(sentence) if not grammar_result else grammar_result
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_X_coord = (image.shape[1] - textsize[0]) // 2
            cv2.putText(image, text, (text_X_coord, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # JPEG로 인코딩
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('sign_language_translator.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/reset')
def reset():
    global sentence, keypoints, last_prediction, grammar_result
    sentence, keypoints, last_prediction = [], [], []
    grammar_result = ""
    return "Reset successful"


@app.route('/grammar_check')
def check_grammar():
    global sentence, grammar_result
    text = ' '.join(sentence)
    grammar_result = tool.correct(text)
    return grammar_result


if __name__ == '__main__':
    app.run(debug=True)