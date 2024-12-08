# app.py
import os
from flask import Flask, render_template, Response, jsonify
import subprocess
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import language_tool_python
import string
from my_functions import *

from PIL import ImageFont, ImageDraw, Image
import openai

# GPT API 키 설정
openai.api_key = "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"

# 폰트 경로 설정
fontpath = "C:\\Users\\brant\\Desktop\\3-2\\creative1\\Sign-Language-Translator\\gulim.ttc"  # 시스템에 설치된 한글 폰트 경로
font = ImageFont.truetype(fontpath, 24)  # 폰트 크기 설정

# 한글 텍스트 추가 함수
def draw_text_korean(img, text, pos):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=(255, 255, 255))
    return np.array(img_pil)


app = Flask(__name__)
#경로변경
PATH = r'C:\Users\brant\Desktop\3-2\creative1\Sign-Language-Translator\data'
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

            if len(keypoints) == 30:
                keypoints_array = np.array(keypoints)
                prediction = model.predict(keypoints_array[np.newaxis, :, :])
                keypoints = []

                if np.amax(prediction) > 0.7:
                    if last_prediction != actions[np.argmax(prediction)]:
                        sentence.append(actions[np.argmax(prediction)])
                        last_prediction = actions[np.argmax(prediction)]

            if len(sentence) > 7:
                sentence = sentence[-7:]

            if sentence:
                sentence[0] = sentence[0].capitalize()

            # if len(sentence) >= 2:
            #     if sentence[-1] in string.ascii_letters and sentence[-2] in string.ascii_letters:
            #         sentence[-1] = sentence[-2] + sentence[-1]
            #         sentence.pop(len(sentence) - 2)
            #         sentence[-1] = sentence[-1].capitalize()

            text_Y_coord = 430
            # 텍스트 표시
            text = ' '.join(sentence)
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

            # textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            # text_X_coord = (image.shape[1] - textsize[0]) // 2
            # cv2.putText(image, text, (text_X_coord, 470),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

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

    if not text.strip():
        return "검사할 텍스트가 없습니다."

    try:
        # OpenAI API 요청
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a grammar correction assistant."},
                {"role": "user", "content": f"Please correct the grammar of the following text: {text}"}
            ]
        )
        # GPT 응답 가져오기
        grammar_result = response['choices'][0]['message']['content']
        return grammar_result.strip()
    except Exception as e:
        print(f"Error with GPT API: {e}")
        return "문법 검사 중 오류가 발생했습니다."


if __name__ == '__main__':
    app.run(debug=True)