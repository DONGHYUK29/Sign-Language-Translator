import os
import numpy as np
import cv2
import mediapipe as mp
import base64
from flask import Flask, render_template, Response, request, jsonify
from PIL import ImageFont, ImageDraw, Image
from my_functions import *

app = Flask(__name__)

# 한글 폰트 경로
fontpath = "C:\\Users\\brant\\Desktop\\3-2\\creative1\\Sign-Language-Translator\\gulim.ttc"  # 사용 중인 시스템의 한글 폰트 경로
font = ImageFont.truetype(fontpath, 24)

class DataCollector:
    def __init__(self, base_path=r'C:\Users\brant\Desktop\3-2\creative1\Sign-Language-Translator\data'):
        self.sequences = 10
        self.frames = 30
        self.BASE_PATH = base_path
        self.current_action = None
        self.current_sequence = 0
        self.current_frame = 0
        self.is_collecting = False
        self.camera = cv2.VideoCapture(0)

    def setup_data_directory(self, action_name):
        """사용자가 입력한 액션 이름으로 디렉토리 생성"""
        self.current_action = action_name
        action_path = os.path.join(self.BASE_PATH, action_name)

        # 디렉토리 생성
        for sequence in range(self.sequences):
            try:
                os.makedirs(os.path.join(action_path, str(sequence)), exist_ok=True)
            except Exception as e:
                print(f"디렉토리 생성 중 오류: {e}")

    def draw_text_korean(self, img, text, pos):
        """한글 텍스트를 이미지에 표시"""
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text(pos, text, font=font, fill=(255, 255, 255))  # 흰색 텍스트
        return np.array(img_pil)

    def process_frame(self):
        if not self.camera.isOpened():
            return None

        success, frame = self.camera.read()
        if not success:
            return None

        with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
            # 이미지 처리 및 랜드마크 추출
            results = image_process(frame, holistic)
            draw_landmarks(frame, results)

            # 현재 상태 텍스트 표시 (시퀸스 및 프레임 포함)
            text = f'동작: "{self.current_action}" | 시퀸스: {self.current_sequence} | 프레임: {self.current_frame}'
            frame = self.draw_text_korean(frame, text, (10, 20))

            # 키포인트 추출 및 저장
            if self.is_collecting:
                keypoints = keypoint_extraction(results)
                frame_path = os.path.join(self.BASE_PATH, self.current_action,
                                          str(self.current_sequence), str(self.current_frame))
                np.save(frame_path, keypoints)

                # 프레임 및 시퀸스 카운터 업데이트
                self.current_frame += 1
                if self.current_frame >= self.frames:
                    self.current_frame = 0
                    self.current_sequence += 1
                    if self.current_sequence >= self.sequences:
                        self.is_collecting = False

            return frame

    def start_collection(self, action):
        self.setup_data_directory(action)
        self.current_sequence = 0
        self.current_frame = 0
        self.is_collecting = True

    def stop_collection(self):
        self.is_collecting = False
        self.current_sequence = 0
        self.current_frame = 0


collector = DataCollector()


def generate_frames():
    while True:
        frame = collector.process_frame()
        if frame is None:
            break

        # 프레임 인코딩
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = base64.b64encode(buffer).decode('utf-8')

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + base64.b64decode(frame) + b'\r\n')


@app.route('/')
def index():
    return render_template('data_collect.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_collection', methods=['POST'])
def start_collection():
    action = request.form.get('action')
    if action:
        collector.start_collection(action)
        return jsonify({"status": "success", "message": f"{action} 데이터 수집 시작"})
    return jsonify({"status": "error", "message": "동작 이름이 비어 있습니다"})


@app.route('/stop_collection', methods=['POST'])
def stop_collection():
    collector.stop_collection()
    return jsonify({"status": "success", "message": "데이터 수집 중지"})


if __name__ == '__main__':
    app.run(debug=True)
