import os
import cv2
import numpy as np
import mediapipe as mp
import json
from my_functions import draw_landmarks, image_process, keypoint_extraction
from PIL import ImageFont, ImageDraw, Image

# 데이터 경로 설정
morpheme_path = 'morpheme'
video_path = 'video_data_1501_1520'
actions = np.array([])

# 결과를 저장할 리스트 초기화
start_list, end_list, name_list, duration_list, video_list = [], [], [], [], []

# morpheme 디렉토리 내의 모든 JSON 파일을 읽어들임
for file_name in os.listdir(morpheme_path):
    if file_name.endswith('.json'):
        file_path = os.path.join(morpheme_path, file_name)

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            duration_list.append(data['metaData']['duration'])
            video_list.append(data['metaData']['name'])

            for item in data['data']:
                start_list.append(item['start'])
                end_list.append(item['end'])
                name_list.append(item['attributes'][0]['name'])

# actions 배열을 name_list로 설정
actions = np.array(name_list)

print(video_list)

# 한글 폰트 설정 (폰트 파일 경로를 시스템에 맞게 설정)
fontpath = "C:\\Windows\\Fonts\\malgun.ttf"  # 예시로 'malgun.ttf' 사용
font = ImageFont.truetype(fontpath, 24)  # 폰트 크기 설정

# 한글 텍스트 그리기 함수
def draw_text_korean(img, text, pos):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=(255, 255, 255))  # 흰색 텍스트
    return np.array(img_pil)

# 학습 데이터 저장 경로
PATH = os.path.join('data')
sequences = 1  # 각 액션당 시퀸스 5개로 설정
sequence_count = -1

# 비디오 파일을 읽고 학습을 위한 프레임 추출
for idx, action in enumerate(actions):
    video_file = os.path.join(video_path, video_list[idx])
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Cannot open video {video_file}")
        continue

    # 시작 및 종료 시간 설정
    start_time = start_list[idx]
    end_time = end_list[idx]

    # 비디오 프레임 레이트 가져오기
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int((end_time - start_time) * fps)

    # 시퀸스당 프레임 수 설정
    frames_per_sequence = total_frames

    if sequence_count == 4:
        sequence_count = -1
    sequence_count += 1


    with mp.solutions.holistic.Holistic(min_detection_confidence=0.1, min_tracking_confidence=0.1) as holistic:
        for sequence in range(sequences):
            frame_count = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))  # 시작 프레임으로 이동

            while frame_count < frames_per_sequence:
                ret, image = cap.read()
                if not ret or (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000) > end_time:
                    break

                # 랜드마크 추출
                results = image_process(image, holistic)
                if results.right_hand_landmarks or results.left_hand_landmarks:
                    draw_landmarks(image, results)
                    keypoints = keypoint_extraction(results)
                    frame_path = os.path.join(PATH, action, str(sequence+sequence_count), str(frame_count))
                    os.makedirs(os.path.dirname(frame_path), exist_ok=True)
                    np.save(frame_path, keypoints)

                frame_count += 1
                # 한글로 처리 중인 액션과 시퀸스 표시
                image = draw_text_korean(image, f'처리 중: "{action}" - 시퀸스 {sequence+sequence_count}, 프레임 {frame_count}', (20, 20))
                cv2.imshow('Frame', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if cv2.getWindowProperty('Frame', cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
cv2.destroyAllWindows()


sequence_count = 4
# 비디오 파일을 읽고 학습을 위한 프레임 추출
for idx, action in enumerate(actions):
    video_file = os.path.join(video_path, video_list[idx])
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Cannot open video {video_file}")
        continue

    # 시작 및 종료 시간 설정
    start_time = start_list[idx]
    end_time = end_list[idx]

    # 비디오 프레임 레이트 가져오기
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int((end_time - start_time) * fps)

    # 시퀸스당 프레임 수 설정
    frames_per_sequence = total_frames

    if sequence_count == 9:
        sequence_count = 4
    sequence_count += 1


    with mp.solutions.holistic.Holistic(min_detection_confidence=0.1, min_tracking_confidence=0.1) as holistic:
        for sequence in range(sequences):
            frame_count = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))  # 시작 프레임으로 이동

            while frame_count < frames_per_sequence:
                ret, image = cap.read()
                if not ret or (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000) > end_time:
                    break

                # 랜드마크 추출
                results = image_process(image, holistic)
                if results.right_hand_landmarks or results.left_hand_landmarks:
                    draw_landmarks(image, results)
                    keypoints = keypoint_extraction(results)
                    frame_path = os.path.join(PATH, action, str(sequence+sequence_count), str(frame_count))
                    os.makedirs(os.path.dirname(frame_path), exist_ok=True)
                    np.save(frame_path, keypoints)

                frame_count += 1
                # 한글로 처리 중인 액션과 시퀸스 표시
                image = draw_text_korean(image, f'처리 중: "{action}" - 시퀸스 {sequence+sequence_count}, 프레임 {frame_count}', (20, 20))
                cv2.imshow('Frame', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if cv2.getWindowProperty('Frame', cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
cv2.destroyAllWindows()
