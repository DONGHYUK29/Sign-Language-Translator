import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from itertools import product
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from scipy import interpolate
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# 데이터 경로 설정
PATH = 'C:\\Users\\brant\\Desktop\\3-2\\creative1\\Sign-Language-Translator\\data'
# 액션(손짓) 라벨을 데이터 디렉토리 목록으로 설정
actions = np.array(os.listdir(PATH))
# 시퀸스 개수와 목표 프레임 수 설정
sequences = 10
TARGET_FRAMES = 30  # 목표 프레임 수


def interpolate_frames(frames, target_length):
    #시퀀스의 프레임 수를 목표 길이로 보간
    if len(frames) == 0:
        return np.array([])

    # 현재 프레임 수
    current_length = len(frames)

    if current_length == 1:
        # 프레임이 하나뿐이면 같은 프레임을 반복
        return np.array([frames[0]] * target_length)

    # 각 특성에 대해 보간 수행
    n_features = frames[0].shape[0]
    interpolated_frames = np.zeros((target_length, n_features))

    # 원본 프레임의 시간 포인트
    x = np.linspace(0, 1, current_length)
    # 목표 프레임의 시간 포인트
    x_new = np.linspace(0, 1, target_length)

    # 각 특성별로 보간
    for feature in range(n_features):
        y = [frame[feature] for frame in frames]
        # 1D 보간 함수 생성
        f = interpolate.interp1d(x, y, kind='linear')
        # 새로운 시간 포인트에 대한 값 계산
        interpolated_frames[:, feature] = f(x_new)

    return interpolated_frames


# 랜드마크와 라벨을 저장할 리스트 초기화
landmarks, labels = [], []

# 액션과 시퀸스를 순회하며 랜드마크와 라벨을 로드
for action, sequence in product(actions, range(sequences)):
    sequence_path = os.path.join(PATH, action, str(sequence))
    if os.path.exists(sequence_path):
        # 프레임 파일 로드 및 정렬
        frames = sorted(
            [f for f in os.listdir(sequence_path) if f.endswith('.npy')],
            key=lambda x: int(x.split('.')[0])
        )

        # 프레임 데이터 로드
        frame_data = []
        for frame_file in frames:
            frame_path = os.path.join(sequence_path, frame_file)
            npy = np.load(frame_path)
            frame_data.append(npy)

        if len(frame_data) > 0:
            # 프레임 수를 TARGET_FRAMES로 보간
            interpolated_sequence = interpolate_frames(frame_data, TARGET_FRAMES)
            if len(interpolated_sequence) > 0:
                landmarks.append(interpolated_sequence)
                labels.append(action)

# numpy 배열로 변환
X = np.array(landmarks)
print(f"Data shape: {X.shape}")

# 라벨을 카테고리 형식으로 변환
label_map = {label: num for num, label in enumerate(actions)}
Y = to_categorical([label_map[action] for action in labels])
print(label_map)

# 훈련과 테스트 세트로 분리
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=34, stratify=Y)

# 모델 아키텍처 정의
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(TARGET_FRAMES, X.shape[2])))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Adam 사용, categorical crossentropy 손실 함수로 모델 컴파일
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# 모델 훈련
model.fit(X_train, Y_train, epochs=80)

# 모델 저장
model.save('my_model.h5')

# 테스트 세트에 대한 예측 수행
predictions = np.argmax(model.predict(X_test), axis=1)
test_labels = np.argmax(Y_test, axis=1)
accuracy = metrics.accuracy_score(test_labels, predictions)
print(f"Test Accuracy: {accuracy}")
