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
# 액션 라벨을 데이터 디렉토리로 설정
actions = np.array(os.listdir(PATH))

sequences = 10 # 시퀸스 개수
frames = 30  # 목표 프레임 수

label_map = {label:num for num, label in enumerate(actions)}

# def interpolate_frames(frames, target_length):
#     #시퀀스의 프레임 수를 목표 길이로 보간
#     if len(frames) == 0:
#         return np.array([])
#
#     # 현재 프레임 수
#     current_length = len(frames)
#
#     if current_length == 1:
#         # 프레임이 하나뿐이면 같은 프레임을 반복
#         return np.array([frames[0]] * target_length)
#
#     # 각 특성에 대해 보간 수행
#     n_features = frames[0].shape[0]
#     interpolated_frames = np.zeros((target_length, n_features))
#
#     # 원본 프레임의 시간 포인트
#     x = np.linspace(0, 1, current_length)
#     # 목표 프레임의 시간 포인트
#     x_new = np.linspace(0, 1, target_length)
#
#     # 각 특성별로 보간
#     for feature in range(n_features):
#         y = [frame[feature] for frame in frames]
#         # 1D 보간 함수 생성
#         f = interpolate.interp1d(x, y, kind='linear')
#         # 새로운 시간 포인트에 대한 값 계산
#         interpolated_frames[:, feature] = f(x_new)
#
#     return interpolated_frames


# 랜드마크와 라벨을 저장할 리스트 초기화
landmarks, labels = [], []

# 액션과 시퀸스를 순회하며 랜드마크와 라벨을 로드
for action, sequence in product(actions, range(sequences)):
    temp = []
    for frame in range(frames):
        npy = np.load(os.path.join(PATH, action, str(sequence), str(frame) + '.npy'))
        temp.append(npy)
    landmarks.append(temp)
    labels.append(label_map[action])

# numpy 배열로 변환
X, Y = np.array(landmarks), to_categorical(labels).astype(int)


# 훈련과 테스트 세트로 분리
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=34, stratify=Y)

# 모델 아키텍처 정의
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, X.shape[2])))
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
model.save('test.h5')

# 테스트 세트에 대한 예측 수행
predictions = np.argmax(model.predict(X_test), axis=1)
test_labels = np.argmax(Y_test, axis=1)
accuracy = metrics.accuracy_score(test_labels, predictions)
print(f"Test Accuracy: {accuracy}")
