import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from itertools import product
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set the path to the data directory
#PATH = os.path.join('data')
PATH = "C:\Users\brant\Desktop\3-2\creative1\Sign-Language-Translator\data"

# Create an array of actions (signs) labels by listing the contents of the data directory
actions = np.array(os.listdir(PATH))

# Define the number of sequences
sequences = 5  # You specified that sequences are 5

# Initialize empty lists to store landmarks and labels
landmarks, labels = [], []

# Iterate over actions and sequences to load landmarks and corresponding labels
for action, sequence in product(actions, range(sequences)):
    temp = []
    for frame in range(1, 11):  # Assuming the frame numbers are between 1 and 10
        frame_path = os.path.join(PATH, action, str(sequence), str(frame) + '.npy')
        if os.path.exists(frame_path):  # Check if the file exists
            npy = np.load(frame_path)
            temp.append(npy)
        else:
            print(f"Warning: {frame_path} does not exist.")  # Print a warning if the file is missing

    if len(temp) > 0:  # Append only if there are frames
        landmarks.append(temp)
        labels.append(action)  # Assuming 'action' is the label for each sequence

# Convert landmarks and labels to numpy arrays
landmarks = np.array(landmarks)
labels = np.array(labels)

# Create a label map to map each action label to a numeric value
label_map = {label: num for num, label in enumerate(actions)}

# Convert labels to categorical format
Y = to_categorical([label_map[action] for action in labels])

# Pad sequences to have equal lengths
max_sequence_length = max([len(seq) for seq in landmarks])  # Find the longest sequence
X = pad_sequences(landmarks, maxlen=max_sequence_length, dtype='float32', padding='post', value=0.0)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=34, stratify=Y)

# Define the model architecture
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(None, X.shape[2])))  # Mask padding during training
model.add(LSTM(32, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(LSTM(32, return_sequences=False, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile the model with Adam optimizer and categorical cross-entropy loss
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model with batch size and epochs
model.fit(X_train, Y_train, epochs=100, batch_size=32)

# Save the trained model
model.save('my_model.h5')

# Make predictions on the test set
predictions = np.argmax(model.predict(X_test), axis=1)

# Get the true labels from the test set
test_labels = np.argmax(Y_test, axis=1)

# Calculate the accuracy of the predictions
accuracy = metrics.accuracy_score(test_labels, predictions)
print(f"Test Accuracy: {accuracy}")