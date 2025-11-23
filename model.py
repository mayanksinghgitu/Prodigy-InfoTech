import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

num_samples = 500
num_landmarks = 21 * 2
num_classes = 10

X = np.random.rand(num_samples, num_landmarks) - 0.5
y = np.random.randint(0, num_classes, size=(num_samples,))
y_cat = to_categorical(y, num_classes=num_classes)

model = Sequential([
    Dense(64, activation='relu', input_shape=(num_landmarks,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y_cat, epochs=20, batch_size=32, validation_split=0.2)
model.save("hand_gesture_landmark_model.h5")
print("Model saved as 'hand_gesture_landmark_model.h5'")
