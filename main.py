import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

# Fake data
data = np.array([
    [3000, 8, 30, 128, 100, 0],  # [CPU Power, RAM Size, Latency, Security, Bandwidth, Cryptographic Function]
    [3500, 16, 25, 256, 150, 1], 
    [2500, 4, 40, 128, 50, 2],   
])

# Separate features (X) and labels (y)
X = data[:, :-1]  # All columns except the last (features)
y = data[:, -1]   # The last column (labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define neural network model
model = Sequential()
model.add(Dense(64, input_dim=X.shape[1], activation='relu'))  # Input layer with ReLU activation
model.add(Dense(32, activation='relu'))  # Hidden layer with ReLU activation
model.add(Dense(3, activation='softmax'))  # Output layer (3 classes: AES-128, AES-256, ChaCha20)

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_test, y_test))

# Evaluate model
accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy[1]:.2f}")

# Use trained model to predict
sample_input = np.array([[3200, 8, 20, 128, 120]])  # A new scenario
predicted_class = np.argmax(model.predict(sample_input), axis=1)
print(f"Predicted cryptographic function: {predicted_class[0]}")
