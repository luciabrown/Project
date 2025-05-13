# pip install -r requirements.txt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random

# Fake data
def generate_fake_data(num_samples=1000):
    data = []
    for _ in range(num_samples):
        cpu_power = random.randint(2000, 4000)   # Random CPU power between 2000 and 4000
        ram_size = random.choice([4, 8, 16, 32])  # Random RAM size from a set of options
        latency = random.randint(10, 50)         # Random latency between 10 and 50 ms
        security = random.choice([128, 256])     # Random security options: 128-bit or 256-bit
        bandwidth = random.randint(50, 200)      # Random bandwidth between 50 and 200 Mbps
        cryptographic_function = random.choice([0, 1, 2])  # Randomly choose a cryptographic function (0, 1, 2)
        
        # Append this data point
        data.append([cpu_power, ram_size, latency, security, bandwidth, cryptographic_function])
    
    # Convert the list to a numpy array
    return np.array(data)

# Generate more synthetic data (1000 samples for this example)
data = generate_fake_data(1000)

# Separate features (X) and labels (y)
X = data[:, :-1]  # All columns except the last (features)
y = data[:, -1]   # The last column (labels)

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define neural network model with enhanced architecture
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.2),  # Dropout layer to prevent overfitting
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # Output layer (3 classes: AES-128, AES-256, ChaCha20)
])

# Compile the model with an optimized learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Implement EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate the model
accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy[1]:.2f}")

# Use trained model to predict
sample_input = np.array([[3200, 8, 20, 128, 120]])  # A new scenario
sample_input_scaled = scaler.transform(sample_input)  # Scale the sample input
predicted_class = np.argmax(model.predict(sample_input_scaled), axis=1)
print(f"Predicted cryptographic function: {predicted_class[0]}")