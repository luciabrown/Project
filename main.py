# pip install -r requirements.txt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Import synthetic data generators
from data.A_osType import generate_FAKE_os_type
from data.B_cpuInfo import generate_FAKE_cpu_info
from data.C_ramInfo import generate_FAKE_ram_info
from data.D_capabilities import generate_FAKE_security_capabilities
from data.E_networkInfo import generate_FAKE_network_info
from data.F_algos import generate_FAKE_algorithms_choices

# Map OS types to numeric values for ML input
os_map = {
    "Windows": 0,
    "Linux": 1,
    "macOS": 2,
    "Android": 3,
    "iOS": 4,
    "EmbeddedLinux": 5
}

# Generate synthetic dataset using imported realistic generators
def generate_synthetic_dataset(num_samples=1000):
    dataset = []
    for _ in range(num_samples):
        os_type = os_map[generate_os_type()]
        cpu = generate_cpu_info()
        ram = generate_ram_info()
        net = generate_network_info()
        sec = generate_security_capabilities()
        label = generate_supported_algorithms()  # crypto function: 0=AES-128, 1=AES-256, 2=ChaCha20

        cpu_power = cpu["freq"] * cpu["cores"]  # Combine frequency and core count

        # Input format: [cpu_power, ram_size, latency, bandwidth, aes_ni, tpm, enclave, os_type, label]
        dataset.append([
            cpu_power,
            ram["total_gb"],
            net["latency_ms"],
            net["bandwidth_mbps"],
            sec["aes_ni"],
            sec["tpm"],
            sec["enclave"],
            os_type,
            label
        ])
    
    return np.array(dataset)

# Generate the dataset
data = generate_synthetic_dataset(1000)

# Split into features and labels
X = data[:, :-1]  # Features
y = data[:, -1]   # Labels

# Normalize input features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # 3 cryptographic function classes
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate performance
accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy[1]:.2f}")

# Example prediction
sample_input = np.array([[32.0, 8, 20, 200, 1, 0, 1, 0]])  # [cpu_power, ram, latency, bandwidth, aes_ni, tpm, enclave, os_type]
sample_input_scaled = scaler.transform(sample_input)
predicted_class = np.argmax(model.predict(sample_input_scaled), axis=1)
print(f"Predicted cryptographic function: {predicted_class[0]}")