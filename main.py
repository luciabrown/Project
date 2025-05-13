# pip install -r requirements.txt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Data generators
from data.A_osType import generate_FAKE_os_type
from data.B_cpuInfo import generate_FAKE_cpu_info
from data.C_ramInfo import generate_FAKE_ram_size
from data.D_capabilities import generate_FAKE_security_capabilities
from data.E_networkInfo import generate_FAKE_network_info
from data.F_algos import generate_FAKE_algorithms_choices

# Map OS types (string) to numeric values
os_map = {
    "Windows": 0,
    "Linux": 1,
    "macOS": 2,
    "Android": 3,
    "iOS": 4,
    "EmbeddedLinux": 5,
    "FreeBSD": 6
}

cpu_architecture_map = {
    "x86_64":0,
    "armv8":1,
    "armv7":2
}

# Generate dataset using generators
def generate_synthetic_dataset(num_samples=10000):
    dataset = []

    # Helper function to flatten the generated data
    for _ in range(num_samples):
        # Generate data for each feature
        os_type = os_map[generate_FAKE_os_type()]
        cpu = generate_FAKE_cpu_info()
        cpu_arch_num = cpu_architecture_map.get(cpu["arch"], -1)
        ram = generate_FAKE_ram_size()
        sec = generate_FAKE_security_capabilities()
        net = generate_FAKE_network_info()
        algos = generate_FAKE_algorithms_choices()

        # Flatten the data and combine into a single list (representing one sample)
        data = [
            os_type,                                  # OS Type
            cpu["freq"],                              # CPU frequency
            cpu["cores"],                             # CPU cores
            cpu_arch_num,                              # CPU architecture
            cpu["aes_ni"],                            # AES-NI availability
            ram["total_mb"],                          # RAM size
            sec["aes_ni"],                            # Security capabilities - AES-NI
            sec["tpm"],                               # TPM availability
            sec["secure_enclave"],                    # Secure enclave availability
            net["latency_ms"],                        # Latency (ms)
            net["bandwidth_mbps"],                    # Bandwidth (Mbps)
            len(algos)                               # Number of available algorithms
        ]
        
        dataset.append(data)

    return np.array(dataset)

# Generate dataset
data = generate_synthetic_dataset(10000)

# Split into features and labels
X = data[:, :-1]  # Features (exclude the last column)
y = data[:, -1]   # Labels (last column)

# Apply preprocessing: One-Hot Encoding for CPU Architecture (arch) and MinMax Scaling for other features

# Define column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('arch', OneHotEncoder(), [3]),  # Apply one-hot encoding on CPU architecture column (index 3)
        ('num', MinMaxScaler(), [0, 1, 2, 4, 5, 6, 7, 8, 9, 10])  # Scale numeric columns
    ])

# Apply the transformations
X_processed = preprocessor.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
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

# Generate 25 new synthetic samples using your existing generators
def generate_sample():
    os_type = os_map[generate_FAKE_os_type()]
    cpu = generate_FAKE_cpu_info()
    cpu_arch_num = cpu_architecture_map.get(cpu["arch"], -1)
    ram = generate_FAKE_ram_size()
    sec = generate_FAKE_security_capabilities()
    net = generate_FAKE_network_info()

    return [
        os_type,
        cpu["freq"],
        cpu["cores"],
        cpu_arch_num,
        cpu["aes_ni"],
        ram["total_mb"],
        sec["aes_ni"],
        sec["tpm"],
        sec["secure_enclave"],
        net["latency_ms"],
        net["bandwidth_mbps"]
    ]

# Generate 25 samples
sample_inputs = [generate_sample() for _ in range(25)]

# Preprocess the inputs
sample_inputs_processed = preprocessor.transform(sample_inputs)

# Predict
predictions = model.predict(sample_inputs_processed)
predicted_classes = np.argmax(predictions, axis=1)

# Print results
print("\nPredictions for 25 new synthetic environments:\n")
for i, (inp, pred) in enumerate(zip(sample_inputs, predicted_classes), 1):
    print(f"Sample {i}: Input={inp} Predicted Crypto Function: {pred}")
