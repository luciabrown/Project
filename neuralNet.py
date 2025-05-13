import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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

# Map CPU architectures (string) to numeric values
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
        os_type = os_map[generate_FAKE_os_type()]
        cpu = generate_FAKE_cpu_info()
        cpu_arch_num = cpu_architecture_map.get(cpu["arch"], -1)
        ram = generate_FAKE_ram_size()
        sec = generate_FAKE_security_capabilities()
        net = generate_FAKE_network_info()
        algos = generate_FAKE_algorithms_choices()

        # Flatten the data and combine into a single list
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
data = generate_synthetic_dataset(10000)

# Split into features and labels
X = data[:, :-1]  # Features 
y = data[:, -1]   # Labels

# Define column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('arch', OneHotEncoder(), [3]),  # Apply one-hot encoding on CPU architecture column (index 3)
        ('num', MinMaxScaler(), [0, 1, 2, 4, 5, 6, 7, 8, 9, 10])  # Scale numeric columns
    ])
X_processed = preprocessor.fit_transform(X)

# Multiply important features with higher weights
def custom_scaler(X):
    X[:, 4] *= 2            # AES-NI 
    X[:, 5] *= 1.5          # RAM size
    X[:, 6] *= 2            #  AES-NI availability
    X[:, 3] *= 1.5          # CPU architecture
    return X
X_processed = custom_scaler(X_processed)

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

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',  # Monitor  validation loss
    factor=0.5,          # Reduce the learning rate by  0.5
    patience=3,          # Wait for 3 epochs without improvement before reducing the learning rate
    min_lr=1e-6,         # Lower bound for the learning rate
    verbose=1            # Print when the learning rate is reduced
)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

class_weights = {
    0: 2.0,  
    1: 1.5,
    2: 1.0,
    3: 1.0
}

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=10,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, lr_scheduler]
)

# Evaluate performance
accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy[1]:.2f}")

model.save('my_model.keras') 
#from tensorflow.keras.models import load_model
#loaded_model = load_model('my_model.h5')
#predictions = loaded_model.predict(X_new_data)