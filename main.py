# Generate 25 new synthetic samples using your existing generators
import neuralNet as nn

def generate_sample():
    os_type = nn.os_map[nn.generate_FAKE_os_type()]
    cpu = nn.generate_FAKE_cpu_info()
    cpu_arch_num = nn.cpu_architecture_map.get(cpu["arch"], -1)
    ram = nn.generate_FAKE_ram_size()
    sec = nn.generate_FAKE_security_capabilities()
    net = nn.generate_FAKE_network_info()

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
sample_inputs_processed = nn.preprocessor.transform(sample_inputs)

# Predict
predictions = nn.model.predict(sample_inputs_processed)
predicted_classes = nn.np.argmax(predictions, axis=1)

# Print results
print("\nPredictions for 25 new synthetic environments:\n")
for i, (inp, pred) in enumerate(zip(sample_inputs, predicted_classes), 1):
    print(f"Sample {i}: Input={inp} Predicted Crypto Function: {pred}")