import psutil,platform,subprocess
import neuralNet as nn

def get_os_type():
    os_name = platform.system()
    os_type = nn.os_map.get(os_name, -1)
    print(f"[OS] Detected: {os_name} → Mapped: {os_type}")
    return os_type


def get_cpu_info():
    try:
        freq = psutil.cpu_freq().current
    except Exception:
        freq = 0

    cores = psutil.cpu_count(logical=False) or 1
    arch = platform.processor()
    aes_ni_supported = 0
    try:
        if platform.system() == "Linux":
            aes_ni_supported = int("aes" in subprocess.check_output("cat /proc/cpuinfo", shell=True).decode())
        elif platform.system() == "Windows":
            aes_ni_supported = 0# hardcoded, windows specific check needed here
    except Exception:
        pass

    return {
        "freq": freq,
        "cores": cores,
        "arch": arch,
        "aes_ni": aes_ni_supported
    }

def get_ram_size():
    try:
        total_mb = psutil.virtual_memory().total // (1024 * 1024)
    except Exception:
        total_mb = 0
    print(f"[RAM] Total RAM: {total_mb} MB")
    return {"total_mb": total_mb}

def get_security_capabilities():
    return {
        "aes_ni": 1,# hardcoded
        "tpm": 1,# hardcoded
        "secure_enclave": 0# hardcoded
    }

def get_network_info():
    return {
        "latency_ms": 50,  # hardcoded
        "bandwidth_mbps": 1000  # hardcoded
    }

def generate_real_sample():
    os_type = get_os_type()
    cpu = get_cpu_info()
    cpu_arch_num = nn.cpu_architecture_map.get(cpu["arch"], -1)
    ram = get_ram_size()
    sec = get_security_capabilities()
    net = get_network_info()

    return [[ 
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
    ]]

real_input = generate_real_sample()
real_input_processed = nn.preprocessor.transform(real_input)

prediction = nn.model.predict(real_input_processed)
predicted_class = nn.np.argmax(prediction, axis=1)[0]

print("\nPredicted Cryptographic Function for This Device:")
print(f"Input = {real_input[0]}")
print(f"Predicted Crypto Function Class = {predicted_class}")