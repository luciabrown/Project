import random

# Fake data for security capabilities - NN training purposes 
def generate_FAKE_security_capabilities():
    return {
        "aes_ni": random.choice([True, False]),
        "tpm": random.choice([True, False]),
        "secure_enclave": random.choice([True, False])
    }

if __name__ == "__main__":
    for _ in range(10000):
        print(generate_FAKE_security_capabilities())
