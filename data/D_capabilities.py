import random

# Fake data for security capabilities - NN training purposes 
def generate_FAKE_security_capabilities():
    return {
        "aes_ni": random.choice([0, 1]),
        "tpm": random.choice([0, 1]),
        "secure_enclave": random.choice([0, 1])
    }
