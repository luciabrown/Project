import random

# Fake choices for cryptographic algorithms - NN training purposes 
def generate_FAKE_algorithms_choices():
    algos = [
        ["AES-128", "AES-256", "ChaCha20"],
        ["AES-128", "AES-256"],
        ["AES-128"],
        ["ChaCha20", "AES-256"],
        ["RSA-2048", "ECDSA", "AES-128"]
    ]
    return random.choice(algos)

if __name__ == "__main__":
    for _ in range(10000):
        print(generate_FAKE_algorithms_choices())
