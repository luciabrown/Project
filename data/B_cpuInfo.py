import random

# Fake choices for CPU information - NN training purposes 
def generate_FAKE_cpu_info():
    cpus = [
        {"arch": "x86_64", "freq": random.uniform(1.6, 4.0), "cores": random.choice([2, 4, 8, 16]), "aes_ni": 1},
        {"arch": "armv8", "freq": random.uniform(1.0, 2.8), "cores": random.choice([2, 4, 6, 8]), "aes_ni": random.choice([1, 0])},
        {"arch": "armv7", "freq": random.uniform(0.8, 1.6), "cores": random.choice([1, 2, 4]), "aes_ni": 0},
    ]
    return random.choice(cpus)