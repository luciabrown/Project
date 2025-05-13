import random

# Fake choices for CPU information - NN training purposes 
def generate_FAKE_cpu_info():
    cpus = [
        {"arch": "x86_64", "freq": random.uniform(1.6, 4.0), "cores": random.choice([2, 4, 8, 16]), "aes_ni": True},
        {"arch": "armv8", "freq": random.uniform(1.0, 2.8), "cores": random.choice([2, 4, 6, 8]), "aes_ni": random.choice([True, False])},
        {"arch": "armv7", "freq": random.uniform(0.8, 1.6), "cores": random.choice([1, 2, 4]), "aes_ni": False},
    ]
    return random.choice(cpus)

if __name__ == "__main__":
    for _ in range(10000):
        print(generate_FAKE_cpu_info())
