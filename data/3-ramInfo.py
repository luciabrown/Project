import random

# Fake choices for RAM sizes - NN training purposes 
def generate_FAKE_ram_size():
    return random.choice([256, 512, 1024, 2048, 4096, 8192, 16384, 32768])  # MB

if __name__ == "__main__":
    for _ in range(10000):
        print(generate_FAKE_ram_size())
