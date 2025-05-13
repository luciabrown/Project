import random

# Fake choices for RAM sizes - NN training purposes 
def generate_FAKE_ram_size():
    return{ 
    "total_mb": random.choice([256, 512, 1024, 2048, 4096, 8192, 16384, 32768])  # MB
    }
