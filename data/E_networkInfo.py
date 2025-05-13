import random

# Fake data for latency and bandwidth - NN training purposes 
def generate_FAKE_network_info():
    return {
        "latency_ms": round(random.uniform(5, 500), 2),
        "bandwidth_mbps": round(random.uniform(0.1, 1000), 2)
    }