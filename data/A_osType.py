import random

# Fake choices for OS - NN training purposes 
def generate_FAKE_os_type():
    return random.choice(["Windows", "Linux", "macOS", "Android", "iOS", "EmbeddedLinux", "FreeBSD"])

if __name__ == "__main__":
    for _ in range(10000):
        print(generate_FAKE_os_type())
