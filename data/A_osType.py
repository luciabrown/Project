import random

# Fake choices for OS - NN training purposes 
def generate_FAKE_os_type():
    return random.choice(["Windows", "Linux", "macOS", "Android", "iOS", "EmbeddedLinux", "FreeBSD"])