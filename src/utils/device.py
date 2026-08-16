import torch

def get_device():
    return "cuda" if touch.cuda.is_available() else "cpu"
