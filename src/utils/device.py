"""
CoCoNuT: device.py CPU/GPU device selection
"""
import torch

def get_device():
    """Returns 'cuda' if GPU available, else 'cpu'"""
    return "cuda" if torch.cuda.is_available() else "cpu"
