import os
import random
import getpass
from pathlib import Path
import numpy as np
import torch


def _env_first(*names, default=""):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def get_hw_info(hw_type, device_number=None):
    if hw_type == 'rpi':
        password = _env_first("FL_RPI_PASSWORD", "FL_DEVICE_PASSWORD")
    elif hw_type == 'mc1':
        password = _env_first("FL_MC1_PASSWORD", "FL_DEVICE_PASSWORD")
    elif hw_type.split('_')[0] == 'laptop':
        hw_type_env = hw_type.upper().replace("-", "_")
        password = _env_first(f"FL_{hw_type_env}_PASSWORD", "FL_LAPTOP_PASSWORD")
        username = _env_first(f"FL_{hw_type_env}_USERNAME", "FL_LAPTOP_USERNAME", default=getpass.getuser())
        local_path = _env_first(
            f"FL_{hw_type_env}_PATH",
            "FL_LAPTOP_PATH",
            default=str(Path(__file__).resolve().parent.parent),
        )
    else:
        print("[!] ERROR wrong device type.")
        return None

    if device_number is not None:
        device_number_str = str(device_number).zfill(2)
        return password, "student", f"/home/student/ece_361e_fl/logs", f"sld-{hw_type}-{device_number_str}.ece.utexas.edu"
    elif hw_type == 'rpi' or hw_type == 'mc1':
        return password, "student", f"/home/student/ece_361e_fl/files/"
    elif hw_type.split('_')[0] == "laptop":
        return password, username, f"{local_path}/files{hw_type.split('_')[1]}/"


def get_loss_func(loss_name):
    if loss_name == "cross_entropy":
        return torch.nn.CrossEntropyLoss()


def seed_everything(seed=42, verbose=False):
    """
    Sets seed for all the possible sources of randomness to ensure reproducibility.

    Parameters:
        seed (int): The seed value to be set.
    """

    # Set the seed for Python's random module
    random.seed(seed)

    # Set the seed for Python's hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Set the seed for numpy operations
    np.random.seed(seed)

    # Set the seed for PyTorch operations
    torch.manual_seed(seed)

    # Set the seed for PyTorch GPU operations if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

        # Ensuring deterministic behavior in PyTorch for certain cudnn algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if verbose:
        print(f"[+] Seeds for all randomness sources set to: {seed}")
