import torch
import os

device_index = 0
device = torch.device(
    f'cuda:{device_index}' if torch.cuda.is_available() else 'cpu')
print("device=", device)


def use_cpu():
    global device_index, device
    device = torch.device(f'cpu')
    print(device)


def set_device(idx):
    global device_index, device
    if idx == 'cpu' or idx == -1:
        use_cpu()
    else:
        device_index = idx
        device = torch.device(
            f'cuda:{device_index}' if torch.cuda.is_available() else 'cpu')
        print("device=", device)
