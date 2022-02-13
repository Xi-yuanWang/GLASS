import torch

device = None


def set_device(idx):
    '''
    set device for torch.
    idx = -1 means cpu.
    else use cuda:idx.
    '''
    global device_index, device
    if idx == 'cpu' or idx == -1:
        device = torch.device(f'cpu')
    else:
        device_index = idx
        device = torch.device(
            f'cuda:{device_index}' if torch.cuda.is_available() else 'cpu')
    print("device=", device)
