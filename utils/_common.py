# @AlaoCode#> This module places some common and frequently used methods
import os
import torch

def print_debug(objects):
    '''
    printing information in development mode
    '''
    DEBUG_MODE = True
    '''
    SELECT[DEBUG_MODE]===>
    *True: development mode, printing debugging information
    *False: production mode, not printing debugging information
    '''
    if DEBUG_MODE:
        print('DEBUG:', objects)


def check_dir(path: str):
    '''
    check if the file path exists, if not, create a path
    '''
    directory, filename = os.path.split(path)
    if not os.path.isdir(directory):
        os.makedirs(directory)

def get_pytorch_device(use_device='gpu', device_id=0):
    '''
    obtain CUDA device
    '''
    import torch
    device = torch.device("cpu")
    if use_device == 'gpu' and torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        device = torch.device("cuda")  # select GPU
        print(f"CUDA is available. Running on GPU: {device_id}.")
    else:
        print("Running on CPU.")
    print(" ")
    return device

def check_pytorch_device():
    '''
    check CUDA device
    '''
    device = get_pytorch_device()
    print(f"Current pytorch device: {device}")
    print(torch.__version__)                # Check the installed PyTorch version
    print(torch.cuda.is_available())        # Check if CUDA is available. True means GPU-enabled PyTorch
    print(torch.cuda.device_count())        # Return the number of available CUDA devices; 0 means none
    print(torch.version.cuda)               # Check the CUDA version
    print(torch.cuda.get_device_name(0))    # Return the GPU model name


