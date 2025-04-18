import torch
from . import *

print(\
    f"Device : {f"GPU ({torch.cuda.get_device_name(0)})" \
                    if torch.cuda.is_available() \
                    else "cpu"}")

n_epochs = ...

if __name__ == '__main__':
    for epoch in range(n_epochs):
        