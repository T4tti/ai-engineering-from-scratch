import sys
import time
import tracemalloc
import logging
from memory_profiler import profile
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

@profile
def debug_print(name, tensor):
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, "
          f"device={tensor.device}, "
          f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, "
          f"mean={tensor.mean().item():.4f}, "
          f"has_nan={tensor.isnan().any().item()}")
        
def train_step(model, batch, criterion, optimizer):
    inputs, target = batch

    optimizer.zero_grad()

    output = model(inputs)
    loss = criterion(output, target)

    if loss.item() > 100 or torch.isnan(loss):
        breakpoint(model, inputs, output, target, loss)

    loss.backward()
    optimizer.step()

    return loss

class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        logger.info(f"{self.name} took {self.end - self.start} seconds")
    with Timer("Data Loading"):
       batch = next(dataloader_iter)
    
    with Timer("forward pass"):
        outputs = model(batch)

    with Timer("backward pass"):
        loss.backward()

def demo_print_debugging():
    print("\n--- 1. Print Debugging for Tensors ---")
    x = torch.randn(32, 784)
    debug_print("input batch", x)

    w = torch.randn(784, 128)
    out = x @ w
    debug_print("after matmul", out)

    with_nan = out.clone()
    with_nan[0, 0] = float("nan")
    debug_print("with injected NaN", with_nan)

if __name__ == "__main__":
    demo_print_debugging()

        