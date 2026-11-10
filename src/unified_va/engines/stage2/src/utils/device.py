"""Device utilities: auto-select free GPU."""
import os
import torch


def get_free_gpu() -> str:
    """Auto-select the GPU with most free memory.

    Returns:
        Device string: 'cuda:X' or 'cpu' if no GPU available.
    """
    if not torch.cuda.is_available():
        return "cpu"

    try:
        # Get memory info for all GPUs
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            return "cpu"

        # Use nvidia-ml-py if available for accurate memory query
        try:
            import pynvml
            pynvml.nvmlInit()
            free_memory = []
            for i in range(n_gpus):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_memory.append(mem_info.free)
            best_gpu = int(max(range(n_gpus), key=lambda i: free_memory[i]))
            return f"cuda:{best_gpu}"
        except ImportError:
            # Fallback: use torch memory stats
            free_memory = []
            for i in range(n_gpus):
                props = torch.cuda.get_device_properties(i)
                # Get allocated memory (approximation)
                torch.cuda.set_device(i)
                allocated = torch.cuda.memory_allocated(i)
                free = props.total_memory - allocated
                free_memory.append(free)
            best_gpu = int(max(range(n_gpus), key=lambda i: free_memory[i]))
            return f"cuda:{best_gpu}"
    except Exception:
        # If anything fails, fall back to cuda:0
        return "cuda:0" if torch.cuda.is_available() else "cpu"


def setup_device(cfg_device: str | None = None) -> torch.device:
    """Setup device for training.

    Args:
        cfg_device: Device from config. If None or 'auto', auto-select free GPU.
                   If 'cuda' without index, auto-select.
                   Otherwise use as-is.

    Returns:
        torch.device object
    """
    if cfg_device is None or cfg_device == "auto":
        device_str = get_free_gpu()
    elif cfg_device == "cuda":
        device_str = get_free_gpu()
    elif cfg_device.startswith("cuda:"):
        device_str = cfg_device
    else:
        device_str = cfg_device

    device = torch.device(device_str)

    # Log device info
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        mem_allocated = torch.cuda.memory_allocated(device) / 1024**2
        mem_reserved = torch.cuda.memory_reserved(device) / 1024**2
        print(f"[GPU] Using {device} | {props.name} | "
              f"Allocated: {mem_allocated:.1f}MB | Reserved: {mem_reserved:.1f}MB")
    else:
        print(f"[CPU] Using CPU device")

    return device
