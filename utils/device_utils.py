"""
设备检测和管理工具
支持GPU/CPU自动切换
"""
import torch
from typing import Optional


def get_device(preferred_device: Optional[str] = None) -> torch.device:
    """
    获取最佳可用设备

    Args:
        preferred_device: 首选设备 ("cuda", "cpu", 或 None 自动检测)

    Returns:
        torch.device 对象
    """
    if preferred_device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("警告: CUDA 不可用，回退到 CPU 模式")
            return torch.device("cpu")
    elif preferred_device == "cpu":
        return torch.device("cpu")
    else:
        # 自动检测
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")


def get_dtype(device: torch.device, use_bfloat16: bool = True) -> torch.dtype:
    """
    根据设备获取合适的数据类型

    Args:
        device: 计算设备
        use_bfloat16: 是否尝试使用 bfloat16

    Returns:
        torch.dtype 数据类型
    """
    if device.type == "cpu":
        # CPU 不支持 bfloat16，使用 float32
        return torch.float32

    if use_bfloat16:
        # 检查 GPU 是否支持 bfloat16
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            print("GPU 不支持 bfloat16，使用 float16")
            return torch.float16

    return torch.float16


def print_device_info():
    """打印设备信息"""
    print("\n" + "=" * 50)
    print("设备信息")
    print("=" * 50)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        cuda_version = torch.version.cuda
        bf16_support = torch.cuda.is_bf16_supported()

        print(f"CUDA: 可用")
        print(f"GPU: {gpu_name}")
        print(f"显存: {gpu_mem:.1f} GB")
        print(f"CUDA版本: {cuda_version}")
        print(f"BFloat16支持: {'是' if bf16_support else '否'}")
    else:
        print("CUDA: 不可用")
        print("将使用 CPU 模式（速度较慢）")

    print("=" * 50)


def clear_gpu_cache():
    """清理 GPU 缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
