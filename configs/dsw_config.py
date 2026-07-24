"""
阿里云 DSW 服务器专用配置
"""
import os
import torch

# ============================================================
# 阿里云 DSW 路径配置
# ============================================================

# DSW 主目录（持久化存储）
DSW_PERSISTENT_ROOT = "/mnt/workspace"

# 数据和模型存储路径（持久化，重启不丢失）
DATA_ROOT = os.path.join(DSW_PERSISTENT_ROOT, "data")
MODEL_CACHE_DIR = os.path.join(DSW_PERSISTENT_ROOT, "models")
RESULTS_DIR = os.path.join(DSW_PERSISTENT_ROOT, "results", "emoedit")

# EmoSet 数据集路径（用户数据盘）
EMOSET_DIR = "/data/emoset"

# Hugging Face 模型缓存目录
HF_CACHE_DIR = os.path.join(MODEL_CACHE_DIR, "huggingface")

# 项目工作目录（代码目录）
WORK_DIR = os.path.join(DSW_PERSISTENT_ROOT, "emotion-image-editor")

# ============================================================
# 目录创建
# ============================================================

def ensure_directories():
    """确保所有必要的目录存在"""
    dirs_to_create = [
        DATA_ROOT,
        MODEL_CACHE_DIR,
        RESULTS_DIR,
        EMOSET_DIR,
        HF_CACHE_DIR,
    ]

    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ 目录就绪: {dir_path}")


# ============================================================
# 环境变量设置（在模型加载前调用）
# ============================================================

def setup_environment():
    """设置 DSW 环境变量"""
    # 设置 Hugging Face 缓存目录
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
    os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")

    # 设置 PyTorch 缓存
    os.environ["TORCH_HOME"] = os.path.join(MODEL_CACHE_DIR, "torch")

    print(f"\n环境变量已配置:")
    print(f"  HF_HOME: {os.environ.get('HF_HOME')}")
    print(f"  TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE')}")


# ============================================================
# GPU 信息
# ============================================================

def print_gpu_info():
    """打印 GPU 信息"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"\nGPU 信息:")
        print(f"  设备: {gpu_name}")
        print(f"  显存: {gpu_mem:.1f} GB")
        print(f"  CUDA 版本: {torch.version.cuda}")
    else:
        print("\n警告: 未检测到 GPU")


# ============================================================
# 快速测试数据集路径
# ============================================================

# 从 Hugging Face 下载 EmoSet 的简化版
EMOSET_HF_REPO = "xiangliangpro/emoset"

# 测试用样例图片路径（可以放在持久化存储）
SAMPLE_IMAGES_DIR = os.path.join(DATA_ROOT, "sample_images")
