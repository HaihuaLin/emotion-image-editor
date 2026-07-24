#!/bin/bash
# ============================================================
# 阿里云 DSW 环境初始化脚本
# 在 Notebook 中运行: !bash dsw_setup.sh
# ============================================================

set -e

echo "=========================================="
echo "EmoEdit DSW 环境初始化"
echo "=========================================="

# 1. 创建持久化目录结构
echo ""
echo "[1/6] 创建持久化目录..."
mkdir -p /mnt/user/data/emoset
mkdir -p /mnt/user/data/sample_images
mkdir -p /mnt/user/models/huggingface
mkdir -p /mnt/user/models/modelscope
mkdir -p /mnt/user/models/torch
mkdir -p /mnt/user/results/emoedit
echo "✓ 目录创建完成"

# 2. 设置环境变量
echo ""
echo "[2/6] 配置环境变量..."
export HF_HOME=/mnt/user/models/huggingface
export TRANSFORMERS_CACHE=/mnt/user/models/huggingface
export HF_DATASETS_CACHE=/mnt/user/models/huggingface/datasets
export TORCH_HOME=/mnt/user/models/torch
echo "✓ 环境变量配置完成"

# 3. 安装依赖（如果需要）
echo ""
echo "[3/6] 检查并安装依赖..."
pip install -q --upgrade pip
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -q transformers diffusers accelerate
pip install -q gradio Pillow opencv-python matplotlib tqdm
pip install -q modelscope  # 魔搭SDK
echo "✓ 依赖安装完成"

# 4. 从魔搭下载模型
echo ""
echo "[4/6] 从魔搭下载模型..."
echo "下载 CLIP 模型..."
python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('AI-ModelScope/CLIP-GmP-ViT-L-14', cache_dir='/mnt/user/models/modelscope')
print('✓ CLIP 模型下载完成')
"

echo "下载 BLIP-2 模型..."
python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('goldslj/blip2-opt-2.7b', cache_dir='/mnt/user/models/modelscope')
print('✓ BLIP-2 模型下载完成')
"

echo "下载 Stable Diffusion 模型..."
python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('AI-ModelScope/stable-diffusion-v1.5-no-safetensor', cache_dir='/mnt/user/models/modelscope')
print('✓ Stable Diffusion 模型下载完成')
"

echo "下载 ControlNet 模型..."
python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('AI-ModelScope/sd-controlnet-canny', cache_dir='/mnt/user/models/modelscope')
print('✓ ControlNet 模型下载完成')
"

echo "✓ 所有模型下载完成"

# 5. 下载深度估计模型（从魔搭或HuggingFace）
echo ""
echo "[5/6] 下载深度估计模型..."
python -c "
from modelscope.hub.snapshot_download import snapshot_download
try:
    snapshot_download('Intel/dpt-large', cache_dir='/mnt/user/models/modelscope')
    print('✓ 深度估计模型下载完成')
except:
    print('从HuggingFace下载深度估计模型...')
    from transformers import DPTFeatureExtractor, DPTForDepthEstimation
    DPTFeatureExtractor.from_pretrained('Intel/dpt-large')
    DPTForDepthEstimation.from_pretrained('Intel/dpt-large')
    print('✓ 深度估计模型下载完成')
"

# 6. 验证安装
echo ""
echo "[6/6] 验证安装..."
python -c "
import torch
import transformers
import diffusers
import gradio
import modelscope

print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Transformers: {transformers.__version__}')
print(f'Diffusers: {diffusers.__version__}')
print(f'Gradio: {gradio.__version__}')
print(f'ModelScope: {modelscope.__version__}')

# 检查模型路径
import os
model_paths = [
    '/mnt/user/models/modelscope/AI-ModelScope--CLIP-GmP-ViT-L-14',
    '/mnt/user/models/modelscope/goldslj--blip2-opt-2.7b',
    '/mnt/user/models/modelscope/AI-ModelScope--stable-diffusion-v1.5-no-safetensor',
    '/mnt/user/models/modelscope/AI-ModelScope--sd-controlnet-canny',
]
print('\n模型路径检查:')
for path in model_paths:
    exists = os.path.exists(path)
    status = '✓' if exists else '✗'
    print(f'  {status} {path}')
"

echo ""
echo "=========================================="
echo "✓ 环境初始化完成！"
echo ""
echo "启动命令:"
echo "  cd /home/work/emotion-image-editor"
echo "  python dsw_launcher.py"
echo "=========================================="
