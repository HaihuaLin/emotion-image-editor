"""
魔搭（ModelScope）模型配置
用于从魔搭平台下载模型
"""

# ============================================================
# 魔搭模型ID配置
# ============================================================

MODELS = {
    # CLIP 模型 - 用于情感诊断和评分
    "clip": {
        "model_id": "AI-ModelScope/CLIP-GmP-ViT-L-14",
        "local_dir": "/mnt/workspace/models/modelscope/AI-ModelScope--CLIP-GmP-ViT-L-14/snapshots",
        "description": "CLIP 模型，用于零样本情感分类和生成质量评估",
    },

    # BLIP-2 模型 - 用于语义解析
    "blip2": {
        "model_id": "goldsj/blip2-opt-2.7b",
        "local_dir": "/mnt/workspace/models/modelscope/goldsj--blip2-opt-2.7b/snapshots",
        "description": "BLIP-2 模型，用于生成客观场景描述",
    },

    # Stable Diffusion 模型 - 用于图像生成
    "sd": {
        "model_id": "AI-ModelScope/stable-diffusion-v1.5-no-safetensor",
        "local_dir": "/mnt/workspace/models/modelscope/AI-ModelScope--stable-diffusion-v1.5-no-safetensor/snapshots",
        "description": "Stable Diffusion v1.5，用于图像生成",
    },

    # ControlNet 模型 - 用于结构控制（必须与SD 1.5匹配）
    "controlnet": {
        "model_id": "lllyasviel/sd-controlnet-depth",
        "local_dir": "/mnt/workspace/models/huggingface/controlnet-depth",
        "description": "ControlNet depth 模型，用于保持图像结构（兼容SD 1.5）",
    },
}
