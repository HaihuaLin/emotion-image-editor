"""
项目配置文件
"""
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum
import torch
import os

class EmotionCategory(Enum):
    """EmoSet 情感类别"""
    AWE = "awe"
    ANGER = "anger"
    CONTENTMENT = "contentment"
    DISGUST = "disgust"
    EXCITEMENT = "excitement"
    FEAR = "fear"
    JOY = "joy"
    SADNESS = "sadness"

# 情感中文名称映射
EMOTION_NAMES_CN: Dict[EmotionCategory, str] = {
    EmotionCategory.AWE: "敬畏",
    EmotionCategory.ANGER: "愤怒",
    EmotionCategory.CONTENTMENT: "满足",
    EmotionCategory.DISGUST: "厌恶",
    EmotionCategory.EXCITEMENT: "兴奋",
    EmotionCategory.FEAR: "恐惧",
    EmotionCategory.JOY: "快乐",
    EmotionCategory.SADNESS: "悲伤",
}

# EmoSet 情感视觉属性先验
EMOTION_VISUAL_ATTRIBUTES: Dict[EmotionCategory, Dict] = {
    EmotionCategory.AWE: {
        "keywords": ["majestic", "grand", "vast", "spectacular", "ethereal"],
        "brightness": "medium",
        "saturation": "medium-high",
        "contrast": "high",
    },
    EmotionCategory.ANGER: {
        "keywords": ["intense", "harsh", "aggressive", "volatile", "stormy"],
        "brightness": "low-medium",
        "saturation": "high",
        "contrast": "high",
    },
    EmotionCategory.CONTENTMENT: {
        "keywords": ["peaceful", "serene", "calm", "gentle", "harmonious"],
        "brightness": "medium",
        "saturation": "medium",
        "contrast": "medium",
    },
    EmotionCategory.DISGUST: {
        "keywords": ["repulsive", "decaying", "foul", "unsanitary"],
        "brightness": "low",
        "saturation": "low-medium",
        "contrast": "medium",
    },
    EmotionCategory.EXCITEMENT: {
        "keywords": ["energetic", "vibrant", "dynamic", "thrilling", "vivid"],
        "brightness": "high",
        "saturation": "high",
        "contrast": "high",
    },
    EmotionCategory.FEAR: {
        "keywords": ["ominous", "eerie", "dark", "threatening", "suspenseful"],
        "brightness": "low",
        "saturation": "low",
        "contrast": "high",
    },
    EmotionCategory.JOY: {
        "keywords": ["bright", "cheerful", "vibrant", "warm", "radiant"],
        "brightness": "high",
        "saturation": "high",
        "contrast": "medium-high",
    },
    EmotionCategory.SADNESS: {
        "keywords": ["somber", "melancholic", "gloomy", "desolate", "muted"],
        "brightness": "low",
        "saturation": "low",
        "contrast": "low",
    },
}

@dataclass
class ModelConfig:
    """模型配置 - 使用魔搭（ModelScope）模型"""
    # 模型来源：魔搭
    use_modelscope: bool = True

    # BLIP-2 语义解析器（魔搭）
    blip2_model_id: str = "goldslj/blip2-opt-2.7b"
    blip2_local_dir: str = "/mnt/user/models/modelscope/goldslj--blip2-opt-2.7b"
    blip2_device: str = "auto"  # 自动检测
    blip2_dtype: str = "bfloat16"

    # CLIP 诊断与裁判（魔搭）
    clip_model_id: str = "AI-ModelScope/CLIP-GmP-ViT-L-14"
    clip_local_dir: str = "/mnt/user/models/modelscope/AI-ModelScope--CLIP-GmP-ViT-L-14"
    clip_device: str = "auto"  # 自动检测

    # Stable Diffusion + ControlNet 生成器（魔搭）
    sd_model_id: str = "AI-ModelScope/stable-diffusion-v1.5-no-safetensor"
    sd_local_dir: str = "/mnt/user/models/modelscope/AI-ModelScope--stable-diffusion-v1.5-no-safetensor"
    controlnet_model_id: str = "AI-ModelScope/sd-controlnet-canny"
    controlnet_local_dir: str = "/mnt/user/models/modelscope/AI-ModelScope--sd-controlnet-canny"
    sd_device: str = "auto"  # 自动检测
    sd_dtype: str = "bfloat16"

    # ControlNet 参数
    controlnet_conditioning_scale: float = 0.85

    # 生成参数
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    num_candidates: int = 3  # 生成候选图数量

    def __post_init__(self):
        """初始化后处理：解析自动设备，检查本地模型路径"""
        # 检测可用设备
        if torch.cuda.is_available():
            default_device = "cuda"
        else:
            default_device = "cpu"

        # 解析设备配置
        if self.blip2_device == "auto":
            self.blip2_device = default_device
        if self.clip_device == "auto":
            self.clip_device = default_device
        if self.sd_device == "auto":
            self.sd_device = default_device

        # 如果是CPU，强制使用float32
        if default_device == "cpu":
            self.blip2_dtype = "float32"
            self.sd_dtype = "float32"

        # 检查本地模型路径是否存在
        if self.use_modelscope:
            self._check_model_paths()

    def _check_model_paths(self):
        """检查魔搭模型本地路径"""
        paths_to_check = {
            "BLIP-2": self.blip2_local_dir,
            "CLIP": self.clip_local_dir,
            "SD": self.sd_local_dir,
            "ControlNet": self.controlnet_local_dir,
        }

        missing = []
        for name, path in paths_to_check.items():
            if not os.path.exists(path):
                missing.append(f"{name}: {path}")

        if missing:
            print("\n⚠️  警告：以下模型路径不存在，请先运行 download_models.sh 下载模型：")
            for m in missing:
                print(f"   - {m}")
            print("\n下载命令: bash download_models.sh\n")

@dataclass
class AppConfig:
    """应用配置"""
    project_name: str = "EmoEdit - 情感驱动图像编辑系统"
    version: str = "1.0.0"
    max_image_size: int = 1024
    sample_images_dir: str = "assets/samples"
    results_dir: str = "assets/results"
    server_port: int = 7860
    server_share: bool = True

# 全局配置实例
model_config = ModelConfig()
app_config = AppConfig()
