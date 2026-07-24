"""
Stable Diffusion 图像生成器
结合新情绪提示词进行重绘
"""
import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Dict
from diffusers import (
    StableDiffusionPipeline,
    UniPCMultistepScheduler,
)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import model_config


class StableDiffusionGenerator:
    """
    跨情感画师
    - 结合新情绪提示词进行重绘
    """

    def __init__(self):
        self.device = model_config.sd_device
        # 根据配置的 dtype 字符串获取 torch.dtype
        if model_config.sd_dtype == "bfloat16":
            self.dtype = torch.bfloat16
        elif model_config.sd_dtype == "float16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        print(f"[SD] 使用设备: {self.device}, 数据类型: {self.dtype}")

        # 选择SD模型路径
        if model_config.use_modelscope and os.path.exists(model_config.sd_local_dir):
            sd_path = model_config.sd_local_dir
            print(f"[SD] 从魔搭加载Stable Diffusion: {sd_path}")
        else:
            sd_path = model_config.sd_model_id
            print(f"[SD] 从HuggingFace加载Stable Diffusion: {sd_path}")

        # 根据设备选择加载方式
        if self.device == "cuda":
            self.pipe = StableDiffusionPipeline.from_pretrained(
                sd_path,
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            # 启用内存优化
            self.pipe.enable_model_cpu_offload()
        else:
            # CPU模式
            self.pipe = StableDiffusionPipeline.from_pretrained(
                sd_path,
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            self.pipe = self.pipe.to(self.device)

        # 使用高效的调度器
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        print("[SD] 模型加载完成")

    def generate_candidates(
        self,
        prompt: str,
        negative_prompt: str,
        num_candidates: int = None,
        num_inference_steps: int = None,
        guidance_scale: float = None,
        width: int = 512,
        height: int = 512,
    ) -> List[Image.Image]:
        """
        生成多张候选图

        Args:
            prompt: 正向提示词
            negative_prompt: 负向提示词
            num_candidates: 候选图数量
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            width: 输出宽度
            height: 输出高度

        Returns:
            候选图列表
        """
        num_candidates = num_candidates or model_config.num_candidates
        num_inference_steps = num_inference_steps or model_config.num_inference_steps
        guidance_scale = guidance_scale or model_config.guidance_scale

        # 设置随机种子以获得多样性
        generator = torch.Generator(device=self.device)

        candidates = []
        for i in range(num_candidates):
            print(f"[SD] 生成第 {i+1}/{num_candidates} 张候选图...")
            generator.manual_seed(i * 42)  # 不同的种子

            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                width=width,
                height=height,
            )

            candidates.append(result.images[0])

        return candidates

    def generate_single(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int = 42,
        **kwargs,
    ) -> Image.Image:
        """
        生成单张图像（用于快速测试）
        """
        candidates = self.generate_candidates(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_candidates=1,
            **kwargs,
        )
        return candidates[0]
