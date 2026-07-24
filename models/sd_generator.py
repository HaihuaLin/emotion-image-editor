"""
Stable Diffusion 图像生成器 - img2img模式
在原图基础上修改情感氛围
"""
import torch
from PIL import Image
from typing import List
from diffusers import StableDiffusionImg2ImgPipeline, UniPCMultistepScheduler

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import model_config


class StableDiffusionGenerator:
    def __init__(self):
        self.device = model_config.sd_device
        self.dtype = torch.bfloat16 if model_config.sd_dtype == "bfloat16" else torch.float16

        model_path = model_config.sd_local_dir if os.path.exists(model_config.sd_local_dir) else model_config.sd_model_id
        print(f"[SD] 加载: {model_path}")

        # 使用 img2img 管道
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)

        print("[SD] 完成")

    def generate_candidates(
        self,
        prompt: str,
        negative_prompt: str,
        init_image: Image.Image = None,
        strength: float = 0.65,
        num_candidates: int = 3,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
    ) -> List[Image.Image]:
        """
        基于原图生成候选图（img2img）
        strength: 0=完全不变, 1=完全重绘, 0.65=保持结构但改变风格
        """
        num_candidates = num_candidates or model_config.num_candidates
        num_inference_steps = num_inference_steps or model_config.num_inference_steps
        guidance_scale = guidance_scale or model_config.guidance_scale

        # 预处理原图
        if init_image is not None:
            init_image = init_image.convert("RGB")
            init_image = init_image.resize((width, height), Image.LANCZOS)

        generator = torch.Generator(device=self.device)

        candidates = []
        for i in range(num_candidates):
            print(f"[SD] 生成第 {i+1}/{num_candidates} 张...")
            generator.manual_seed(i * 42 + 100)

            if init_image is not None:
                # img2img: 基于原图修改
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=init_image,
                    strength=strength,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
            else:
                # txt2img: 纯文生图（备用）
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
