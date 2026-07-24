"""
BLIP-2 语义解析器模块
"""
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import re

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import model_config


class BLIP2Parser:
    """
    BLIP-2 语义解析器
    """

    def __init__(self):
        self.device = model_config.blip2_device

        if model_config.blip2_dtype == "bfloat16":
            self.dtype = torch.bfloat16
        elif model_config.blip2_dtype == "float16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        # 选择模型路径
        if model_config.use_modelscope and os.path.exists(model_config.blip2_local_dir):
            model_path = model_config.blip2_local_dir
        else:
            model_path = model_config.blip2_model_id

        print(f"[BLIP-2] 加载模型: {model_path}")

        self.processor = Blip2Processor.from_pretrained(model_path)

        if self.device == "cuda":
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=self.dtype,
            )
        else:
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
            )
            self.model = self.model.to(self.device)

        self.model.eval()
        print("[BLIP-2] 模型加载完成")

    @torch.no_grad()
    def generate_caption(self, image: Image.Image, max_length: int = 77) -> str:
        """生成图像的客观描述"""
        prompt = "a photo of"
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device, self.dtype)

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,
        )

        caption = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
        caption = re.sub(r'^a photo of\s*', '', caption, flags=re.IGNORECASE)
        return caption
