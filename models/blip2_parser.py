"""
BLIP-2 语义解析器模块
剥离情感色彩，提取纯客观画面描述
"""
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import re
from typing import Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import model_config


class BLIP2Parser:
    """
    BLIP-2 语义解析器
    - 对原图进行视觉问答
    - 生成不带感情色彩的客观描述
    """

    def __init__(self):
        self.device = model_config.blip2_device
        # 根据配置的 dtype 字符串获取 torch.dtype
        if model_config.blip2_dtype == "bfloat16":
            self.dtype = torch.bfloat16
        elif model_config.blip2_dtype == "float16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        # 选择模型路径：优先使用本地模型
        if model_config.use_modelscope and os.path.exists(model_config.blip2_local_dir):
            model_path = model_config.blip2_local_dir
            print(f"[BLIP-2] 从魔搭加载模型: {model_path}")
        else:
            model_path = model_config.blip2_model_id
            print(f"[BLIP-2] 从HuggingFace加载模型: {model_path}")

        print(f"[BLIP-2] 使用设备: {self.device}, 数据类型: {self.dtype}")

        # 加载处理器
        self.processor = Blip2Processor.from_pretrained(model_path)

        # 加载模型
        if self.device == "cuda":
            # GPU模式：使用 device_map 自动分配
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=self.dtype,
            )
        else:
            # CPU模式：直接加载到CPU
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
            )
            self.model = self.model.to(self.device)

        self.model.eval()
        print("[BLIP-2] 模型加载完成")

    @torch.no_grad()
    def generate_caption(
        self,
        image: Image.Image,
        max_length: int = 77,
        num_beams: int = 5,
    ) -> str:
        """
        生成图像的客观描述
        使用视觉问答模式，提示模型描述场景
        """
        # 使用特定的提示词引导模型生成客观描述
        prompt = "a photo of"
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            truncation=True,
            max_length=64,
        ).to(self.device, self.dtype)

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )

        caption = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
        # 去除提示词前缀
        caption = re.sub(r'^a photo of\s*', '', caption, flags=re.IGNORECASE)
        return caption

    @torch.no_grad()
    def visual_question_answering(
        self,
        image: Image.Image,
        question: str,
        max_length: int = 100,
    ) -> str:
        """
        视觉问答
        可用于更精细的场景理解
        """
        inputs = self.processor(
            images=image,
            text=question,
            return_tensors="pt",
            truncation=True,
            max_length=64,
        ).to(self.device, self.dtype)

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,
        )

        answer = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
        return answer

    def get_structured_description(self, image: Image.Image) -> dict:
        """
        获取结构化的场景描述
        返回包含主体、动作、场景、物体等信息的字典
        """
        # 基础描述
        caption = self.generate_caption(image)

        # 使用VQA获取更多细节
        scene_type = self.visual_question_answering(
            image,
            "What type of scene is this?"
        )

        objects = self.visual_question_answering(
            image,
            "What are the main objects in this image?"
        )

        return {
            "caption": caption,
            "scene_type": scene_type,
            "objects": objects,
        }

    def get_objective_prompt(self, image: Image.Image) -> str:
        """
        生成用于图像生成的客观提示词
        剥离所有情感色彩，只保留物理描述
        """
        caption = self.generate_caption(image)

        # 移除可能的情感词汇
        emotion_words = [
            "sad", "happy", "angry", "fearful", "beautiful", "ugly",
            "gloomy", "cheerful", "dark", "bright", "emotional",
            "lonely", "joyful", "melancholic", "vibrant", "dull"
        ]

        clean_caption = caption
        for word in emotion_words:
            clean_caption = re.sub(
                r'\b' + word + r'\b',
                '',
                clean_caption,
                flags=re.IGNORECASE
            )

        # 清理多余的空格和标点
        clean_caption = re.sub(r'\s+', ' ', clean_caption).strip()
        clean_caption = re.sub(r'[,.]+$', '', clean_caption).strip()

        return clean_caption
