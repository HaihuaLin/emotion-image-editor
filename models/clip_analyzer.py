"""
CLIP 分析器模块
用于情感诊断和生成质量评估
"""
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import EmotionCategory, EMOTION_NAMES_CN, model_config


class CLIPAnalyzer:
    """
    CLIP 诊断与裁判员
    """

    def __init__(self):
        self.device = model_config.clip_device

        # 选择模型路径
        if model_config.use_modelscope and os.path.exists(model_config.clip_local_dir):
            model_path = model_config.clip_local_dir
        else:
            model_path = model_config.clip_model_id

        print(f"[CLIP] 加载模型: {model_path}")

        # 使用 float16 节省显存
        self.model = CLIPModel.from_pretrained(model_path, torch_dtype=torch.float16)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("[CLIP] 模型加载完成")

    @torch.no_grad()
    def analyze_emotion(self, image: Image.Image) -> Tuple[EmotionCategory, float]:
        """零样本情感分类"""
        emotion_texts = [
            f"a photo expressing {emotion.value}" for emotion in EmotionCategory
        ]

        inputs = self.processor(
            text=emotion_texts,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)
        logits = outputs.logits_per_image[0]
        probs = logits.softmax(dim=-1)

        max_idx = probs.argmax().item()
        emotion = list(EmotionCategory)[max_idx]
        confidence = probs[max_idx].item()

        return emotion, confidence

    @torch.no_grad()
    def score_candidates(
        self,
        candidates: List[Image.Image],
        target_emotion: EmotionCategory,
    ) -> List[Tuple[int, float]]:
        """评估候选图与目标情感的匹配度"""
        target_text = f"a photo expressing {target_emotion.value}"

        scores = []
        for idx, candidate in enumerate(candidates):
            inputs = self.processor(
                text=[target_text],
                images=candidate,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            outputs = self.model(**inputs)
            score = outputs.logits_per_image[0][0].item()
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
