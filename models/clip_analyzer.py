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
    - 前期：评估原图情绪
    - 后期：给生成的候选图打分并挑选最佳结果
    """

    def __init__(self):
        self.device = model_config.clip_device

        # 选择模型路径：优先使用魔搭本地模型
        if model_config.use_modelscope and os.path.exists(model_config.clip_local_dir):
            model_path = model_config.clip_local_dir
            print(f"[CLIP] 从魔搭加载模型: {model_path}")
        else:
            model_path = model_config.clip_model_id
            print(f"[CLIP] 从HuggingFace加载模型: {model_path}")

        print(f"[CLIP] 使用设备: {self.device}")

        # 加载模型
        self.model = CLIPModel.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model.eval()
        print("[CLIP] 模型加载完成")

    @torch.no_grad()
    def analyze_emotion(self, image: Image.Image) -> Tuple[EmotionCategory, float]:
        """
        零样本情感分类
        返回: (主导情感类别, 置信度)
        """
        # 构建情感文本描述
        emotion_texts = [
            f"a photo expressing {emotion.value}" for emotion in EmotionCategory
        ]

        # 预处理
        inputs = self.processor(
            text=emotion_texts,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)

        # 推理
        outputs = self.model(**inputs)
        logits = outputs.logits_per_image[0]
        probs = logits.softmax(dim=-1)

        # 获取最高置信度的情感
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
        """
        评估候选图与目标情感的匹配度
        返回: [(索引, 分数), ...] 按分数降序排列
        """
        # 目标情感文本
        target_text = f"a photo expressing {target_emotion.value}"

        scores = []
        for idx, candidate in enumerate(candidates):
            inputs = self.processor(
                text=[target_text],
                images=candidate,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(self.device)

            outputs = self.model(**inputs)
            # 使用余弦相似度作为分数
            score = outputs.logits_per_image[0][0].item()
            scores.append((idx, score))

        # 按分数降序排列
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def get_best_candidate(
        self,
        candidates: List[Image.Image],
        target_emotion: EmotionCategory,
    ) -> Tuple[Image.Image, float]:
        """
        返回最佳候选图及其分数
        """
        scores = self.score_candidates(candidates, target_emotion)
        best_idx, best_score = scores[0]
        return candidates[best_idx], best_score

    def batch_analyze(self, images: List[Image.Image]) -> List[Tuple[EmotionCategory, float]]:
        """
        批量分析多张图片的情感
        """
        results = []
        for image in images:
            emotion, confidence = self.analyze_emotion(image)
            results.append((emotion, confidence))
        return results
