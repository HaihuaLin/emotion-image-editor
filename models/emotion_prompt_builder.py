"""
情感提示词构建器
结合EmoSet视觉属性先验和BLIP-2客观描述，生成精准的提示词
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List
from configs.config import (
    EmotionCategory,
    EMOTION_VISUAL_ATTRIBUTES,
    EMOTION_NAMES_CN,
)


class EmotionPromptBuilder:
    """
    情感提示词构建器
    - 将BLIP-2的客观描述与目标情绪自动拼接
    - 注入EmoSet验证过的视觉特征词
    """

    def __init__(self):
        self.emotion_attributes = EMOTION_VISUAL_ATTRIBUTES
        self.emotion_names = EMOTION_NAMES_CN

    def build_prompt(
        self,
        objective_description: str,
        target_emotion: EmotionCategory,
        include_visual_attributes: bool = True,
        negative_prompt: bool = True,
    ) -> Dict[str, str]:
        """
        构建完整的提示词对（正向+负向）

        Args:
            objective_description: BLIP-2生成的客观描述
            target_emotion: 目标情感类别
            include_visual_attributes: 是否包含视觉属性描述
            negative_prompt: 是否生成负向提示词

        Returns:
            包含positive和negative提示词的字典
        """
        attributes = self.emotion_attributes[target_emotion]
        emotion_cn = self.emotion_names[target_emotion]

        # 构建正向提示词
        prompt_parts = [objective_description.strip()]

        if include_visual_attributes:
            # 添加情感关键词
            keywords = ", ".join(attributes["keywords"][:3])  # 取前3个关键词
            prompt_parts.append(f"expressing {target_emotion.value}")

            # 添加视觉属性
            visual_attrs = self._build_visual_attribute_text(attributes)
            if visual_attrs:
                prompt_parts.append(visual_attrs)

            # 添加情绪中文标签（可选，用于强调）
            prompt_parts.append(f"{emotion_cn} emotion")

        positive_prompt = ", ".join(prompt_parts)

        # 构建负向提示词
        negative = ""
        if negative_prompt:
            negative = self._build_negative_prompt(target_emotion)

        return {
            "positive": positive_prompt,
            "negative": negative,
        }

    def _build_visual_attribute_text(self, attributes: Dict) -> str:
        """
        根据视觉属性构建文本描述
        """
        parts = []

        brightness_map = {
            "low": "dim lighting, low brightness",
            "low-medium": "slightly dim lighting",
            "medium": "balanced lighting",
            "medium-high": "slightly bright lighting",
            "high": "bright lighting, high brightness",
        }

        saturation_map = {
            "low": "desaturated colors, muted tones",
            "low-medium": "slightly desaturated",
            "medium": "natural color saturation",
            "medium-high": "vivid colors",
            "high": "highly saturated, vibrant colors",
        }

        contrast_map = {
            "low": "low contrast, soft tones",
            "medium": "balanced contrast",
            "medium-high": "enhanced contrast",
            "high": "high contrast, dramatic lighting",
        }

        brightness = attributes.get("brightness", "medium")
        saturation = attributes.get("saturation", "medium")
        contrast = attributes.get("contrast", "medium")

        if brightness in brightness_map:
            parts.append(brightness_map[brightness])
        if saturation in saturation_map:
            parts.append(saturation_map[saturation])
        if contrast in contrast_map:
            parts.append(contrast_map[contrast])

        return ", ".join(parts)

    def _build_negative_prompt(self, target_emotion: EmotionCategory) -> str:
        """
        构建负向提示词，排除与目标情感相反的特征
        """
        # 基础负向提示词
        base_negative = [
            "blurry", "low quality", "distorted", "deformed",
            "watermark", "text", "signature", "logo",
        ]

        # 根据目标情感排除相反的视觉特征
        emotion_opposites = {
            EmotionCategory.JOY: ["dark", "gloomy", "sad", "muted colors"],
            EmotionCategory.SADNESS: ["bright", "cheerful", "vibrant", "happy"],
            EmotionCategory.FEAR: ["safe", "comfortable", "peaceful"],
            EmotionCategory.ANGER: ["calm", "peaceful", "serene"],
            EmotionCategory.EXCITEMENT: ["calm", "still", "static"],
            EmotionCategory.AWE: ["ordinary", "mundane", "plain"],
            EmotionCategory.CONTENTMENT: ["chaotic", "stressful", "tense"],
            EmotionCategory.DISGUST: ["clean", "beautiful", "attractive"],
        }

        negatives = base_negative + emotion_opposites.get(target_emotion, [])
        return ", ".join(negatives)

    def build_batch_prompts(
        self,
        objective_description: str,
        target_emotions: List[EmotionCategory],
    ) -> Dict[str, Dict[str, str]]:
        """
        批量构建多个情感的提示词
        """
        prompts = {}
        for emotion in target_emotions:
            prompts[emotion.value] = self.build_prompt(
                objective_description,
                emotion,
            )
        return prompts

    def get_emotion_style_guide(self, emotion: EmotionCategory) -> str:
        """
        获取情感风格指南（用于调试和展示）
        """
        attributes = self.emotion_attributes[emotion]
        cn_name = self.emotion_names[emotion]

        guide = f"""
情感: {cn_name} ({emotion.value})
关键词: {', '.join(attributes['keywords'])}
亮度: {attributes['brightness']}
饱和度: {attributes['saturation']}
对比度: {attributes['contrast']}
"""
        return guide
