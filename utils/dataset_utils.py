"""
EmoSet 数据集工具
用于加载和管理EmoSet数据集中的图片
"""
import os
import random
from typing import List, Dict, Tuple, Optional
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import EmotionCategory


class EmoSetLoader:
    """
    EmoSet 数据集加载器
    """

    # EmoSet 文件名前缀到情感的映射
    EMOTION_MAPPING = {
        "amusement": EmotionCategory.JOY,
        "awe": EmotionCategory.AWE,
        "contentment": EmotionCategory.CONTENTMENT,
        "disgust": EmotionCategory.DISGUST,
        "excitement": EmotionCategory.EXCITEMENT,
        "fear": EmotionCategory.FEAR,
        "anger": EmotionCategory.ANGER,
        "sadness": EmotionCategory.SADNESS,
    }

    def __init__(self, dataset_dir: str = "/mnt/workspace/data/emoset"):
        """
        初始化数据集加载器

        Args:
            dataset_dir: EmoSet 数据集目录路径
        """
        self.dataset_dir = dataset_dir
        self.image_files = []
        self.emotion_index = {}  # 按情感分类索引

        if os.path.exists(dataset_dir):
            self._build_index()
        else:
            print(f"警告: 数据集目录不存在: {dataset_dir}")

    def _build_index(self):
        """构建数据集索引"""
        print(f"[EmoSet] 正在扫描数据集: {self.dataset_dir}")

        # 扫描所有图片文件
        for filename in os.listdir(self.dataset_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                filepath = os.path.join(self.dataset_dir, filename)
                self.image_files.append(filepath)

                # 解析情感标签
                emotion = self._parse_emotion(filename)
                if emotion:
                    if emotion not in self.emotion_index:
                        self.emotion_index[emotion] = []
                    self.emotion_index[emotion].append(filepath)

        # 统计信息
        print(f"[EmoSet] 扫描完成:")
        print(f"  - 总图片数: {len(self.image_files)}")
        for emotion, files in self.emotion_index.items():
            print(f"  - {emotion.value}: {len(files)} 张")

    def _parse_emotion(self, filename: str) -> Optional[EmotionCategory]:
        """从文件名解析情感标签"""
        # 文件名格式: amusement_00000.jpg
        prefix = filename.split("_")[0].lower()
        return self.EMOTION_MAPPING.get(prefix)

    def get_all_images(self) -> List[str]:
        """获取所有图片路径"""
        return self.image_files

    def get_images_by_emotion(self, emotion: EmotionCategory) -> List[str]:
        """获取指定情感的所有图片"""
        return self.emotion_index.get(emotion, [])

    def get_random_image(self, emotion: Optional[EmotionCategory] = None) -> Tuple[str, EmotionCategory]:
        """
        随机获取一张图片

        Args:
            emotion: 指定情感（可选）

        Returns:
            (图片路径, 情感类别)
        """
        if emotion and emotion in self.emotion_index:
            files = self.emotion_index[emotion]
        else:
            files = self.image_files

        if not files:
            raise ValueError("没有可用的图片")

        filepath = random.choice(files)
        parsed_emotion = self._parse_emotion(os.path.basename(filepath))
        return filepath, parsed_emotion

    def get_sample_images(self, n: int = 8) -> List[Tuple[str, EmotionCategory]]:
        """
        获取每种情感的样本图片

        Args:
            n: 每种情感返回的图片数量

        Returns:
            [(图片路径, 情感类别), ...]
        """
        samples = []
        for emotion in EmotionCategory:
            files = self.get_images_by_emotion(emotion)
            if files:
                selected = random.sample(files, min(n, len(files)))
                for f in selected:
                    samples.append((f, emotion))
        return samples

    def load_image(self, filepath: str) -> Image.Image:
        """加载图片"""
        return Image.open(filepath).convert("RGB")

    def get_statistics(self) -> Dict[str, int]:
        """获取数据集统计信息"""
        stats = {
            "total": len(self.image_files),
        }
        for emotion in EmotionCategory:
            stats[emotion.value] = len(self.emotion_index.get(emotion, []))
        return stats


# 全局实例
emoset_loader = EmoSetLoader()
