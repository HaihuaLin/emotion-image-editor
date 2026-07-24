"""
图像处理工具
"""
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from typing import Tuple, Optional
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import app_config


class ImageProcessor:
    """
    图像处理工具类
    提供图像预处理、后处理、保存等功能
    """

    @staticmethod
    def load_image(image_path: str) -> Image.Image:
        """加载图像"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        return Image.open(image_path).convert("RGB")

    @staticmethod
    def resize_image(
        image: Image.Image,
        max_size: int = None,
        maintain_aspect: bool = True,
    ) -> Image.Image:
        """
        调整图像大小
        """
        max_size = max_size or app_config.max_image_size

        if maintain_aspect:
            image.thumbnail((max_size, max_size), Image.LANCZOS)
        else:
            image = image.resize((max_size, max_size), Image.LANCZOS)

        return image

    @staticmethod
    def center_crop(
        image: Image.Image,
        target_size: Tuple[int, int] = (512, 512),
    ) -> Image.Image:
        """
        中心裁剪到目标尺寸
        """
        width, height = image.size
        target_w, target_h = target_size

        # 计算裁剪区域
        left = (width - target_w) // 2
        top = (height - target_h) // 2
        right = left + target_w
        bottom = top + target_h

        return image.crop((left, top, right, bottom))

    @staticmethod
    def adjust_brightness(
        image: Image.Image,
        factor: float = 1.0,
    ) -> Image.Image:
        """调整亮度"""
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    @staticmethod
    def adjust_saturation(
        image: Image.Image,
        factor: float = 1.0,
    ) -> Image.Image:
        """调整饱和度"""
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)

    @staticmethod
    def adjust_contrast(
        image: Image.Image,
        factor: float = 1.0,
    ) -> Image.Image:
        """调整对比度"""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    @staticmethod
    def enhance_image(
        image: Image.Image,
        brightness: float = 1.0,
        saturation: float = 1.0,
        contrast: float = 1.0,
    ) -> Image.Image:
        """
        综合调整图像
        """
        image = ImageProcessor.adjust_brightness(image, brightness)
        image = ImageProcessor.adjust_saturation(image, saturation)
        image = ImageProcessor.adjust_contrast(image, contrast)
        return image

    @staticmethod
    def save_image(
        image: Image.Image,
        save_path: str,
        quality: int = 95,
    ) -> str:
        """
        保存图像
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path, quality=quality)
        return save_path

    @staticmethod
    def create_comparison_grid(
        images: list,
        labels: list = None,
        cols: int = 3,
        padding: int = 10,
        bg_color: str = "white",
    ) -> Image.Image:
        """
        创建对比网格图
        """
        if not images:
            return Image.new("RGB", (100, 100), bg_color)

        n = len(images)
        rows = (n + cols - 1) // cols

        # 获取最大尺寸
        max_w = max(img.width for img in images)
        max_h = max(img.height for img in images)

        # 计算网格尺寸
        grid_w = cols * (max_w + padding) + padding
        grid_h = rows * (max_h + padding + 30) + padding  # 30 for labels

        # 创建画布
        grid = Image.new("RGB", (grid_w, grid_h), bg_color)

        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols

            x = padding + col * (max_w + padding)
            y = padding + row * (max_h + padding + 30)

            # 粘贴图像
            grid.paste(img, (x, y))

        return grid

    @staticmethod
    def image_to_numpy(image: Image.Image) -> np.ndarray:
        """PIL图像转NumPy数组"""
        return np.array(image)

    @staticmethod
    def numpy_to_image(array: np.ndarray) -> Image.Image:
        """NumPy数组转PIL图像"""
        return Image.fromarray(array)
