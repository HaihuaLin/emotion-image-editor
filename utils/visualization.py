"""
可视化工具
用于展示情感分析结果和生成过程
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image
from typing import List, Dict, Optional

# 使用非交互式后端
matplotlib.use('Agg')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import EmotionCategory, EMOTION_NAMES_CN, EMOTION_VISUAL_ATTRIBUTES


class EmotionVisualizer:
    """
    情感可视化工具
    """

    def __init__(self):
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def plot_emotion_radar(
        self,
        emotion_scores: Dict[EmotionCategory, float],
        title: str = "情感分布雷达图",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        绘制情感分布雷达图
        """
        categories = list(emotion_scores.keys())
        values = list(emotion_scores.values())

        # 添加闭合点
        values += values[:1]

        # 计算角度
        angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
        angles += angles[:1]

        # 创建极坐标图
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.plot(angles, values, 'o-', linewidth=2, label='情感分数')
        ax.fill(angles, values, alpha=0.25)

        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([EMOTION_NAMES_CN[c] for c in categories])
        ax.set_title(title, size=16, fontweight='bold')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_emotion_bar(
        self,
        emotion_scores: Dict[EmotionCategory, float],
        title: str = "情感置信度条形图",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        绘制情感置信度条形图
        """
        categories = [EMOTION_NAMES_CN[c] for c in emotion_scores.keys()]
        values = list(emotion_scores.values())

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(categories)))
        bars = ax.bar(categories, values, color=colors)

        # 添加数值标签
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.2%}',
                ha='center',
                va='bottom',
            )

        ax.set_ylabel('置信度')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_generation_comparison(
        self,
        original: Image.Image,
        candidates: List[Image.Image],
        scores: List[float],
        best_idx: int = 0,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        绘制生成结果对比图
        """
        n_candidates = len(candidates)
        fig, axes = plt.subplots(1, n_candidates + 1, figsize=(4 * (n_candidates + 1), 4))

        # 原图
        axes[0].imshow(original)
        axes[0].set_title('原图', fontsize=12)
        axes[0].axis('off')

        # 候选图
        for idx, (candidate, score) in enumerate(zip(candidates, scores)):
            axes[idx + 1].imshow(candidate)
            color = 'green' if idx == best_idx else 'black'
            weight = 'bold' if idx == best_idx else 'normal'
            axes[idx + 1].set_title(
                f'候选 {idx + 1}\n分数: {score:.2f}',
                fontsize=11,
                color=color,
                fontweight=weight,
            )
            axes[idx + 1].axis('off')

            # 标记最佳
            if idx == best_idx:
                axes[idx + 1].set_edgecolor('green')
                axes[idx + 1].set_linewidth(3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_emotion_style_guide(
        self,
        emotion: EmotionCategory,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        绘制情感风格指南
        """
        attrs = EMOTION_VISUAL_ATTRIBUTES[emotion]
        cn_name = EMOTION_NAMES_CN[emotion]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 亮度指示
        brightness_map = {"low": 0.2, "low-medium": 0.35, "medium": 0.5, "medium-high": 0.65, "high": 0.8}
        brightness_val = brightness_map.get(attrs['brightness'], 0.5)
        brightness_img = np.ones((100, 200, 3)) * brightness_val
        axes[0].imshow(brightness_img)
        axes[0].set_title(f"亮度: {attrs['brightness']}")
        axes[0].axis('off')

        # 饱和度指示
        saturation_map = {"low": 0.1, "low-medium": 0.3, "medium": 0.5, "medium-high": 0.7, "high": 0.9}
        saturation_val = saturation_map.get(attrs['saturation'], 0.5)
        base_color = np.array([0.5, 0.3, 0.8])  # 紫色基调
        saturation_img = np.ones((100, 200, 3))
        saturation_img[:, :, :] = base_color * saturation_val + (1 - saturation_val) * 0.5
        axes[1].imshow(saturation_img)
        axes[1].set_title(f"饱和度: {attrs['saturation']}")
        axes[1].axis('off')

        # 关键词云（简化版）
        axes[2].text(
            0.5, 0.5,
            '\n'.join(attrs['keywords']),
            ha='center', va='center',
            fontsize=12,
            transform=axes[2].transAxes,
        )
        axes[2].set_title("关键词")
        axes[2].axis('off')

        fig.suptitle(f"情感风格指南: {cn_name} ({emotion.value})", fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig
