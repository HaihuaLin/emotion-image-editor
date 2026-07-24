"""
EmoEdit - 情感驱动图像编辑系统
主程序入口
"""
import gradio as gr
import torch
from PIL import Image
from typing import List, Tuple
import os
import time

from configs.config import EmotionCategory, EMOTION_NAMES_CN, model_config, app_config
from models import CLIPAnalyzer, BLIP2Parser, StableDiffusionGenerator, EmotionPromptBuilder
from utils import ImageProcessor, EmotionVisualizer
from utils.device_utils import print_device_info, clear_gpu_cache


class EmoEditApp:
    """
    EmoEdit 应用主类
    协调所有模型完成情感驱动的图像编辑
    """

    def __init__(self):
        print("=" * 60)
        print("EmoEdit - 情感驱动图像编辑系统")
        print("=" * 60)

        # 显示设备信息
        print_device_info()

        print("\n正在初始化模型...")

        # 初始化模型
        self.clip = CLIPAnalyzer()
        self.blip2 = BLIP2Parser()
        self.sd_generator = StableDiffusionGenerator()
        self.prompt_builder = EmotionPromptBuilder()

        # 工具
        self.image_processor = ImageProcessor()
        self.visualizer = EmotionVisualizer()

        print("\n" + "=" * 60)
        print("所有模型加载完成！")
        print("=" * 60)

    def analyze_image(self, image: Image.Image) -> Tuple[dict, str]:
        """
        分析输入图像
        返回: (情感分布, 客观描述)
        """
        if image is None:
            return None, "请先上传图片"

        # 预处理
        image = self.image_processor.resize_image(image)

        # CLIP 情感分析
        emotion, confidence = self.clip.analyze_emotion(image)

        # 获取所有情感的分数
        emotion_texts = [
            f"a photo expressing {e.value}" for e in EmotionCategory
        ]
        inputs = self.clip.processor(
            text=emotion_texts,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.clip.device)

        with torch.no_grad():
            outputs = self.clip.model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = logits.softmax(dim=-1)

        emotion_scores = {
            e: probs[i].item()
            for i, e in enumerate(EmotionCategory)
        }

        # BLIP-2 客观描述
        objective_desc = self.blip2.generate_caption(image)

        # 生成描述文本
        result_text = f"""## 原图分析结果

### 检测到的主导情感
- **情感**: {EMOTION_NAMES_CN[emotion]} ({emotion.value})
- **置信度**: {confidence:.2%}

### 所有情感分数
"""
        for e, score in sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 20)
            result_text += f"- {EMOTION_NAMES_CN[e]}: {score:.2%} {bar}\n"

        result_text += f"""
### BLIP-2 客观描述
> {objective_desc}
"""
        return emotion_scores, objective_desc, result_text

    def generate_images(
        self,
        image: Image.Image,
        objective_desc: str,
        target_emotion: EmotionCategory,
        num_candidates: int = 3,
        guidance_scale: float = 7.5,
        controlnet_scale: float = 0.85,
    ) -> Tuple[List[Image.Image], str]:
        """
        生成情感转换后的图像
        """
        if image is None:
            return [], "请先上传图片"

        # 预处理
        image = self.image_processor.resize_image(image)

        # 提取深度图
        print("[Pipeline] 提取深度图...")
        depth_map = self.sd_generator.extract_depth_map(image)

        # 构建提示词
        print("[Pipeline] 构建提示词...")
        prompts = self.prompt_builder.build_prompt(
            objective_desc,
            target_emotion,
        )

        prompt_text = f"""## 生成参数

### 目标情感
{EMOTION_NAMES_CN[target_emotion]} ({target_emotion.value})

### 正向提示词
```
{prompts['positive']}
```

### 负向提示词
```
{prompts['negative']}
```

### 生成参数
- 候选图数量: {num_candidates}
- 引导强度: {guidance_scale}
- ControlNet强度: {controlnet_scale}

---
正在生成中，请稍候...
"""

        # 生成候选图
        print("[Pipeline] 生成候选图...")
        start_time = time.time()
        candidates = self.sd_generator.generate_candidates(
            prompt=prompts['positive'],
            negative_prompt=prompts['negative'],
            depth_map=depth_map,
            num_candidates=num_candidates,
            guidance_scale=guidance_scale,
            conditioning_scale=controlnet_scale,
        )
        gen_time = time.time() - start_time

        # CLIP 评分
        print("[Pipeline] CLIP评分中...")
        scores = self.clip.score_candidates(candidates, target_emotion)
        best_idx = scores[0][0]

        # 添加评分到输出
        prompt_text += f"\n### 生成完成 (耗时 {gen_time:.1f}秒)\n\n"
        prompt_text += "### 候选图评分\n"
        for idx, score in scores:
            marker = " ⭐ 最佳" if idx == best_idx else ""
            prompt_text += f"- 候选 {idx + 1}: {score:.4f}{marker}\n"

        # 重新排列候选图（最佳在前）
        ordered_candidates = [candidates[idx] for idx, _ in scores]

        return ordered_candidates, prompt_text, best_idx

    def process_image(
        self,
        image: Image.Image,
        target_emotion: str,
        num_candidates: int,
        guidance_scale: float,
        controlnet_scale: float,
    ) -> Tuple:
        """
        完整处理流程
        返回: (分析结果, 客观描述, 深度图, 候选图1, 候选图2, 候选图3, 生成日志)
        """
        if image is None:
            return (
                "请先上传图片",
                "",
                None,
                None, None, None,
                "",
            )

        # 转换情感类别
        emotion_map = {e.value: e for e in EmotionCategory}
        if target_emotion not in emotion_map:
            return (
                f"无效的情感类别: {target_emotion}",
                "",
                None,
                None, None, None,
                "",
            )

        target = emotion_map[target_emotion]

        # Step 1: 分析
        print("\n[Step 1] 分析原图...")
        emotion_scores, objective_desc, analysis_text = self.analyze_image(image)

        # Step 2: 深度图
        print("\n[Step 2] 提取深度图...")
        processed_image = self.image_processor.resize_image(image.copy())
        depth_map = self.sd_generator.extract_depth_map(processed_image)

        # Step 3: 生成
        print("\n[Step 3] 生成候选图...")
        candidates, gen_log, best_idx = self.generate_images(
            image=image,
            objective_desc=objective_desc,
            target_emotion=target,
            num_candidates=num_candidates,
            guidance_scale=guidance_scale,
            controlnet_scale=controlnet_scale,
        )

        # 保存结果
        timestamp = int(time.time())
        os.makedirs(app_config.results_dir, exist_ok=True)
        for idx, candidate in enumerate(candidates):
            save_path = os.path.join(
                app_config.results_dir,
                f"result_{timestamp}_{idx}.png"
            )
            candidate.save(save_path)

        # 生成对比图说明
        comparison_text = f"""
## 最终结果

最佳生成结果为 **候选 {best_idx + 1}** (CLIP分数最高)

已将所有结果保存至: `{app_config.results_dir}/`
"""

        return (
            analysis_text,
            objective_desc,
            depth_map,
            candidates[0] if len(candidates) > 0 else None,
            candidates[1] if len(candidates) > 1 else None,
            candidates[2] if len(candidates) > 2 else None,
            gen_log + comparison_text,
        )


def create_ui(app: EmoEditApp):
    """
    创建 Gradio 界面
    """
    # 情感选项
    emotion_choices = [(EMOTION_NAMES_CN[e], e.value) for e in EmotionCategory]

    with gr.Blocks(
        title=app_config.project_name,
        theme=gr.themes.Soft(),
        css="""
        .main-title {
            text-align: center;
            margin-bottom: 20px;
        }
        .result-box {
            border: 2px solid #4CAF50;
            border-radius: 8px;
            padding: 10px;
        }
        """
    ) as demo:
        gr.Markdown(
            f"""
            # {app_config.project_name}
            ### 基于大模型串联的多模态图像情绪编辑工作流

            **工作流程**: 上传图片 → 选择目标情绪 → AI自动分析并重绘

            ---

            **核心模型**:
            - 🧠 **BLIP-2**: 语义解析器 - 提取客观描述
            - 🎯 **CLIP**: 诊断与裁判 - 情感评分
            - 🎨 **Stable Diffusion + ControlNet**: 跨情感画师
            """
        )

        with gr.Row():
            # 左侧：输入区
            with gr.Column(scale=1):
                gr.Markdown("### 📥 输入")

                input_image = gr.Image(
                    label="上传图片",
                    type="pil",
                    height=300,
                )

                target_emotion = gr.Dropdown(
                    choices=emotion_choices,
                    label="目标情绪",
                    value="joy",
                )

                gr.Markdown("#### 生成参数")
                num_candidates = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="候选图数量",
                )
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=20.0,
                    value=7.5,
                    step=0.5,
                    label="引导强度 (CFG Scale)",
                )
                controlnet_scale = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.85,
                    step=0.05,
                    label="ControlNet 强度",
                )

                run_btn = gr.Button(
                    "🚀 开始转换",
                    variant="primary",
                    size="lg",
                )

            # 右侧：输出区
            with gr.Column(scale=2):
                gr.Markdown("### 📤 输出")

                with gr.Tabs():
                    with gr.TabItem("📊 分析结果"):
                        analysis_output = gr.Markdown(label="分析结果")

                    with gr.TabItem("🔍 深度图"):
                        depth_output = gr.Image(
                            label="深度图 (ControlNet)",
                            height=250,
                        )

                    with gr.TabItem("🎨 生成结果"):
                        with gr.Row():
                            candidate1 = gr.Image(label="候选 1", height=250)
                            candidate2 = gr.Image(label="候选 2", height=250)
                            candidate3 = gr.Image(label="候选 3", height=250)

                    with gr.TabItem("📝 详细日志"):
                        log_output = gr.Markdown(label="处理日志")

        # 示例
        gr.Markdown(
            """
            ---
            ### 📚 使用说明

            1. **上传图片**: 选择一张需要转换情感的图片
            2. **选择目标情绪**: 从8种基础情绪中选择想要转换到的情绪
            3. **调整参数**: 根据需要调整生成参数
            4. **点击转换**: 等待AI完成分析和生成
            5. **查看结果**: 在各标签页查看分析结果、深度图和生成的候选图

            ### 🎭 支持的情感

            | 情感 | 英文 | 视觉特征 |
            |------|------|----------|
            | 敬畏 | Awe | 宏大、壮丽、高对比度 |
            | 愤怒 | Anger | 强烈、激烈、高饱和度 |
            | 满足 | Contentment | 平和、宁静、中等饱和度 |
            | 厌恶 | Disgust | 低饱和度、低亮度 |
            | 兴奋 | Excitement | 活力、鲜艳、高亮度 |
            | 恐惧 | Fear | 阴暗、低亮度、高对比度 |
            | 快乐 | Joy | 明亮、温暖、高饱和度 |
            | 悲伤 | Sadness | 忧郁、低亮度、低饱和度 |
            """
        )

        # 绑定事件
        run_btn.click(
            fn=app.process_image,
            inputs=[
                input_image,
                target_emotion,
                num_candidates,
                guidance_scale,
                controlnet_scale,
            ],
            outputs=[
                analysis_output,
                gr.Textbox(visible=False),  # objective_desc (hidden)
                depth_output,
                candidate1,
                candidate2,
                candidate3,
                log_output,
            ],
        )

    return demo


def main():
    """主函数"""
    # 检查GPU
    if not torch.cuda.is_available():
        print("警告: 未检测到GPU，系统可能无法正常运行")
        print("请确保已安装CUDA版PyTorch")

    # 创建应用
    app = EmoEditApp()

    # 创建并启动界面
    demo = create_ui(app)
    demo.launch(
        server_port=app_config.server_port,
        share=app_config.server_share,
        server_name="0.0.0.0",
    )


if __name__ == "__main__":
    main()
