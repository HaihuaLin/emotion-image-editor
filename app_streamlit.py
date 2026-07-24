"""
EmoEdit - Streamlit 版本
更简单稳定的 Web 界面
"""
import streamlit as st
import torch
from PIL import Image
import os
import time

from configs.config import EmotionCategory, EMOTION_NAMES_CN, model_config, app_config
from models import CLIPAnalyzer, BLIP2Parser, StableDiffusionGenerator, EmotionPromptBuilder
from utils import ImageProcessor

# 页面配置
st.set_page_config(
    page_title="EmoEdit - 情感图像编辑",
    page_icon="🎨",
    layout="wide",
)

st.title("🎨 EmoEdit - 情感驱动图像编辑系统")
st.markdown("基于大模型串联的多模态图像情绪编辑工作流")

# 初始化模型（使用缓存避免重复加载）
@st.cache_resource
def load_models():
    """加载所有模型"""
    st.info("正在加载模型，请稍候...")
    clip = CLIPAnalyzer()
    blip2 = BLIP2Parser()
    sd = StableDiffusionGenerator()
    prompt_builder = EmotionPromptBuilder()
    image_processor = ImageProcessor()
    st.success("模型加载完成！")
    return clip, blip2, sd, prompt_builder, image_processor

# 加载模型
clip, blip2, sd, prompt_builder, image_processor = load_models()

# 侧边栏参数
st.sidebar.header("⚙️ 生成参数")
num_candidates = st.sidebar.slider("候选图数量", 1, 5, 3)
guidance_scale = st.sidebar.slider("引导强度 (CFG)", 1.0, 20.0, 7.5, 0.5)

# 情感选项
emotion_options = {EMOTION_NAMES_CN[e]: e.value for e in EmotionCategory}

# 主界面布局
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📥 输入")

    # EmoSet 数据集选择
    emoset_dir = "/mnt/workspace/data/emoset"
    if os.path.exists(emoset_dir):
        st.subheader("📚 从 EmoSet 选择")
        emoset_files = [f for f in os.listdir(emoset_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if emoset_files:
            selected_file = st.selectbox("选择图片", emoset_files)
            if st.button("加载图片"):
                image_path = os.path.join(emoset_dir, selected_file)
                st.session_state['input_image'] = Image.open(image_path).convert("RGB")
        st.divider()

    st.subheader("📤 或上传图片")
    uploaded_file = st.file_uploader("上传图片", type=['jpg', 'jpeg', 'png', 'bmp'])
    if uploaded_file:
        st.session_state['input_image'] = Image.open(uploaded_file).convert("RGB")

    # 显示输入图片
    if 'input_image' in st.session_state:
        st.image(st.session_state['input_image'], caption="输入图片", use_container_width=True)

    # 目标情感
    target_emotion_name = st.selectbox("🎯 目标情绪", list(emotion_options.keys()))
    target_emotion = EmotionCategory(emotion_options[target_emotion_name])

    # 生成按钮
    if st.button("🚀 开始转换", type="primary", use_container_width=True):
        if 'input_image' not in st.session_state:
            st.error("请先上传或选择一张图片！")
        else:
            with st.spinner("正在分析和生成中..."):
                image = st.session_state['input_image']

                # Step 1: 分析
                st.write("📊 分析原图情感...")
                emotion, confidence = clip.analyze_emotion(image)
                objective_desc = blip2.generate_caption(image)

                st.info(f"检测到的情感: **{EMOTION_NAMES_CN[emotion]}** (置信度: {confidence:.2%})")
                st.info(f"BLIP-2 描述: *{objective_desc}*")

                # Step 2: 构建提示词
                st.write("📝 构建提示词...")
                prompts = prompt_builder.build_prompt(objective_desc, target_emotion)

                with st.expander("查看生成参数"):
                    st.write("**正向提示词:**")
                    st.code(prompts['positive'])
                    st.write("**负向提示词:**")
                    st.code(prompts['negative'])

                # Step 3: 生成
                st.write("🎨 生成候选图...")
                start_time = time.time()
                candidates = sd.generate_candidates(
                    prompt=prompts['positive'],
                    negative_prompt=prompts['negative'],
                    num_candidates=num_candidates,
                    guidance_scale=guidance_scale,
                )
                gen_time = time.time() - start_time

                # Step 4: CLIP 评分
                st.write("🎯 CLIP 评分中...")
                scores = clip.score_candidates(candidates, target_emotion)
                best_idx = scores[0][0]

                st.session_state['candidates'] = candidates
                st.session_state['scores'] = scores
                st.session_state['best_idx'] = best_idx
                st.session_state['gen_time'] = gen_time

                st.success(f"生成完成！耗时 {gen_time:.1f} 秒")

with col2:
    st.header("📤 生成结果")

    if 'candidates' in st.session_state:
        candidates = st.session_state['candidates']
        scores = st.session_state['scores']
        best_idx = st.session_state['best_idx']

        # 显示候选图
        cols = st.columns(len(candidates))
        for idx, (col, (candidate, score)) in enumerate(zip(cols, zip(candidates, scores))):
            with col:
                if idx == best_idx:
                    st.markdown(f"**⭐ 候选 {idx+1} (最佳)**")
                else:
                    st.markdown(f"候选 {idx+1}")
                st.image(candidate, caption=f"分数: {score:.4f}", use_container_width=True)

        # 保存最佳结果
        best_image = candidates[best_idx]
        os.makedirs(app_config.results_dir, exist_ok=True)
        save_path = os.path.join(app_config.results_dir, f"result_{int(time.time())}.png")
        best_image.save(save_path)
        st.success(f"最佳结果已保存至: `{save_path}`")
    else:
        st.info("等待生成结果...")

# 使用说明
st.divider()
st.markdown("""
### 📚 使用说明
1. 从 EmoSet 选择图片或上传自定义图片
2. 选择目标情绪
3. 调整生成参数（可选）
4. 点击"开始转换"
5. 查看生成结果，最佳结果会自动保存
""")
