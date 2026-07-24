"""
EmoEdit - Streamlit 版本（省显存）
"""
import streamlit as st
import torch
from PIL import Image
import os
import time

from configs.config import EmotionCategory, EMOTION_NAMES_CN, app_config

st.set_page_config(page_title="EmoEdit", page_icon="🎨", layout="wide")
st.title("🎨 EmoEdit - 情感驱动图像编辑系统")

# 懒加载
@st.cache_resource
def load_clip():
    from models.clip_analyzer import CLIPAnalyzer
    return CLIPAnalyzer()

@st.cache_resource
def load_blip2():
    from models.blip2_parser import BLIP2Parser
    return BLIP2Parser()

@st.cache_resource
def load_sd():
    from models.sd_generator import StableDiffusionGenerator
    return StableDiffusionGenerator()

# 侧边栏
st.sidebar.header("⚙️ 参数")
num_candidates = st.sidebar.slider("候选图数量", 1, 5, 3)
guidance_scale = st.sidebar.slider("引导强度", 1.0, 20.0, 7.5, 0.5)

emotion_options = {EMOTION_NAMES_CN[e]: e.value for e in EmotionCategory}

col1, col2 = st.columns([1, 2])

with col1:
    st.header("📥 输入")

    # EmoSet
    emoset_dir = "/mnt/workspace/data/emoset"
    if os.path.exists(emoset_dir):
        st.subheader("📚 EmoSet 数据集")
        emoset_files = sorted([f for f in os.listdir(emoset_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        if emoset_files:
            selected_file = st.selectbox("选择图片", emoset_files)
            if st.button("加载"):
                st.session_state['input_image'] = Image.open(os.path.join(emoset_dir, selected_file)).convert("RGB")
        st.divider()

    # 上传
    uploaded_file = st.file_uploader("📤 上传图片", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        st.session_state['input_image'] = Image.open(uploaded_file).convert("RGB")

    if 'input_image' in st.session_state:
        st.image(st.session_state['input_image'], caption="输入图片", width="stretch")

    target_name = st.selectbox("🎯 目标情绪", list(emotion_options.keys()))
    target_emotion = EmotionCategory(emotion_options[target_name])

    if st.button("🚀 开始转换", type="primary", width="stretch"):
        if 'input_image' not in st.session_state:
            st.error("请先选择或上传图片！")
        else:
            # 懒加载模型
            with st.spinner("加载 CLIP 模型..."):
                clip = load_clip()
            with st.spinner("加载 BLIP-2 模型..."):
                blip2 = load_blip2()
            with st.spinner("加载 Stable Diffusion 模型..."):
                sd = load_sd()
            from models.emotion_prompt_builder import EmotionPromptBuilder
            prompt_builder = EmotionPromptBuilder()

            with st.spinner("分析和生成中..."):
                image = st.session_state['input_image']

                st.write("📊 分析原图...")
                emotion, confidence = clip.analyze_emotion(image)
                desc = blip2.generate_caption(image)
                st.info(f"检测情感: **{EMOTION_NAMES_CN[emotion]}** ({confidence:.2%})")
                st.info(f"描述: *{desc}*")

                prompts = prompt_builder.build_prompt(desc, target_emotion)
                with st.expander("查看提示词"):
                    st.code(prompts['positive'])
                    st.code(prompts['negative'])

                st.write("🎨 生成候选图...")
                start = time.time()
                candidates = sd.generate_candidates(
                    prompt=prompts['positive'],
                    negative_prompt=prompts['negative'],
                    num_candidates=num_candidates,
                    guidance_scale=guidance_scale,
                )
                gen_time = time.time() - start

                st.write("🎯 CLIP 评分...")
                scores = clip.score_candidates(candidates, target_emotion)
                best_idx = scores[0][0]

                st.session_state['candidates'] = candidates
                st.session_state['scores'] = scores
                st.session_state['best_idx'] = best_idx
                st.success(f"完成！耗时 {gen_time:.1f}秒")

with col2:
    st.header("📤 结果")
    if 'candidates' in st.session_state:
        candidates = st.session_state['candidates']
        scores = st.session_state['scores']
        best_idx = st.session_state['best_idx']
        cols = st.columns(len(candidates))
        for idx, (col, (img, score)) in enumerate(zip(cols, zip(candidates, scores))):
            with col:
                label = f"⭐ 最佳" if idx == best_idx else f"候选 {idx+1}"
                st.markdown(f"**{label}**")
                st.image(img, caption=f"分数: {score:.4f}", width="stretch")

        os.makedirs(app_config.results_dir, exist_ok=True)
        save_path = os.path.join(app_config.results_dir, f"result_{int(time.time())}.png")
        candidates[best_idx].save(save_path)
        st.success(f"已保存: {save_path}")
    else:
        st.info("等待生成...")
