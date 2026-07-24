import streamlit as st
import torch
from PIL import Image
import os
import time

from configs.config import EmotionCategory, EMOTION_NAMES_CN, app_config

st.set_page_config(page_title="EmoEdit", page_icon="🎨", layout="wide")
st.title("🎨 EmoEdit - 情感驱动图像编辑系统")

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

st.sidebar.header("参数")
num_candidates = st.sidebar.slider("候选图数量", 1, 5, 3)
guidance_scale = st.sidebar.slider("引导强度", 1.0, 20.0, 7.5, 0.5)
strength = st.sidebar.slider("修改强度", 0.1, 1.0, 0.65, 0.05,
    help="0=完全不变, 1=完全重绘, 0.65=保持结构改变风格")

emotion_options = {EMOTION_NAMES_CN[e]: e.value for e in EmotionCategory}

col1, col2 = st.columns([1, 2])

with col1:
    st.header("输入")
    emoset_dir = "/mnt/workspace/data/emoset"
    if os.path.exists(emoset_dir):
        emoset_files = sorted([f for f in os.listdir(emoset_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        if emoset_files:
            selected = st.selectbox("EmoSet图片", emoset_files)
            if st.button("加载"):
                st.session_state['img'] = Image.open(os.path.join(emoset_dir, selected)).convert("RGB")
        st.divider()

    up = st.file_uploader("上传图片", type=['jpg', 'jpeg', 'png'])
    if up:
        st.session_state['img'] = Image.open(up).convert("RGB")

    if 'img' in st.session_state:
        st.image(st.session_state['img'], caption="原图")

    target_name = st.selectbox("目标情绪", list(emotion_options.keys()))
    target_emotion = EmotionCategory(emotion_options[target_name])

    if st.button("开始转换", type="primary", use_container_width=True):
        if 'img' not in st.session_state:
            st.error("请先选择图片！")
        else:
            with st.spinner("加载模型..."):
                clip = load_clip()
                blip2 = load_blip2()
                sd = load_sd()
                from models.emotion_prompt_builder import EmotionPromptBuilder
                pb = EmotionPromptBuilder()

            with st.spinner("分析中..."):
                image = st.session_state['img']
                emotion, confidence = clip.analyze_emotion(image)
                desc = blip2.generate_caption(image)
                st.info("情感: " + EMOTION_NAMES_CN[emotion] + " (" + str(round(confidence*100, 1)) + "%)")
                st.info("描述: " + desc)

                prompts = pb.build_prompt(desc, target_emotion)
                with st.expander("提示词"):
                    st.code(prompts['positive'])

            with st.spinner("基于原图生成中..."):
                start = time.time()
                candidates = sd.generate_candidates(
                    prompt=prompts['positive'],
                    negative_prompt=prompts['negative'],
                    init_image=image,
                    strength=strength,
                    num_candidates=num_candidates,
                    guidance_scale=guidance_scale,
                )
                gen_time = time.time() - start

                scores = clip.score_candidates(candidates, target_emotion)
                best_idx = scores[0][0]

                st.session_state['cands'] = candidates
                st.session_state['scores'] = scores
                st.session_state['best'] = best_idx
                st.session_state['orig'] = image
                st.success("完成！耗时 " + str(round(gen_time, 1)) + "秒")

with col2:
    st.header("结果")
    if 'cands' in st.session_state:
        cands = st.session_state['cands']
        scores = st.session_state['scores']
        best = st.session_state['best']

        # 显示原图
        if 'orig' in st.session_state:
            st.subheader("原图 vs 最佳结果")
            c_orig, c_best = st.columns(2)
            with c_orig:
                st.image(st.session_state['orig'], caption="原图")
            with c_best:
                real_score = scores[best][1] if isinstance(scores[best], tuple) else scores[best]
                st.image(cands[best], caption="最佳 (分数: " + str(round(real_score, 4)) + ")")

        st.divider()
        st.subheader("所有候选图")
        cols = st.columns(min(len(cands), 3))
        for idx in range(len(cands)):
            col = cols[idx % len(cols)]
            with col:
                s = scores[idx]
                real_score = s[1] if isinstance(s, tuple) else s
                if idx == best:
                    st.markdown("**⭐ 最佳**")
                else:
                    st.markdown("**候选 " + str(idx+1) + "**")
                st.image(cands[idx], caption="分数: " + str(round(real_score, 4)))

        os.makedirs(app_config.results_dir, exist_ok=True)
        sp = os.path.join(app_config.results_dir, "result_" + str(int(time.time())) + ".png")
        cands[best].save(sp)
        st.success("已保存: " + sp)
    else:
        st.info("等待生成...")
