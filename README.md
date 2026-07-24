# EmoEdit - 情感驱动图像编辑系统

基于大模型串联（Model Pipeline）的多模态图像情绪编辑工作流。

## 📖 项目简介

EmoEdit 是一个智能图像情绪编辑系统，能够在不改变原图主体结构的前提下，通过自然语言和多模态特征，彻底改写一张图片的"情绪氛围"。

### 示例效果

```
输入: 阴沉悲伤的雨天街道
     ↓ AI自动解析
输出: 阳光明媚、色彩斑斓的晴朗街道
```

整个过程不需要人工编写复杂的提示词，全部由 AI 自动解析并生成。

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     EmoEdit 系统架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────┐    ┌──────────┐    ┌──────────────────┐     │
│   │  CLIP    │    │  BLIP-2  │    │  SD + ControlNet │     │
│   │ 诊断/裁判 │    │ 语义解析  │    │    跨情感画师     │     │
│   └────┬─────┘    └────┬─────┘    └────────┬─────────┘     │
│        │               │                   │                │
│        ▼               ▼                   ▼                │
│   ┌─────────────────────────────────────────────────────┐  │
│   │                    数据流转过程                       │  │
│   │  输入 → 解构 → 重组 → 生成与质检                     │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件

| 组件 | 模型 | 角色定位 | 核心任务 |
|------|------|----------|----------|
| 语义解析器 | BLIP-2 (2.7B) | 剥离情感色彩 | 提取纯客观画面描述 |
| 诊断与裁判 | CLIP (ViT-L/14) | 前期诊断/后期评分 | 情感验证与生成质量评估 |
| 跨情感画师 | SD v1.5 + ControlNet | 结构保持重绘 | 基于深度图的风格转换 |

## 🚀 快速开始

### 方式一：阿里云 DSW（推荐）

#### 环境要求
- 阿里云 DSW 实例（推荐 A10 GPU，24GB 显存）

#### 初始化

```bash
# 1. 克隆项目到工作目录
cd /home/work
git clone <your-repo-url> emotion-image-editor
cd emotion-image-editor

# 2. 运行初始化脚本
bash dsw_setup.sh

# 3. 启动应用
python dsw_launcher.py
```

启动后，Notebook 会输出一个 `https://xxxx.gradio.live` 的公网链接，点击即可访问。

#### DSW 存储结构

```
/mnt/user/                    ← 持久化存储（重启不丢失）
├── data/                     ← 数据集
│   ├── emoset/               ← EmoSet 数据集
│   └── sample_images/        ← 测试图片
├── models/                   ← 预训练模型缓存
│   └── huggingface/          ← HF 模型缓存
└── results/                  ← 生成结果
    └── emoedit/

/home/work/emotion-image-editor/  ← 项目代码（系统盘）
```

### 方式二：本地部署

#### 环境要求
- Python 3.8+
- CUDA 11.7+ (推荐)
- GPU显存: 16GB+ (推荐24GB A10)

#### 安装依赖

```bash
# 创建虚拟环境
conda create -n emoedit python=3.9
conda activate emoedit

# 安装PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
pip install -r requirements.txt
```

#### 启动应用

```bash
cd emotion-image-editor
python app.py
```

启动后，浏览器会自动打开 `http://localhost:7860`

## 📁 项目结构

```
emotion-image-editor/
├── app.py                      # 主程序入口
├── requirements.txt            # 依赖列表
├── README.md                   # 项目文档
│
├── configs/                    # 配置模块
│   ├── __init__.py
│   └── config.py               # 模型配置、情感类别定义
│
├── models/                     # 模型模块
│   ├── __init__.py
│   ├── clip_analyzer.py        # CLIP 分析器
│   ├── blip2_parser.py         # BLIP-2 语义解析器
│   ├── sd_generator.py         # Stable Diffusion 生成器
│   └── emotion_prompt_builder.py  # 情感提示词构建器
│
├── utils/                      # 工具模块
│   ├── __init__.py
│   ├── image_utils.py          # 图像处理工具
│   └── visualization.py        # 可视化工具
│
└── assets/                     # 资源目录
    ├── samples/                # 示例图片
    └── results/                # 生成结果
```

## 🔄 数据流转过程

### 1. 输入阶段
- 用户上传一张图片
- 指定希望转换成的目标情绪（如：从 悲伤 Sadness → 快乐 Joy）

### 2. 解构阶段
- **CLIP** 对原图进行零样本推理，验证当前画面的主导情绪
- **BLIP-2** 对原图进行视觉问答，生成不带感情色彩的客观描述

```
示例输出: "A lonely street with a person holding an umbrella."
```

### 3. 重组阶段
- 系统自动将 BLIP-2 的客观描述与目标情绪拼接
- 结合 EmoSet 视觉属性先验生成精准提示词

```
生成的提示词:
"A lonely street with a person holding an umbrella, expressing joy,
bright sunlight, vibrant colors, high contrast"
```

### 4. 生成与质检阶段
1. 提取原图的 ControlNet 深度图（Depth Map）
2. 将深度图连同提示词送入 Stable Diffusion
3. 生成 3 张不同光影的候选图
4. CLIP 计算候选图与目标情感的匹配度
5. 自动挑选最佳结果呈现给用户

## 🎭 支持的情感

本系统严格遵循 EmoSet 设定的 8 种基础情绪：

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

## ⚙️ 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 候选图数量 | 3 | 生成的候选图数量 |
| 引导强度 (CFG) | 7.5 | 提示词引导强度，越高越贴近提示词 |
| ControlNet强度 | 0.85 | 深度图约束强度，越高越保持原结构 |

## 🛠️ 技术栈

- **深度学习框架**: PyTorch (bfloat16 精度)
- **模型生态**: Hugging Face transformers, diffusers
- **视觉控制**: ControlNet (空间结构约束)
- **前端交互**: Gradio

## 📝 注意事项

1. **显存要求**: 建议使用 24GB 显存的 GPU（如 A10）
2. **首次运行**: 模型会自动下载，需要稳定网络
3. **生成时间**: 单张图片约需 30-60 秒
4. **图片尺寸**: 建议上传 512x512 以上的图片

## 📄 许可证

MIT License

## 🙏 致谢

- [EmoSet](http://emotion Dataset) - 情感视觉属性先验
- [BLIP-2](https://github.com/salesforce/LAVIS) - 语义解析
- [CLIP](https://github.com/openai/CLIP) - 情感诊断与评分
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) - 图像生成
- [ControlNet](https://github.com/lllyasviel/ControlNet) - 结构控制
