from setuptools import setup, find_packages

setup(
    name="emoedit",
    version="1.0.0",
    description="情感驱动图像编辑系统 - 基于大模型串联的多模态图像情绪编辑工作流",
    author="EmoEdit Team",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "diffusers>=0.20.0",
        "accelerate>=0.20.0",
        "Pillow>=9.0.0",
        "opencv-python>=4.7.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "gradio>=3.40.0",
        "tqdm>=4.65.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
    ],
)
