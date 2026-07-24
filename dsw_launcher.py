"""
阿里云 DSW 启动脚本
在 DSW Notebook 中运行此脚本启动应用
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.dsw_config import (
    ensure_directories,
    setup_environment,
    print_gpu_info,
    WORK_DIR,
)


def main():
    """DSW 主启动函数"""
    print("=" * 60)
    print("EmoEdit - 阿里云 DSW 启动器")
    print("=" * 60)

    # 1. 确保持久化目录存在
    print("\n[1/4] 创建持久化目录...")
    ensure_directories()

    # 2. 设置环境变量
    print("\n[2/4] 配置环境变量...")
    setup_environment()

    # 3. 检查 GPU
    print("\n[3/4] 检查 GPU...")
    print_gpu_info()

    # 4. 启动应用
    print("\n[4/4] 启动 EmoEdit...")
    print("-" * 60)

    from app import EmoEditApp, create_ui
    import gradio as gr

    # 初始化应用
    app = EmoEditApp()

    # 创建界面
    demo = create_ui(app)

    # 启动（DSW 需要 share=True 来访问）
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # DSW 必须设置为 True
        prevent_thread_lock=False,
    )


if __name__ == "__main__":
    main()
