"""
阿里云 DSW 启动脚本
在 DSW Notebook 中运行此脚本启动应用
"""
import sys
import os
import subprocess

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.dsw_config import (
    ensure_directories,
    setup_environment,
    print_gpu_info,
    WORK_DIR,
)


def download_frpc():
    """下载 frpc 文件用于公网链接"""
    frpc_dir = "/mnt/workspace/models/huggingface/gradio/frpc"
    frpc_path = os.path.join(frpc_dir, "frpc_linux_amd64_v0.3")

    if os.path.exists(frpc_path):
        return True

    print("[DSW] 下载 frpc 文件...")
    os.makedirs(frpc_dir, exist_ok=True)

    try:
        subprocess.run([
            "wget", "-q",
            "https://cdn-media.huggingface.co/frpc-gradio-0.3/frpc_linux_amd64",
            "-O", frpc_path
        ], check=True)
        os.chmod(frpc_path, 0o755)
        print("[DSW] frpc 下载完成")
        return True
    except Exception as e:
        print(f"[DSW] frpc 下载失败: {e}")
        return False


def main():
    """DSW 主启动函数"""
    print("=" * 60)
    print("EmoEdit - 阿里云 DSW 启动器")
    print("=" * 60)

    # 1. 确保持久化目录存在
    print("\n[1/5] 创建持久化目录...")
    ensure_directories()

    # 2. 设置环境变量
    print("\n[2/5] 配置环境变量...")
    setup_environment()

    # 3. 检查 GPU
    print("\n[3/5] 检查 GPU...")
    print_gpu_info()

    # 4. 下载 frpc（用于公网链接）
    print("\n[4/5] 准备公网链接...")
    frpc_ready = download_frpc()

    # 5. 启动应用
    print("\n[5/5] 启动 EmoEdit...")
    print("-" * 60)

    from app import EmoEditApp, create_ui
    import gradio as gr

    # 初始化应用
    app = EmoEditApp()

    # 创建界面
    demo = create_ui(app)

    # 启动
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            prevent_thread_lock=False,
        )
    except Exception as e:
        print(f"\n[DSW] 公网链接创建失败: {e}")
        print("[DSW] 尝试仅本地模式...")
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            prevent_thread_lock=False,
        )


if __name__ == "__main__":
    main()
