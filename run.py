"""
快速启动脚本
用于测试和调试
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_environment():
    """检查运行环境"""
    print("=" * 60)
    print("环境检查")
    print("=" * 60)

    # 检查Python版本
    python_version = sys.version.split()[0]
    print(f"Python 版本: {python_version}")

    # 检查PyTorch
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
            print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    except ImportError:
        print("错误: 未安装 PyTorch")
        return False

    # 检查关键依赖
    dependencies = [
        ("transformers", "transformers"),
        ("diffusers", "diffusers"),
        ("gradio", "gradio"),
        ("PIL", "Pillow"),
        ("cv2", "opencv-python"),
    ]

    print("\n依赖检查:")
    all_ok = True
    for module, package in dependencies:
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (未安装)")
            all_ok = False

    print("=" * 60)
    return all_ok


def main():
    """主函数"""
    print("\nEmoEdit - 情感驱动图像编辑系统")
    print("启动中...\n")

    # 环境检查
    if not check_environment():
        print("\n请先安装缺失的依赖:")
        print("pip install -r requirements.txt")
        return

    # 启动应用
    from app import main as app_main
    app_main()


if __name__ == "__main__":
    main()
