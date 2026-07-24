"""
from PIL import ImageFilter
Stable Diffusion 图像生成器 - Inpainting模式
只修改脸部表情，保持其他区域不变
"""
import torch
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Optional
from diffusers import StableDiffusionInpaintPipeline, UniPCMultistepScheduler

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import model_config


class StableDiffusionGenerator:
    def __init__(self):
        self.device = model_config.sd_device
        self.dtype = torch.bfloat16 if model_config.sd_dtype == "bfloat16" else torch.float16

        model_path = model_config.sd_local_dir if os.path.exists(model_config.sd_local_dir) else model_config.sd_model_id
        print(f"[SD] 加载: {model_path}")

        # 使用 inpaint 管道
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)

        print("[SD] 完成")

    def detect_faces_simple(self, image: Image.Image) -> List[tuple]:
        """
        简单人脸检测 - 使用颜色和位置启发式
        返回: [(x1, y1, x2, y2), ...]
        """
        w, h = image.size

        # 尝试用 mediapipe
        try:
            import mediapipe as mp
            mp_face = mp.solutions.face_detection
            with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as fd:
                img_np = np.array(image)
                results = fd.process(img_np)
                if results.detections:
                    faces = []
                    for det in results.detections:
                        bb = det.location_data.relative_bounding_box
                        x1 = max(0, int(bb.xmin * w) - 20)
                        y1 = max(0, int(bb.ymin * h) - 20)
                        x2 = min(w, int((bb.xmin + bb.width) * w) + 20)
                        y2 = min(h, int((bb.ymin + bb.height) * h) + 20)
                        faces.append((x1, y1, x2, y2))
                    return faces
        except ImportError:
            pass

        # 降级：假设人脸在图片中上方
        center_x, center_y = w // 2, h // 3
        face_w, face_h = w // 3, h // 3
        return [(center_x - face_w//2, center_y - face_h//2,
                 center_x + face_w//2, center_y + face_h//2)]

    def create_face_mask(self, image: Image.Image, faces: List[tuple]) -> Image.Image:
        """创建脸部区域的mask"""
        w, h = image.size
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        for (x1, y1, x2, y2) in faces:
            # 扩大区域，确保覆盖完整
            pad = int((x2 - x1) * 0.2)
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            draw.rectangle([x1, y1, x2, y2], fill=255)

        # 高斯模糊边缘，让过渡更自然
        mask = mask.filter(ImageFilter.GaussianBlur(radius=10))
        return mask

    def generate_candidates(
        self,
        prompt: str,
        negative_prompt: str,
        init_image: Image.Image = None,
        strength: float = 0.65,
        num_candidates: int = 3,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
    ) -> List[Image.Image]:
        """生成候选图"""
        num_candidates = num_candidates or model_config.num_candidates
        num_inference_steps = num_inference_steps or model_config.num_inference_steps
        guidance_scale = guidance_scale or model_config.guidance_scale

        generator = torch.Generator(device=self.device)
        candidates = []

        if init_image is not None:
            init_image = init_image.convert("RGB")
            orig_size = init_image.size
            init_resized = init_image.resize((width, height), Image.LANCZOS)

            # 检测人脸并创建mask
            faces = self.detect_faces_simple(init_resized)
            print(f"[SD] 检测到 {len(faces)} 张人脸")
            mask = self.create_face_mask(init_resized, faces)

            for i in range(num_candidates):
                print(f"[SD] 生成第 {i+1}/{num_candidates} 张...")
                generator.manual_seed(i * 42 + 100)

                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=init_resized,
                    mask_image=mask,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )

                # 缩放回原图大小
                out = result.images[0].resize(orig_size, Image.LANCZOS)
                candidates.append(out)
        else:
            for i in range(num_candidates):
                print(f"[SD] 生成第 {i+1}/{num_candidates} 张...")
                generator.manual_seed(i * 42 + 100)
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    width=width,
                    height=height,
                )
                candidates.append(result.images[0])

        return candidates


# 需要导入 ImageFilter
from PIL import ImageFilter
