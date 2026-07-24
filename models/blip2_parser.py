"""
BLIP-2 语义解析器 - 兼容新版transformers
"""
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import re
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import model_config


class BLIP2Parser:
    def __init__(self):
        self.device = model_config.blip2_device
        self.dtype = torch.bfloat16 if model_config.blip2_dtype == "bfloat16" else torch.float32

        model_path = model_config.blip2_local_dir if os.path.exists(model_config.blip2_local_dir) else model_config.blip2_model_id
        print(f"[BLIP-2] 加载: {model_path}")

        self.processor = Blip2Processor.from_pretrained(model_path)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_path, device_map="auto", torch_dtype=self.dtype
        )
        self.model.eval()
        print("[BLIP-2] 完成")

    @torch.no_grad()
    def generate_caption(self, image, max_length=77):
        """生成图像描述 - 使用兼容性更好的方式"""
        try:
            # 尝试标准方式
            inputs = self.processor(images=image, text="a photo of", return_tensors="pt")
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            if 'pixel_values' in inputs and inputs['pixel_values'].dtype != self.dtype:
                inputs['pixel_values'] = inputs['pixel_values'].to(self.dtype)

            outputs = self.model.generate(**inputs, max_length=max_length, num_beams=5)
            caption = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
            return re.sub(r'^a photo of\s*', '', caption, flags=re.IGNORECASE)
        except Exception as e:
            print(f"[BLIP-2] 标准方式失败: {e}")
            # 降级：只用pixel_values
            try:
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                if 'pixel_values' in inputs and inputs['pixel_values'].dtype != self.dtype:
                    inputs['pixel_values'] = inputs['pixel_values'].to(self.dtype)

                outputs = self.model.generate(**inputs, max_length=max_length, num_beams=5)
                caption = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
                return caption
            except Exception as e2:
                print(f"[BLIP-2] 降级方式也失败: {e2}")
                return "a scene"
