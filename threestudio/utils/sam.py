import argparse
import json
from pathlib import Path
from PIL import Image
import torch
from einops import rearrange
from torchvision.transforms import ToPILImage, ToTensor

from lang_sam import LangSAM

# from threestudio.utils.typing import *


class LangSAMTextSegmentor(torch.nn.Module):
    def __init__(self, sam_type="sam2.1_hiera_base_plus"):
        super().__init__()
        self.model = LangSAM(sam_type)

        self.to_pil_image = ToPILImage(mode="RGB")
        self.to_tensor = ToTensor()

    def forward(self, images, prompt: str):
        images = rearrange(images, "b h w c -> b c h w")
        masks = []
        for image in images:
            # breakpoint()
            image = self.to_pil_image(image.clamp(0.0, 1.0))
            all_results = self.model.predict([image], [prompt])
            mask = all_results[0]['masks']
            score_arr  = all_results[0]["mask_scores"]  # shape: (K,)
            box_scores  = all_results[0]["scores"]       # shape: (N_box,)

            if mask is None or (isinstance(mask, list) and len(mask) == 0):
                print(f"None {prompt} Detected")
                masks.append(torch.zeros_like(images[0, 0:1]))  # 添加空mask占位
                continue
            # breakpoint()
            if mask.ndim == 3:
                #masks.append(mask[0:1].to(torch.float32))
                mask_tensor = torch.tensor(mask[0:1], dtype=torch.float32)  # 先转换为 Tensor
                masks.append(mask_tensor)
                # best = score_arr.argmax()               # 取分最高的索引
                # best_mask = torch.tensor(
                #     mask[best : best + 1],          # 保持 (1, H, W) 形状
                #     dtype=torch.float32,
                # )
                # masks.append(best_mask)
            else:
                print(f"None {prompt} Detected")
                masks.append(torch.zeros_like(images[0, 0:1]))


        return torch.stack(masks, dim=0)


if __name__ == "__main__":
    model = LangSAMTextSegmentor()

    image = Image.open("load/lego_bulldozer.jpg")
    prompt = "a lego bulldozer"

    image = ToTensor()(image)

    image = image.unsqueeze(0)

    mask = model(image, prompt)

    breakpoint()
