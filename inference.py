"""NitroFusion2 text-to-image with diffusers (1-4 steps).

    python inference.py --steps 2 --out example.png
"""
import argparse

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, ZImagePipeline, ZImageTransformer2DModel

BASE = "Tongyi-MAI/Z-Image-Turbo"
REPO = "ChenDY/NitroFusion2"
DEDISTILL_LORA = "nf2-dedistill-lora.safetensors"
SHIFT = {1: 3.0, 2: 6.0, 3: 8.0, 4: 9.0}
LORA_SCALES = {1: (0.0,), 2: (0.5, 0.0), 3: (0.5, 0.6, 0.0), 4: (0.5, 0.6, 0.7, 0.0)}


def load(repo: str = REPO, dedistill_lora: bool = True) -> ZImagePipeline:
    transformer = ZImageTransformer2DModel.from_pretrained(repo, subfolder="transformer", torch_dtype=torch.bfloat16)
    pipe = ZImagePipeline.from_pretrained(BASE, transformer=transformer, torch_dtype=torch.bfloat16)
    if dedistill_lora:
        pipe.load_lora_weights(repo, weight_name=DEDISTILL_LORA, adapter_name="dedistill")
    return pipe


@torch.no_grad()
def generate(pipe: ZImagePipeline, prompt: str, steps: int = 2, height: int = 1024, width: int = 1024,
             seed: int = 0, dedistill_lora: bool = True):
    assert steps in SHIFT, "NitroFusion2 is distilled for 1-4 steps"
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, shift=SHIFT[steps])
    sigmas = [1.0 - i / steps for i in range(steps)]
    scales = LORA_SCALES[steps] if dedistill_lora else (0.0,) * steps
    callback = None
    if dedistill_lora:
        pipe.set_adapters("dedistill", scales[0])

        def callback(p, i, t, kwargs):
            if i + 1 < steps:
                p.set_adapters("dedistill", scales[i + 1])
            return kwargs

    image = pipe(prompt=prompt, height=height, width=width, num_inference_steps=steps, sigmas=sigmas,
                 guidance_scale=0.0, generator=torch.Generator("cuda").manual_seed(seed),
                 callback_on_step_end=callback).images[0]
    return image


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default='Latina female with thick wavy hair, harbor boats and pastel houses behind. Breezy seaside light, warm tones, cinematic close-up.')
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-dedistill-lora", action="store_true")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--out", default="nitrofusion2.png")
    args = ap.parse_args()
    pipe = load(args.repo, dedistill_lora=not args.no_dedistill_lora)
    pipe.enable_model_cpu_offload()
    generate(pipe, args.prompt, args.steps, args.height, args.width, args.seed, dedistill_lora=not args.no_dedistill_lora).save(args.out)
    print("saved", args.out)
