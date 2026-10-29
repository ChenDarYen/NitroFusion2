# NitroFusion2: Bringing Adversarial Loss to the Text Stream of Multimodal Diffusion Transformers

[![Checkpoint](https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoint-NitroFusion2-ffc107?labelColor=555)](https://huggingface.co/ChenDY/NitroFusion2)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-NitroFusion2-ff7a00?labelColor=555)](https://huggingface.co/spaces/ChenDY/NitroFusion2)
[![Project Page](https://img.shields.io/badge/%F0%9F%8C%90%20Project%20Page-NitroFusion2-1f72b8?labelColor=555)](https://chendaryen.github.io/NitroFusion2.github.io/)
![Paper](https://img.shields.io/badge/%F0%9F%93%84%20Paper-soon-9e9e9e?labelColor=555)

![](./assets/banner.jpg)

## 🔥 News

- [2026-10-29] NitroFusion2 is released! The checkpoint is on [Hugging Face](https://huggingface.co/ChenDY/NitroFusion2), with a [ComfyUI workflow](#comfyui), [diffusers](#diffusers) code and an [online demo](https://huggingface.co/spaces/ChenDY/NitroFusion2).

## ✨ NitroFusion2

NitroFusion2 is a 1–2 step text-to-image model built on [Z-Image-Turbo](https://github.com/Tongyi-MAI/Z-Image). It moves the adversarial loss onto the text stream of the MMDiT. At one step it renders sharp text and fine detail.

![](./assets/compare_book_dancer.jpg)

<p align="center"><img src="./assets/compare_opera.jpg" width="80%"></p>

## ComfyUI

- 1 step: [`workflows/nitrofusion2_1step.json`](./workflows/nitrofusion2_1step.json)
- 2 steps: [`workflows/nitrofusion2_2step.json`](./workflows/nitrofusion2_2step.json)

For two steps, we recommend our de-distillation LoRA, which takes NitroFusion2 back to Z-Image-Turbo, to eliminate some artifacts. The workflow applies it at scale 0.5 on the first step, then runs NitroFusion2 alone on the second step.

We also release [GGUF](https://huggingface.co/ChenDY/NitroFusion2/tree/main) tiers from `nf2-Q2_K_4XS.gguf` (2.0 GB) to `nf2-Q8_0.gguf` (7.2 GB).

GGUF workflows: [1 step](./workflows/nitrofusion2_1step_gguf.json), [2 steps](./workflows/nitrofusion2_2step_gguf.json).

![](./assets/compare_gguf_latina.jpg)

## Diffusers

Run [`inference.py`](./inference.py) (1–4 steps):

```bash
python inference.py --prompt "a cat holding a sign that says hello world" --steps 2 --out cat.png
```

Or in Python:

```python
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, ZImagePipeline, ZImageTransformer2DModel

transformer = ZImageTransformer2DModel.from_pretrained(
    "ChenDY/NitroFusion2",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.load_lora_weights("ChenDY/NitroFusion2", weight_name="nf2-dedistill-lora.safetensors", adapter_name="dedistill")
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, shift=6.0)
pipe.enable_model_cpu_offload()

pipe.set_adapters("dedistill", 0.5)
def lora_off(pipe, step, timestep, callback_kwargs):
    pipe.set_adapters("dedistill", 0.0)
    return callback_kwargs

prompt = "Latina female with thick wavy hair, harbor boats and pastel houses behind. Breezy seaside light, warm tones, cinematic close-up."

image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=2,
    sigmas=[1.0, 0.5],
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(42),
    callback_on_step_end=lora_off,
).images[0]

image.save("example.png")
```

## Method

> If this paper has one message, it is this: in an MMDiT, the text stream is no longer only conditioning. Once it reads the image, it becomes a discriminative representation, opening a new direction for adversarial diffusion distillation and allowing that discrimination to be steered toward specific aspects.

![](./assets/architecture.jpg)

In an MMDiT, text and image tokens move through the same attention layers, so by the middle of the network the text tokens have already read the image. They hold a compact summary of what was generated, organised by what the prompt asked for. NitroFusion2 puts an adversarial loss there, on that summary, alongside the conventional one on the image features. Because the summary stays tied to the text condition, changing the condition changes what it reports — so one frozen discriminator can be asked a different question simply by swapping the cue. A typography cue makes it attend to rendered text, an object cue to the things named, a style cue to how the image looks. One backbone, many critics, at no extra cost.

## Citation

The paper is coming soon. Until then, if you find NitroFusion2 useful or relevant to your research, please kindly cite this release:

```bib
@misc{chen2026nitrofusion2,
    title={NitroFusion2: Bringing Adversarial Loss to the Text Stream of Multimodal Diffusion Transformers},
    author={Dar-Yen Chen and Yi-Zhe Song},
    year={2026},
    howpublished={\url{https://github.com/ChenDarYen/NitroFusion2}},
    note={Paper coming soon}
}
```

Please also consider citing NitroFusion:

```bib
@inproceedings{chen2025nitrofusion,
    title={NitroFusion: High-Fidelity Single-Step Diffusion through Dynamic Adversarial Training},
    author={Dar-Yen Chen and Hmrishav Bandyopadhyay and Kai Zou and Yi-Zhe Song},
    booktitle={CVPR},
    year={2025}
}
```
