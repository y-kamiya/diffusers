import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from diffusers import DiffusionPipeline
from experiments import utils


config = {
    "sd15": ("runwayml/stable-diffusion-v1-5", 512),
    "sd21": ("stabilityai/stable-diffusion-2-1", 768),
}
model_type = "sd15"
resolution = config[model_type][1]

prompts = ["A painting of a squirrel eating a burger",
           "A painting of a lion eating a burger"]

cross_attention_kwargs = {
    "edit_type": "replace",
    "cross_replace_steps": 0.8,
    "self_replace_steps": 0.4,
    "local_blend_words": ["squirrel ", "lion"],
}
generator = torch.Generator(device="cuda").manual_seed(8888)


pipe = DiffusionPipeline.from_pretrained(
    config[model_type][0],
    custom_pipeline="./experiments/p2p/",
    # custom_pipeline="pipeline_prompt2prompt",
    use_safetensors=True,
).to("cuda")

outputs = pipe(
    prompt=prompts,
    height=resolution,
    width=resolution,
    num_inference_steps=25,
    generator=generator,
    cross_attention_kwargs=cross_attention_kwargs
)

imgs = outputs[0]
w, h = imgs[0].size
image = Image.new("RGB", (2 * w, h))
image.paste(imgs[0], (0, 0))
image.paste(imgs[1], (w, 0))
image.save("p2p_output.png")


for n in [16, 32, 64]:
    ca_size = resolution // n
    if ca_size <= 32:
        utils.show_cross_attention(pipe, prompts, res=ca_size, from_where=("up", "down"), select=0)
