from PIL import Image
from diffusers import AutoPipelineForText2Image
from diffusers.schedulers import DDIMScheduler
import torch

scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.0120, beta_schedule="scaled_linear", clip_sample=False, steps_offset=1)
pipeline = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True,
    scheduler=scheduler,
).to("cuda")

batch_size = 4
# prompt = "A turtle playing with a ball"
prompt = "photo of a cat riding on a bicycle"
# prompt = "peasant and dragon combat, wood cutting style, viking era, bevel with rune"

imgs = pipeline([prompt] * batch_size, num_inference_steps=25).images

w, h = imgs[0].size
image = Image.new("RGB", (batch_size * w, h))
for i, img in enumerate(imgs):
    image.paste(imgs[i], (i * w, 0))
image.save("sample_output.png")
