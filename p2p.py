import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from diffusers import DiffusionPipeline


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


def show_cross_attention(pipe, prompts, res: int, from_where, select: int = 0):
    tokens = pipe.tokenizer.encode(prompts[select])
    decoder = pipe.tokenizer.decode
    attention_maps = aggregate_attention(prompts, pipe.controller, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    tiled_img = create_tiled_image(np.stack(images, axis=0))
    tiled_img.save(f"p2p_ca_{res}_p{select}.png")


def aggregate_attention(prompts, attention_store, res: int, from_where, is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            print(item.shape)
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def text_under_image(image, text: str, text_color = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def create_tiled_image(images, num_rows: int = 1, offset_ratio: float = 0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    return pil_img


for n in [16, 32, 64]:
    ca_size = resolution // n
    if ca_size <= 32:
        show_cross_attention(pipe, prompts, res=ca_size, from_where=("up", "down"), select=0)
