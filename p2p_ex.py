import sys
import os
import argparse
from pathlib import Path
from diffusers.schedulers import DDIMScheduler
from diffusers import DiffusionPipeline
import torch
from torchvision.utils import save_image

from experiments.nullinversion.pipeline import NullTextPipeline
from experiments import utils


CACHE_FILE_NAME = "cache.pt"

scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.0120, beta_schedule="scaled_linear", clip_sample=False, steps_offset=1)

prompts = ["a girl, pink hair, blue eyes, red ribbon",
           "a girl, pink hair, blue eyes, black ribbon"]


def nulltext_inversion(cfg):
    cache_path = cfg.output_dir / CACHE_FILE_NAME
    if cache_path.exists():
        cache = torch.load(cache_path)
        inverted_latent = cache["latent"]
        uncond_embeddings = cache["embeddings"]
    else:
        pipeline = NullTextPipeline.from_pretrained(
            cfg.model_path,
            scheduler=scheduler,
            torch_dtype=torch.float32,
        ).to(cfg.device)

        base_prompt = prompts[0]
        inverted_latent, uncond_embeddings = pipeline.invert(str(cfg.image_path), base_prompt, num_inner_steps=10, early_stop_epsilon=1e-5, num_inference_steps=cfg.n_steps)
        torch.save({
            "latent": inverted_latent,
            "embeddings": uncond_embeddings,
        }, cache_path)

        result = pipeline(base_prompt, uncond_embeddings, inverted_latent, guidance_scale=cfg.guidance_scale, num_inference_steps=cfg.n_steps)
        result.images[0].save(cfg.output_dir / "inverted.png")

    return inverted_latent, uncond_embeddings


def ddim_inversion(cfg):
    pipeline = NullTextPipeline.from_pretrained(
        cfg.model_path,
        scheduler=scheduler,
        torch_dtype=torch.float32,
    ).to(cfg.device)

    base_prompt = prompts[0]
    inverted_latents, uncond_embeddings = pipeline.ddim_inversion(str(cfg.image_path), base_prompt, num_inference_steps=cfg.n_steps)

    inverted_latent = inverted_latents[-1]
    result = pipeline(base_prompt, uncond_embeddings, inverted_latent, start_step=0, guidance_scale=cfg.guidance_scale, num_inference_steps=cfg.n_steps)
    result.images[0].save(cfg.output_dir / f"inverted.png")

    return inverted_latent, uncond_embeddings


def main(cfg):
    if cfg.inversion_type == "ddim":
        inverted_latent, uncond_embeddings = ddim_inversion(cfg)
    else:
        inverted_latent, uncond_embeddings = nulltext_inversion(cfg)

    output_dir = cfg.output_dir / cfg.output_image_dir_name
    os.makedirs(output_dir, exist_ok=True)
    (output_dir / "prompts.txt").write_text("\n".join(prompts))

    p2p_pipe = DiffusionPipeline.from_pretrained(
        cfg.model_path,
        custom_pipeline="./experiments/p2p/",
        scheduler=scheduler,
        torch_dtype=torch.float16 if cfg.fp16 else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(cfg.device)

    generator = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
    resolution = 512
    layer_min = 4
    layer_max = 14
    layer_ids = range(layer_min, layer_max + 1)

    cross_attention_kwargs = {
        "edit_type": "replace",
        "n_cross_replace": 0.0,
        "local_blend_words": [["hair,"], ["hair,"]],
        # "local_blend_words": [["long ", "blue"], ["long", "blond"], ["short", "blue"]],
        "layer_ids": layer_ids,
    }

    def generate(n_self_replace, n_lb_threshold):
        kwargs = cross_attention_kwargs.update({
            "n_self_replace": n_self_replace * 0.1,
            "local_blend_threshold": n_lb_threshold * 0.1,
        })

        print(f"generate with self_replace: {n}, lb_threshold: {n_lb_threshold}")

        with torch.autocast(device_type=cfg.device_name, dtype=torch.float16, enabled=cfg.fp16):
            result = p2p_pipe(
                prompt=prompts,
                latents=inverted_latent.repeat(len(prompts), 1, 1, 1),
                negative_prompt_embeds_by_timesteps=uncond_embeddings,
                height=resolution,
                width=resolution,
                num_inference_steps=cfg.n_steps,
                generator=generator,
                output_type="pt",
                cross_attention_kwargs=cross_attention_kwargs
            )

        name = f"cross00_self{n:02}_layer{layer_min}-{layer_max}"
        if "local_blend_words" in cross_attention_kwargs:
            word = cross_attention_kwargs["local_blend_words"][0][0]
            name += f"_lb-{word}{n_lb_threshold:02}"
        save_image(result.images, output_dir / f"{name}.png")

        if cfg.save_separated:
            sub_dir = output_dir / name
            os.makedirs(sub_dir, exist_ok=True)
            for i, img in enumerate(result.images):
                save_image(img, sub_dir / f"{i}.png")

        for i in len(prompts):
            utils.show_cross_attention(p2p_pipe, prompts, res=resolution//16, from_where=("up", "down"), select=i, output_dir=cfg.output_dir)
            utils.show_cross_attention(p2p_pipe, prompts, res=resolution//32, from_where=("up", "down"), select=i, output_dir=cfg.output_dir)


    for n in cfg.self_replaces:
        for lb_threshold in cfg.lb_thresholds:
            generate(n, lb_threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--cpu", action="store_true", help="use cpu")
    parser.add_argument("--model_path", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=8888)
    parser.add_argument("--output_dir", type=Path, default="./output")
    parser.add_argument("--output_image_dir_name", type=Path, default="default")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--inversion_type", default="nulltext", choices=["ddim", "nulltext"])
    parser.add_argument("--save_separated", action="store_true")
    parser.add_argument("--self_replaces", type=int, nargs="*", default=[3, 4, 5, 6, 7, 8])
    parser.add_argument("--lb_thresholds", type=int, nargs="*", default=[3])
    args = parser.parse_args()

    args.device_name = "cpu" if args.cpu else "cuda"
    args.device = torch.device(args.device_name)

    args.output_dir = args.output_dir / args.image_path.stem
    os.makedirs(args.output_dir, exist_ok=True)

    print(args)
    main(args)