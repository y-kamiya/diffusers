import sys
import os
import argparse
from pathlib import Path
from diffusers.schedulers import DDIMScheduler
from diffusers import DiffusionPipeline
import torch

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

        cfg.prompt = prompts[0]
        inverted_latent, uncond_embeddings = pipeline.invert(str(cfg.image_path), cfg.prompt, num_inner_steps=10, early_stop_epsilon=1e-5, num_inference_steps=cfg.n_steps)
        torch.save({
            "latent": inverted_latent,
            "embeddings": uncond_embeddings,
        }, cache_path)

        result = pipeline(cfg.prompt, uncond_embeddings, inverted_latent, guidance_scale=cfg.guidance_scale, num_inference_steps=cfg.n_steps)
        result.images[0].save(cfg.output_dir / "inverted.png")

    return inverted_latent, uncond_embeddings


def main(cfg):
    inverted_latent, uncond_embeddings = nulltext_inversion(cfg)

    # if cfg.edit_prompt is None:
    #     print("No edit prompt provided, skipping editing")
    #     return
    # prompts = [cfg.prompt, cfg.edit_prompt]

    (cfg.output_dir / "prompts.txt").write_text("\n".join(prompts))

    p2p_pipe = DiffusionPipeline.from_pretrained(
        cfg.model_path,
        custom_pipeline="./experiments/p2p/",
        scheduler=scheduler,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(cfg.device)

    layer_ids = range(4, 15)

    cross_attention_kwargs = {
        "edit_type": "replace",
        "n_cross_replace": 0.0,
        "n_self_replace": 0.8,
        # "local_blend_words": ["red ", "black"],
        "layer_ids": layer_ids,
    }
    resolution = 512
    generator = torch.Generator(device=cfg.device).manual_seed(cfg.seed)

    result = p2p_pipe(
        prompt=prompts,
        latents=inverted_latent.repeat(len(prompts), 1, 1, 1),
        negative_prompt_embeds_by_timesteps=uncond_embeddings,
        height=resolution,
        width=resolution,
        num_inference_steps=cfg.n_steps,
        generator=generator,
        output_type="np",
        cross_attention_kwargs=cross_attention_kwargs
    )
    img = utils.create_tiled_image(result.images * 255)
    img.save(cfg.output_dir / "edited.png")

    if layer_ids is None:
        utils.show_cross_attention(p2p_pipe, prompts, res=resolution//16, from_where=("up", "down"), select=0, output_dir=cfg.output_dir)
        utils.show_cross_attention(p2p_pipe, prompts, res=resolution//32, from_where=("up", "down"), select=0, output_dir=cfg.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--edit_prompt", default=None)
    parser.add_argument("--cpu", action="store_true", help="use cpu")
    parser.add_argument("--model_path", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=8888)
    parser.add_argument("--output_dir", type=Path, default="./output")
    args = parser.parse_args()

    args.device_name = "cpu" if args.cpu else "cuda"
    args.device = torch.device(args.device_name)

    args.output_dir = args.output_dir / args.image_path.stem
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
