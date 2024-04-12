import sys
import os
import argparse
from pathlib import Path
from diffusers.schedulers import DDIMScheduler
from examples.community.pipeline_null_text_inversion import NullTextPipeline
from diffusers import DiffusionPipeline
import torch


CACHE_FILE_NAME = "cache.pt"


def nulltext_inversion(cfg):
    cache_path = cfg.output_dir / CACHE_FILE_NAME
    if cache_path.exists():
        cache = torch.load(cache_path)
        inverted_latent = cache["latent"]
        uncond_embeddings = cache["embeddings"]
    else:
        scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.0120, beta_schedule="scaled_linear")
        pipeline = NullTextPipeline.from_pretrained(
            cfg.model_path,
            scheduler=scheduler,
            torch_dtype=torch.float32,
        ).to(cfg.device)

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

    prompts = ["a girl, pink hair, blue eyes, red ribbon",
               "a girl, black hair, blue eyes, red ribbon"]

    p2p_pipe = DiffusionPipeline.from_pretrained(
        cfg.model_path,
        custom_pipeline="pipeline_prompt2prompt",
    ).to(cfg.device)

    cross_attention_kwargs = {
        "edit_type": "replace",
        "cross_replace_steps": 0.8,
        "self_replace_steps": 0.4,
        "local_blend_words": ["pink ", "black"],
    }
    generator = torch.Generator(device=cfg.device).manual_seed(cfg.seed)

    result = p2p_pipe(
        prompt=prompts,
        latents=inverted_latent,
        negative_prompt_embeds=uncond_embeddings,
        height=512,
        width=512,
        num_inference_steps=cfg.n_steps,
        generator=generator,
        cross_attention_kwargs=cross_attention_kwargs
    )
    result.images[0].save(cfg.output_dir / "edited.png")


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
