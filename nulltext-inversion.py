import sys
import os
import argparse
from pathlib import Path
from diffusers.schedulers import DDIMScheduler
from examples.community.pipeline_null_text_inversion import NullTextPipeline
import torch


CACHE_FILE_NAME = "cache.pt"


def main(cfg):
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.0120, beta_schedule="scaled_linear")
    pipeline = NullTextPipeline.from_pretrained(
        cfg.model_path,
        scheduler=scheduler,
        torch_dtype=torch.float32,
    ).to(cfg.device)

    cache_path = cfg.output_dir / CACHE_FILE_NAME
    if cache_path.exists():
        cache = torch.load(cache_path)
        inverted_latent = cache["latent"]
        uncond_embeddings = cache["embeddings"]
    else:
        inverted_latent, uncond_embeddings = pipeline.invert(str(cfg.image_path), cfg.prompt, num_inner_steps=10, early_stop_epsilon=1e-5, num_inference_steps=cfg.n_steps)
        torch.save({
            "latent": inverted_latent,
            "embeddings": uncond_embeddings,
        }, cache_path)

    result = pipeline(cfg.prompt, uncond_embeddings, inverted_latent, guidance_scale=cfg.guidance_scale, num_inference_steps=cfg.n_steps)
    result.images[0].save(cfg.output_dir / "inverted.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("image_path", type=Path)
    parser.add_argument("prompt")
    parser.add_argument("--cpu", action="store_true", help="use cpu")
    parser.add_argument("--model_path", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--output_dir", type=Path, default="./output")
    args = parser.parse_args()

    args.device_name = "cpu" if args.cpu else "cuda"
    args.device = torch.device(args.device_name)

    args.output_dir = args.output_dir / args.image_path.stem
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
