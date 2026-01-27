"""
Sample CIFAR-10 images from a DiT checkpoint trained in image space.
"""
import argparse
import json
from pathlib import Path

import torch
from torchvision.utils import save_image
import yaml

from diffusion import create_diffusion
from models import DiT_models


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as handle:
        if config_path.endswith((".yaml", ".yml")):
            return yaml.safe_load(handle)
        return json.load(handle)


def prepare_output_path(out_path: str) -> Path:
    path = Path(out_path)
    if path.suffix == "":
        path.mkdir(parents=True, exist_ok=True)
        path = path / "samples.png"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def override_from_checkpoint(args, ckpt_args):
    override_fields = [
        "model",
        "image_size",
        "patch_size",
        "num_classes",
        "num_latents_basic",
        "num_latents_optional",
        "cross_attn_interval",
        "optional_weight_temperature",
        "optional_weight_bias",
        "optional_target_mean",
    ]
    for field in override_fields:
        if field in ckpt_args:
            setattr(args, field, ckpt_args[field])


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.ckpt is None:
        raise ValueError("--ckpt is required for CIFAR-10 sampling.")

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    override_from_checkpoint(args, ckpt_args)

    model = DiT_models[args.model](
        input_size=args.image_size,
        num_classes=args.num_classes,
        num_latents_basic=args.num_latents_basic,
        num_latents_optional=args.num_latents_optional,
        cross_attn_interval=args.cross_attn_interval,
        optional_weight_temperature=args.optional_weight_temperature,
        optional_weight_bias=args.optional_weight_bias,
        optional_target_mean=args.optional_target_mean,
        # patch_size=args.patch_size,
    ).to(device)

    state_dict = None
    if args.use_ema and "ema" in checkpoint:
        state_dict = checkpoint["ema"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        raise KeyError("Checkpoint does not contain 'ema' or 'model' weights.")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch. Missing: {missing}, unexpected: {unexpected}")
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))

    class_ids = list(args.class_ids)
    if len(class_ids) == 0:
        raise ValueError("Provide at least one class id via --class-ids.")
    if args.batch_size is not None and args.batch_size > 0:
        if len(class_ids) < args.batch_size:
            repeats = (args.batch_size + len(class_ids) - 1) // len(class_ids)
            class_ids = (class_ids * repeats)[: args.batch_size]
        elif len(class_ids) > args.batch_size:
            class_ids = class_ids[: args.batch_size]
    batch_size = len(class_ids)

    labels = torch.tensor(class_ids, device=device, dtype=torch.long)
    noise = torch.randn(batch_size, 3, args.image_size, args.image_size, device=device)

    noise_input = torch.cat([noise, noise], dim=0)
    null_labels = torch.full((batch_size,), args.num_classes, device=device, dtype=torch.long)
    labels_input = torch.cat([labels, null_labels], dim=0)
    model_kwargs = dict(y=labels_input, cfg_scale=args.cfg_scale)

    with torch.no_grad():
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg,
            noise_input.shape,
            noise_input,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=device,
        )
    samples, _ = samples.chunk(2, dim=0)

    out_path = prepare_output_path(args.out_path)
    save_image(samples, str(out_path), nrow=int(batch_size ** 0.5) or 1, normalize=True, value_range=(-1, 1))
    print(f"Saved CIFAR-10 samples to {out_path}")


if __name__ == "__main__":
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", type=str, default=None, help="Optional JSON/YAML config file")

    parser = argparse.ArgumentParser(parents=[base_parser])
    parser.add_argument("--ckpt", type=str, required=True, help="Path to training checkpoint")
    parser.add_argument("--out-path", type=str, default="results_cifar10/samples.png", help="Output image path or directory")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-B/2")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--num-latents-basic", type=int, default=24)
    parser.add_argument("--num-latents-optional", type=int, default=8)
    parser.add_argument("--cross-attn-interval", type=int, default=4)
    parser.add_argument("--optional-weight-temperature", type=float, default=1.0)
    parser.add_argument("--optional-weight-bias", type=float, default=0.0)
    parser.add_argument("--optional-target-mean", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=None, help="Target number of samples to generate")
    parser.add_argument("--class-ids", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7], help="Class ids to sample")
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-ema", action="store_true", default=True, help="Use EMA weights when available")
    parser.add_argument("--no-ema", dest="use_ema", action="store_false", help="Disable EMA weights")

    config_parser = argparse.ArgumentParser(parents=[base_parser])
    config_args, _ = config_parser.parse_known_args()

    if config_args.config:
        cfg = load_config(config_args.config)
        parser.set_defaults(**cfg)

    args = parser.parse_args()
    if args.config and not Path(args.config).exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    main(args)
