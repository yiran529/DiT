"""
Single-GPU training script for DiT on CIFAR-10 with optional class filtering.
- Supports selecting a subset of classes via --classes (e.g., 0 1 2).
- No DDP; intended for single-node, single-GPU training.
"""
import argparse
import json
import logging
import os
from copy import deepcopy
from time import time

import yaml

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from diffusion import create_diffusion
from models import DiT_models


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def create_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(log_dir, "log.txt"))],
    )
    return logging.getLogger(__name__)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def load_config(config_path):
    with open(config_path, "r") as f:
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            return yaml.safe_load(f)
        return json.load(f)


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

def make_cifar10_dataloader(data_path, image_size, batch_size, num_workers, selected_classes=None):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )

    dataset = CIFAR10(root=data_path, train=True, download=True, transform=transform)

    if selected_classes:
        class_set = set(selected_classes)
        indices = [i for i, (_, y) in enumerate(dataset) if y in class_set]
        dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, len(dataset)


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def main(args):
    assert torch.cuda.is_available(), "CUDA is required for training."
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    logger = create_logger(args.results_dir)

    # Model setup (directly on images, no VAE)
    assert args.image_size % args.patch_size == 0, "image_size must be divisible by patch_size."
    model = DiT_models[args.model](
        input_size=args.image_size,
        num_classes=args.num_classes,
        num_latents_basic=args.num_latents_basic,
        num_latents_optional=args.num_latents_optional,
        cross_attn_interval=args.cross_attn_interval,
        optional_weight_temperature=args.optional_weight_temperature,
        optional_weight_bias=args.optional_weight_bias,
        optional_target_mean=args.optional_target_mean,
        patch_size=args.patch_size,
    )
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = model.to(device)
    diffusion = create_diffusion(timestep_respacing="")

    logger.info(f"Model: {args.model}, Params: {sum(p.numel() for p in model.parameters()):,}")

    # Data
    selected_classes = args.classes if args.classes else []
    loader, num_samples = make_cifar10_dataloader(
        args.data_path,
        args.image_size,
        args.batch_size,
        args.num_workers,
        selected_classes,
    )
    logger.info(f"Dataset size: {num_samples:,} (classes={selected_classes if selected_classes else 'all'})")

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    # Training loop
    train_steps = 0
    log_steps = 0
    running_loss = 0.0
    start_time = time()
    best_loss = float("inf")
    patience_ctr = 0
    stop_training = False

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            # direct images (no VAE)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            aux = getattr(model, "last_optional_aux_loss", None)
            if aux is not None:
                loss = loss + args.optional_aux_weight * aux

            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = running_loss / log_steps
                logger.info(
                    f"(epoch={epoch}, step={train_steps}) loss={avg_loss:.4f}, steps/sec={steps_per_sec:.2f}, weight_mean={getattr(model, 'last_optional_weight_mean', torch.tensor(float('nan')))}, aux={float('nan') if getattr(model, 'last_optional_aux_loss', None) is None else float(getattr(model, 'last_optional_aux_loss').detach().cpu().item()):.4f}"
                )

                if avg_loss < best_loss - args.early_stop_min_delta:
                    best_loss = avg_loss
                    patience_ctr = 0
                    best_path = os.path.join(args.results_dir, "ckpt_best.pt")
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "args": vars(args),
                            "step": train_steps,
                            "best_loss": best_loss,
                        },
                        best_path,
                    )
                    logger.info(f"New best loss {best_loss:.4f}; saved best checkpoint to {best_path}")
                else:
                    patience_ctr += 1
                    if patience_ctr >= args.early_stop_patience:
                        logger.info(
                            f"Early stopping triggered at step {train_steps}: no improvement for {args.early_stop_patience} log intervals"
                        )
                        stop_training = True

                running_loss = 0.0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0:
                ckpt_path = os.path.join(args.results_dir, f"ckpt_{train_steps:07d}.pt")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                logger.info(f"Saved checkpoint to {ckpt_path}")

            if stop_training:
                break

        if stop_training:
            break

    logger.info("Done!")


if __name__ == "__main__":
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config file")

    def build_parser():
        parser = argparse.ArgumentParser(parents=[base_parser])
        parser.add_argument("--data-path", type=str, required=True, help="CIFAR-10 root directory")
        parser.add_argument("--results-dir", type=str, default="results_cifar10")
        parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-B/2")
        parser.add_argument("--image-size", type=int, default=32)
        parser.add_argument("--num-classes", type=int, default=10)
        parser.add_argument("--epochs", type=int, default=200)
        parser.add_argument("--batch-size", type=int, default=64)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--patch-size", type=int, default=2)
        parser.add_argument("--num-workers", type=int, default=4)
        parser.add_argument("--log-every", type=int, default=100)
        parser.add_argument("--ckpt-every", type=int, default=10_000)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--num-latents-basic", type=int, default=32)
        parser.add_argument("--num-latents-optional", type=int, default=32)
        parser.add_argument("--cross-attn-interval", type=int, default=4)
        parser.add_argument("--optional-weight-temperature", type=float, default=1.0)
        parser.add_argument("--optional-weight-bias", type=float, default=0.0)
        parser.add_argument("--optional-target-mean", type=float, default=0.5)
        parser.add_argument("--optional-aux-weight", type=float, default=0.1, help="Weight for optional latent aux loss")
        parser.add_argument(
            "--classes",
            type=int,
            nargs="*",
            default=None,
            help="List of CIFAR-10 class indices to train on (e.g., 0 1 2). If omitted, use all classes.",
        )
        parser.add_argument(
            "--early-stop-patience",
            type=int,
            default=10,
            help="#log intervals without improvement before stopping",
        )
        parser.add_argument(
            "--early-stop-min-delta",
            type=float,
            default=0.0,
            help="Minimum loss improvement to reset patience",
        )
        return parser

    config_parser = argparse.ArgumentParser(parents=[base_parser])
    config_args, _ = config_parser.parse_known_args()

    parser = build_parser()
    if config_args.config:
        cfg = load_config(config_args.config)
        parser.set_defaults(**cfg)

    args = parser.parse_args()
    main(args)
